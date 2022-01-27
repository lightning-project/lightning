use crate::prelude::*;
use crate::types::{ByteStrides, DeviceId, Dim, MemoryKind, MAX_DEVICES, MAX_DIMS};
use cuda_driver_sys::cudaError_enum::CUDA_ERROR_INVALID_VALUE;
use cuda_driver_sys::{
    cuMemcpy3DPeerAsync, cuMemcpyDtoDAsync_v2, cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2,
    cuMemcpyPeerAsync, CUmemorytype_enum, CUDA_MEMCPY3D_PEER,
};
use lightning_cuda::prelude::*;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::Arc;

#[derive(Debug)]
struct Device {
    context: CudaContextHandle,
    d2h_lo: CudaStream,
    h2d_lo: CudaStream,
    d2h_hi: CudaStream,
    h2d_hi: CudaStream,
}

#[derive(Debug)]
struct Inner {
    streams: Vec<Device>,
    peer_supported: [[bool; MAX_DEVICES]; MAX_DEVICES],
}

#[derive(Clone, Debug)]
pub(crate) struct CopyEngine {
    inner: Arc<Inner>,
}

impl CopyEngine {
    pub(crate) fn new(contexts: Vec<CudaContextHandle>) -> CudaResult<Self> {
        let mut peer_supported = [[false; MAX_DEVICES]; MAX_DEVICES];
        let mut streams = vec![];

        if contexts.len() > 1 {
            for context in &contexts {
                if let Err(e) = context.enable_peer_access() {
                    warn!(
                        "cannot enable peer access for device {:?}: {}",
                        context.device()?,
                        e
                    );
                }
            }
        }

        for context in contexts {
            context.try_with(|| {
                streams.push(Device {
                    context,
                    d2h_lo: CudaStream::new()?,
                    h2d_lo: CudaStream::new()?,
                    d2h_hi: CudaStream::new()?,
                    h2d_hi: CudaStream::new()?,
                });

                Ok(())
            })?;
        }

        for (i, a) in enumerate(&streams) {
            for (j, b) in enumerate(&streams) {
                let supported = if i != j {
                    let left = a.context.device()?;
                    let right = b.context.device()?;

                    CudaDevice::can_access_peer(&left, right)?
                } else {
                    true
                };

                peer_supported[i][j] = supported;
            }
        }

        let inner = Arc::new(Inner {
            streams,
            peer_supported,
        });

        Ok(Self { inner })
    }

    unsafe fn submit<F, G>(
        &self,
        src: MemoryKind,
        dst: MemoryKind,
        nbytes: usize,
        complete: F,
        callback: G,
    ) where
        F: FnOnce(CudaResult) + Send + 'static,
        G: FnOnce(&CudaStream) -> CudaResult,
    {
        // no bytes, nothing to do!
        if nbytes == 0 {
            return complete(Ok(()));
        }

        let high_priority = nbytes < 1024;
        let streams = &*self.inner.streams;

        let context = match (src, dst) {
            (MemoryKind::Device(src), _) => &streams[src.get()].context,
            (_, MemoryKind::Device(dst)) => &streams[dst.get()].context,
            _ => panic!("unsupported copy: {:?} -> {:?}", src, dst),
        };

        let stream = match (src, dst) {
            (MemoryKind::Device(src), _) => match high_priority {
                true => &streams[src.get()].d2h_hi,
                false => &streams[src.get()].d2h_lo,
            },
            (_, MemoryKind::Device(dst)) => match high_priority {
                true => &streams[dst.get()].h2d_hi,
                false => &streams[dst.get()].h2d_lo,
            },
            _ => panic!("unsupported copy: {:?} -> {:?}", src, dst),
        };

        context
            .with(|| {
                let result_submit = callback(stream);

                stream.add_callback(move |result_sync| {
                    let result = CudaResult::and(result_submit, result_sync);
                    complete(result)
                });
            })
            .unwrap();
    }
    pub(crate) fn supported_d2d(&self, src_id: DeviceId, dst_id: DeviceId) -> bool {
        self.inner.peer_supported[src_id.get()][dst_id.get()]
    }

    pub(crate) unsafe fn copy_h2d<F>(
        &self,
        src_ptr: *const (),
        dst_id: DeviceId,
        dst_ptr: CudaDevicePtr,
        nbytes: usize,
        complete: F,
    ) where
        F: FnOnce(CudaResult) + Send + 'static,
    {
        self.submit(
            MemoryKind::Host,
            MemoryKind::Device(dst_id),
            nbytes,
            complete,
            move |stream| {
                cuda_check(cuMemcpyHtoDAsync_v2(
                    dst_ptr.raw(),
                    src_ptr as *const c_void,
                    nbytes,
                    stream.raw(),
                ))
            },
        );
    }

    pub(crate) unsafe fn copy_h2d_strided<F>(
        &self,
        src_ptr: *const (),
        src_strides: ByteStrides,
        dst_id: DeviceId,
        dst_ptr: CudaDevicePtr,
        dst_strides: ByteStrides,
        counts: Dim,
        elem_size: usize,
        complete: F,
    ) where
        F: FnOnce(CudaResult) + Send + 'static,
    {
        let nbytes = elem_size * counts.volume() as usize;

        self.submit(
            MemoryKind::Host,
            MemoryKind::Device(dst_id),
            nbytes,
            complete,
            move |stream| {
                let devices = &*self.inner.streams;
                let mut params = create_memcpy3d(src_strides, dst_strides, counts, elem_size)?;

                params.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
                params.srcHost = src_ptr as *const c_void;
                params.srcContext = devices[dst_id.get()].context.raw();

                params.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
                params.dstDevice = dst_ptr.raw();
                params.dstContext = devices[dst_id.get()].context.raw();

                cuda_check(cuMemcpy3DPeerAsync(&params, stream.raw()))
            },
        );
    }

    pub(crate) unsafe fn copy_d2h<F>(
        &self,
        src_id: DeviceId,
        src_ptr: CudaDevicePtr,
        dst_ptr: *mut (),
        nbytes: usize,
        complete: F,
    ) where
        F: FnOnce(CudaResult) + Send + 'static,
    {
        self.submit(
            MemoryKind::Device(src_id),
            MemoryKind::Host,
            nbytes,
            complete,
            move |stream| {
                cuda_check(cuMemcpyDtoHAsync_v2(
                    dst_ptr as *mut c_void,
                    src_ptr.raw(),
                    nbytes,
                    stream.raw(),
                ))
            },
        );
    }

    pub(crate) unsafe fn copy_d2h_strided<F>(
        &self,
        src_id: DeviceId,
        src_ptr: CudaDevicePtr,
        src_strides: ByteStrides,
        dst_ptr: *mut (),
        dst_strides: ByteStrides,
        counts: Dim,
        elem_size: usize,
        complete: F,
    ) where
        F: FnOnce(CudaResult) + Send + 'static,
    {
        let nbytes = elem_size * counts.volume() as usize;

        self.submit(
            MemoryKind::Device(src_id),
            MemoryKind::Host,
            nbytes,
            complete,
            move |stream| {
                let devices = &*self.inner.streams;
                let mut params = create_memcpy3d(src_strides, dst_strides, counts, elem_size)?;

                params.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
                params.srcDevice = src_ptr.raw();
                params.srcContext = devices[src_id.get()].context.raw();

                params.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
                params.dstHost = dst_ptr as *mut c_void;
                params.dstContext = devices[src_id.get()].context.raw();

                cuda_check(cuMemcpy3DPeerAsync(&params, stream.raw()))
            },
        );
    }

    pub(crate) unsafe fn copy_d2d<F>(
        &self,
        src_id: DeviceId,
        src_ptr: CudaDevicePtr,
        dst_id: DeviceId,
        dst_ptr: CudaDevicePtr,
        nbytes: usize,
        complete: F,
    ) where
        F: FnOnce(CudaResult) + Send + 'static,
    {
        assert!(self.supported_d2d(src_id, dst_id));

        self.submit(
            MemoryKind::Device(src_id),
            MemoryKind::Device(dst_id),
            nbytes,
            complete,
            move |stream| {
                if src_id == dst_id {
                    cuda_check(cuMemcpyDtoDAsync_v2(
                        dst_ptr.raw(),
                        src_ptr.raw(),
                        nbytes,
                        stream.raw(),
                    ))
                } else {
                    let devices = &*self.inner.streams;

                    cuda_check(cuMemcpyPeerAsync(
                        dst_ptr.raw(),
                        devices[dst_id.get()].context.raw(),
                        src_ptr.raw(),
                        devices[src_id.get()].context.raw(),
                        nbytes,
                        stream.raw(),
                    ))
                }
            },
        );
    }

    pub(crate) unsafe fn copy_d2d_strided<F>(
        &self,
        src_id: DeviceId,
        src_ptr: CudaDevicePtr,
        src_strides: ByteStrides,
        dst_id: DeviceId,
        dst_ptr: CudaDevicePtr,
        dst_strides: ByteStrides,
        counts: Dim,
        elem_size: usize,
        complete: F,
    ) where
        F: FnOnce(CudaResult) + Send + 'static,
    {
        let nbytes = elem_size * counts.volume() as usize;

        self.submit(
            MemoryKind::Device(src_id),
            MemoryKind::Device(dst_id),
            nbytes,
            complete,
            move |stream| {
                let devices = &*self.inner.streams;
                let mut params = create_memcpy3d(src_strides, dst_strides, counts, elem_size)?;

                params.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
                params.srcDevice = src_ptr.raw();
                params.srcContext = devices[src_id.get()].context.raw();

                params.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
                params.dstDevice = dst_ptr.raw();
                params.dstContext = devices[dst_id.get()].context.raw();

                cuda_check(cuMemcpy3DPeerAsync(&params, stream.raw()))
            },
        );
    }
}

fn create_memcpy3d(
    mut src_strides: ByteStrides,
    mut dst_strides: ByteStrides,
    mut counts: Dim,
    elem_size: usize,
) -> CudaResult<CUDA_MEMCPY3D_PEER> {
    for i in 0..MAX_DIMS {
        if counts[i] <= 1 {
            src_strides[i] = elem_size as i64;
            dst_strides[i] = elem_size as i64;
        }

        if src_strides[i] < elem_size as i64 || dst_strides[i] < elem_size as i64 {
            warn!("invalid strides: {:?}", (src_strides, dst_strides, counts));
            return Err(CudaError::new(CUDA_ERROR_INVALID_VALUE).unwrap_err());
        }
    }

    for i in 0..MAX_DIMS {
        for j in (1 + 1)..MAX_DIMS {
            if (src_strides[i], dst_strides[i], counts[i])
                > (src_strides[j], dst_strides[j], counts[j])
            {
                src_strides.swap(i, j);
                dst_strides.swap(i, j);
                counts.swap(i, j);
            }
        }
    }

    let mut ndims = 0;
    let mut width_in_bytes = elem_size as i64;

    while ndims < MAX_DIMS
        && src_strides[ndims] == width_in_bytes
        && dst_strides[ndims] == width_in_bytes
    {
        width_in_bytes = src_strides[ndims] * counts[ndims] as i64;
        ndims += 1;
    }

    let src_pitch;
    let dst_pitch;
    let height;
    let depth;
    let src_height;
    let dst_height;

    if ndims == 3 {
        src_pitch = width_in_bytes;
        dst_pitch = width_in_bytes;
        height = 1; // dont care
        depth = 1; // dont care
        src_height = 1; // dont care
        dst_height = 1; // dont care
    } else if ndims == 2 {
        src_pitch = src_strides[2];
        dst_pitch = dst_strides[2];
        height = counts[2];
        src_height = counts[2] as i64; // dont care
        dst_height = counts[2] as i64; // dont care
        depth = 1;
    } else if ndims == 1
        && src_strides[2] % src_strides[1] == 0
        && dst_strides[2] % dst_strides[1] == 0
        && src_strides[1] > width_in_bytes
        && dst_strides[1] > width_in_bytes
    {
        src_pitch = src_strides[1];
        dst_pitch = dst_strides[1];
        height = counts[1];
        src_height = src_strides[2] / src_strides[1];
        dst_height = dst_strides[2] / dst_strides[1];
        depth = counts[2];
    } else {
        warn!("invalid strides: {:?}", (src_strides, dst_strides, counts));
        return Err(CudaError::new(CUDA_ERROR_INVALID_VALUE).unwrap_err());
    };

    let params = CUDA_MEMCPY3D_PEER {
        srcXInBytes: 0,
        srcY: 0,
        srcZ: 0,
        srcLOD: 0,
        srcMemoryType: CUmemorytype_enum::CU_MEMORYTYPE_HOST,
        srcHost: null_mut(),
        srcDevice: 0,
        srcArray: null_mut(),
        srcContext: null_mut(),
        dstXInBytes: 0,
        dstY: 0,
        dstZ: 0,
        dstLOD: 0,
        dstMemoryType: CUmemorytype_enum::CU_MEMORYTYPE_HOST,
        dstHost: null_mut(),
        dstDevice: 0,
        dstArray: null_mut(),
        dstContext: null_mut(),

        srcPitch: src_pitch as usize,
        srcHeight: src_height as usize,
        dstPitch: dst_pitch as usize,
        dstHeight: dst_height as usize,
        WidthInBytes: width_in_bytes as usize,
        Height: height as usize,
        Depth: depth as usize,
    };

    Ok(params)
}
