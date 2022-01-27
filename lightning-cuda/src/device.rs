//! CUDA device management.
use crate::{cuda_call, cuda_check, Dim3, Error, Result};
use cuda_driver_sys::*;
use std::convert::TryInto;
use std::ffi::CStr;
use std::{fmt, mem};

/// CUDA-capable device.
///
/// This object is a wrapper around a native `CUdevice` with the guarantee that it represents a
/// valid CUDA device.
#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
#[repr(transparent)]
pub struct Device(CUdevice);

// CUdevice is just an integer, it should be thread-safe
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// Returns the number of CUDA-capable devices in the system.
    pub fn count() -> Result<usize> {
        unsafe {
            let n = cuda_call(|ptr| cuDeviceGetCount(ptr))?;
            Ok(n as usize)
        }
    }

    /// Returns all CUDA-capable devices in the system.
    pub fn all() -> Result<Vec<Self>> {
        let mut devices = vec![];

        for i in 0..Self::count()? {
            devices.push(Self::nth(i)?);
        }

        Ok(devices)
    }

    /// Returns `Device` for the given `ordinal`.
    pub fn nth(ordinal: usize) -> Result<Self> {
        unsafe {
            let i = ordinal
                .try_into()
                .map_err(|_| Error::from_raw(CUresult::CUDA_ERROR_INVALID_VALUE))?;
            let device = cuda_call(|ptr| cuDeviceGet(ptr, i))?;
            Ok(Self(device))
        }
    }

    /// Returns the `Device` for the CUDA context currently registered to the calling thread.
    pub fn current() -> Result<Self> {
        unsafe { cuda_call(|dev| cuCtxGetDevice(dev)).map(|d| Self::from_raw(d)) }
    }

    /// Construct a `Device` from the given `CUdevice`.
    ///
    /// # Safety
    /// The given argument must be a valid `CUdevice` since this function performs no additional checks.
    #[inline(always)]
    pub unsafe fn from_raw(device: CUdevice) -> Self {
        Self(device)
    }

    /// Returns the `CUdevice` that is wrapped by this `Device`.
    #[inline(always)]
    pub fn raw(self) -> CUdevice {
        self.0
    }

    /// Returns the ordinal of this device.
    pub fn ordinal(self) -> usize {
        self.0 as usize
    }

    /// Returns the name of this `Device`.
    pub fn name(self) -> Result<String> {
        unsafe {
            let mut buffer = [0; 1024];
            cuda_check(cuDeviceGetName(
                buffer.as_mut_ptr(),
                buffer.len() as i32,
                self.0,
            ))?;

            // [u8; 1024] -> CStr -> &[u8] -> Vec<u8> -> String
            let name = CStr::from_ptr(buffer.as_ptr()).to_bytes().to_vec();
            Ok(String::from_utf8_unchecked(name))
        }
    }

    /// Returns the compute capability of this device. This is expressed as tuple of (major version,
    /// minor version). For example, the value `(3, 0)` represents compute capability 3.0 (Kepler).
    pub fn compute_capability(self) -> Result<(i32, i32)> {
        Ok((
            self.attribute(DeviceAttribute::COMPUTE_CAPABILITY_MAJOR)?,
            self.attribute(DeviceAttribute::COMPUTE_CAPABILITY_MINOR)?,
        ))
    }

    /// Returns the value of the given device attribute.
    #[inline]
    pub fn attribute(self, attrib: DeviceAttribute) -> Result<i32> {
        unsafe {
            let attrib = mem::transmute::<DeviceAttribute, CUdevice_attribute>(attrib);
            let pi = cuda_call(|v| cuDeviceGetAttribute(v, attrib, self.0))?;
            Ok(pi)
        }
    }

    /// Returns the maximum block size that can be launched on this device.
    pub fn max_block_dim(&self) -> Result<Dim3> {
        Ok(Dim3::new(
            self.attribute(DeviceAttribute::MAX_BLOCK_DIM_X)? as u32,
            self.attribute(DeviceAttribute::MAX_BLOCK_DIM_Y)? as u32,
            self.attribute(DeviceAttribute::MAX_BLOCK_DIM_Z)? as u32,
        ))
    }

    /// Returns the maximum grid size that can be launched on this device.
    pub fn max_grid_dim(&self) -> Result<Dim3> {
        Ok(Dim3::new(
            self.attribute(DeviceAttribute::MAX_GRID_DIM_X)? as u32,
            self.attribute(DeviceAttribute::MAX_GRID_DIM_Y)? as u32,
            self.attribute(DeviceAttribute::MAX_GRID_DIM_Z)? as u32,
        ))
    }

    /// Returns the number of streaming multiprocessors (SMs) of this device.
    pub fn multiprocessor_count(&self) -> Result<u32> {
        self.attribute(DeviceAttribute::MULTIPROCESSOR_COUNT)
            .map(|v| v as u32)
    }

    /// Returns the total memory capacity of this device in bytes.
    pub fn total_memory(self) -> Result<usize> {
        unsafe { cuda_call(|v| cuDeviceTotalMem_v2(v, self.0)) }
    }

    /// Check if this device can access a peer device.
    pub fn can_access_peer(&self, peer: Device) -> Result<bool> {
        if self == &peer {
            return Ok(true);
        }

        unsafe { cuda_call(|v| cuDeviceCanAccessPeer(v, self.0, peer.0)) }.map(|v| v != 0)
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = self
            .name()
            .unwrap_or_else(|_| "<unknown device>".to_string());
        f.debug_tuple("CudaDevice")
            .field(&format_args!("{} #{}", name, self.ordinal()))
            .finish()
    }
}

use CUdevice_attribute::*;

/// Attribute which can be queried using [`Device::attribute`].
///
/// [`Device::attribute`]: struct.Device.html#method.attribute
///
/// Enumerates the `CU_DEVICE_ATTRIBUTE_*` constants.
#[repr(u32)]
#[allow(non_camel_case_types)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum DeviceAttribute {
    MAX_BLOCK_DIM_X = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X as u32,
    MAX_BLOCK_DIM_Y = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y as u32,
    MAX_BLOCK_DIM_Z = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z as u32,
    MAX_GRID_DIM_X = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X as u32,
    MAX_GRID_DIM_Y = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y as u32,
    MAX_GRID_DIM_Z = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z as u32,
    MAX_SHARED_MEMORY_PER_BLOCK = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK as u32,
    TOTAL_CONSTANT_MEMORY = CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY as u32,
    WARP_SIZE = CU_DEVICE_ATTRIBUTE_WARP_SIZE as u32,
    MAX_PITCH = CU_DEVICE_ATTRIBUTE_MAX_PITCH as u32,
    MAX_REGISTERS_PER_BLOCK = CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK as u32,
    CLOCK_RATE = CU_DEVICE_ATTRIBUTE_CLOCK_RATE as u32,
    TEXTURE_ALIGNMENT = CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT as u32,
    GPU_OVERLAP = CU_DEVICE_ATTRIBUTE_GPU_OVERLAP as u32,
    MULTIPROCESSOR_COUNT = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as u32,
    KERNEL_EXEC_TIMEOUT = CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT as u32,
    INTEGRATED = CU_DEVICE_ATTRIBUTE_INTEGRATED as u32,
    CAN_MAP_HOST_MEMORY = CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY as u32,
    COMPUTE_MODE = CU_DEVICE_ATTRIBUTE_COMPUTE_MODE as u32,
    MAXIMUM_TEXTURE1D_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH as u32,
    MAXIMUM_TEXTURE2D_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH as u32,
    MAXIMUM_TEXTURE2D_HEIGHT = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT as u32,
    MAXIMUM_TEXTURE3D_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH as u32,
    MAXIMUM_TEXTURE3D_HEIGHT = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT as u32,
    MAXIMUM_TEXTURE3D_DEPTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH as u32,
    MAXIMUM_TEXTURE2D_LAYERED_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH as u32,
    MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT as u32,
    MAXIMUM_TEXTURE2D_LAYERED_LAYERS = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS as u32,
    SURFACE_ALIGNMENT = CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT as u32,
    CONCURRENT_KERNELS = CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS as u32,
    ECC_ENABLED = CU_DEVICE_ATTRIBUTE_ECC_ENABLED as u32,
    PCI_BUS_ID = CU_DEVICE_ATTRIBUTE_PCI_BUS_ID as u32,
    PCI_DEVICE_ID = CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID as u32,
    TCC_DRIVER = CU_DEVICE_ATTRIBUTE_TCC_DRIVER as u32,
    MEMORY_CLOCK_RATE = CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE as u32,
    GLOBAL_MEMORY_BUS_WIDTH = CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH as u32,
    L2_CACHE_SIZE = CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE as u32,
    MAX_THREADS_PER_MULTIPROCESSOR = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR as u32,
    ASYNC_ENGINE_COUNT = CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT as u32,
    UNIFIED_ADDRESSING = CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING as u32,
    MAXIMUM_TEXTURE1D_LAYERED_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH as u32,
    MAXIMUM_TEXTURE1D_LAYERED_LAYERS = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS as u32,
    CAN_TEX2D_GATHER = CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER as u32,
    MAXIMUM_TEXTURE2D_GATHER_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH as u32,
    MAXIMUM_TEXTURE2D_GATHER_HEIGHT = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT as u32,
    MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE =
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE as u32,
    MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE =
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE as u32,
    MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE =
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE as u32,
    PCI_DOMAIN_ID = CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID as u32,
    TEXTURE_PITCH_ALIGNMENT = CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT as u32,
    MAXIMUM_TEXTURECUBEMAP_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH as u32,
    MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH =
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH as u32,
    MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS =
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS as u32,
    MAXIMUM_SURFACE1D_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH as u32,
    MAXIMUM_SURFACE2D_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH as u32,
    MAXIMUM_SURFACE2D_HEIGHT = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT as u32,
    MAXIMUM_SURFACE3D_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH as u32,
    MAXIMUM_SURFACE3D_HEIGHT = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT as u32,
    MAXIMUM_SURFACE3D_DEPTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH as u32,
    MAXIMUM_SURFACE1D_LAYERED_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH as u32,
    MAXIMUM_SURFACE1D_LAYERED_LAYERS = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS as u32,
    MAXIMUM_SURFACE2D_LAYERED_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH as u32,
    MAXIMUM_SURFACE2D_LAYERED_HEIGHT = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT as u32,
    MAXIMUM_SURFACE2D_LAYERED_LAYERS = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS as u32,
    MAXIMUM_SURFACECUBEMAP_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH as u32,
    MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH =
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH as u32,
    MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS =
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS as u32,
    MAXIMUM_TEXTURE1D_LINEAR_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH as u32,
    MAXIMUM_TEXTURE2D_LINEAR_WIDTH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH as u32,
    MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT as u32,
    MAXIMUM_TEXTURE2D_LINEAR_PITCH = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH as u32,
    MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH =
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH as u32,
    MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT =
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT as u32,
    COMPUTE_CAPABILITY_MAJOR = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as u32,
    COMPUTE_CAPABILITY_MINOR = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as u32,
    MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH =
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH as u32,
    STREAM_PRIORITIES_SUPPORTED = CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED as u32,
    GLOBAL_L1_CACHE_SUPPORTED = CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED as u32,
    LOCAL_L1_CACHE_SUPPORTED = CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED as u32,
    MAX_SHARED_MEMORY_PER_MULTIPROCESSOR =
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR as u32,
    MAX_REGISTERS_PER_MULTIPROCESSOR = CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR as u32,
    MANAGED_MEMORY = CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY as u32,
    MULTI_GPU_BOARD = CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD as u32,
    MULTI_GPU_BOARD_GROUP_ID = CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID as u32,
    HOST_NATIVE_ATOMIC_SUPPORTED = CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED as u32,
    SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO =
        CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO as u32,
    PAGEABLE_MEMORY_ACCESS = CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS as u32,
    CONCURRENT_MANAGED_ACCESS = CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS as u32,
    COMPUTE_PREEMPTION_SUPPORTED = CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED as u32,
    CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM =
        CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM as u32,
    CAN_USE_STREAM_MEM_OPS = CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS as u32,
    CAN_USE_64_BIT_STREAM_MEM_OPS = CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS as u32,
    CAN_USE_STREAM_WAIT_VALUE_NOR = CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR as u32,
    COOPERATIVE_LAUNCH = CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH as u32,
    COOPERATIVE_MULTI_DEVICE_LAUNCH = CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH as u32,
    MAX_SHARED_MEMORY_PER_BLOCK_OPTIN =
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN as u32,
    CAN_FLUSH_REMOTE_WRITES = CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES as u32,
    HOST_REGISTER_SUPPORTED = CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED as u32,
    PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES =
        CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES as u32,
    DIRECT_MANAGED_MEM_ACCESS_FROM_HOST =
        CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST as u32,
    VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED =
        CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED as u32,
    HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED =
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED as u32,
    HANDLE_TYPE_WIN32_HANDLE_SUPPORTED =
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED as u32,
    HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED =
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED as u32,
}

impl fmt::Debug for DeviceAttribute {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("DeviceAttribute")
            .field(&(*self as u32))
            .finish()
    }
}
