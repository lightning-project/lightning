use crate::prelude::*;
use lightning_cuda::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

fn align_to(n: usize, m: usize) -> usize {
    let remainder = n % m;
    if remainder == 0 {
        n
    } else {
        n + m - remainder
    }
}

#[derive(Error, Debug)]
pub(crate) enum MemoryError {
    #[error("memory exhausted")]
    MemoryExhausted,
}

/// Simple best-fit memory allocator. Stores allocated ranges in a BTree ordered by pointer value
/// to quickly locate the next and previous allocated range when deallocating. Gaps are also
/// stored in BTree but ordered by size in increasing order, to quickly locate a gap having
/// the best fit size.
#[derive(Debug)]
struct Allocator {
    offset: usize,
    capacity: usize,
    in_use: usize,
    blocks: BTreeMap<usize, usize>,         // (ptr, size)
    gaps_by_size: BTreeSet<(usize, usize)>, // (size, ptr)
}

impl Allocator {
    fn new(offset: usize, capacity: usize) -> Self {
        let mut gaps_by_size = BTreeSet::new();
        gaps_by_size.insert((capacity, offset));

        Self {
            offset,
            capacity,
            in_use: 0,
            blocks: default(),
            gaps_by_size,
        }
    }

    fn allocate(&mut self, size: usize, alignment: usize) -> Result<usize, MemoryError> {
        assert!(alignment.is_power_of_two());
        let size = max(size, 1);

        let &(gap, offset) = self
            .gaps_by_size
            .range(&(size, 0)..)
            .find(move |&&(gap, offset)| (align_to(offset, alignment) - offset) + size <= gap)
            .ok_or(MemoryError::MemoryExhausted)?;

        self.gaps_by_size.remove(&(gap, offset));

        let aligned_offset = align_to(offset, alignment);
        let pre_gap = aligned_offset - offset;
        let post_gap = gap - pre_gap - size;

        if pre_gap > 0 {
            self.gaps_by_size.insert((pre_gap, offset));
        }

        if post_gap > 0 {
            self.gaps_by_size.insert((post_gap, aligned_offset + size));
        }

        self.blocks.insert(aligned_offset, size);
        self.in_use += size;
        Ok(aligned_offset)
    }

    fn deallocate(&mut self, mut offset: usize, size: usize) {
        let size = max(size, 1);
        let mut cap = size;
        assert_eq!(self.blocks.remove(&offset), Some(size));

        // Find the next allocated block A to the right of this block.
        let right_start = self
            .blocks
            .range(offset..)
            .next()
            .map(|(&a, _)| a)
            .unwrap_or(self.offset + self.capacity);

        // Gap starts at gstart and is gsize bytes long.
        let gap_start = offset + size;
        let gap_size = right_start - gap_start;

        if gap_size > 0 {
            assert!(self.gaps_by_size.remove(&(gap_size, gap_start)));
            cap += gap_size;
        }

        // Find the next allocated block B to the left of this block.
        let left_end = self
            .blocks
            .range(..offset)
            .next_back()
            .map(|(&a, &b)| a + b)
            .unwrap_or(self.offset);

        // Gap starts at gap_start and is gap_size bytes long.
        let gap_start = left_end;
        let gap_size = offset - gap_start;

        if gap_size > 0 {
            assert!(self.gaps_by_size.remove(&(gap_size, gap_start)));
            offset = gap_start;
            cap += gap_size;
        }

        self.gaps_by_size.insert((cap, offset));
        self.in_use -= size;
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        if self.in_use > 0 {
            warn!("memory allocator dropped before releasing all memory, this will leak!");
        }
    }
}

#[derive(Debug)]
pub(crate) struct DeviceMemoryPool {
    mem: CudaDeviceMem<u8>,
    alloc: Allocator,
}

impl DeviceMemoryPool {
    pub(crate) fn new(mem: CudaDeviceMem<u8>) -> Self {
        let alloc = Allocator::new(mem.as_ptr().raw() as usize, mem.size_in_bytes());
        Self { mem, alloc }
    }

    pub(crate) fn allocate(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<CudaDevicePtr<u8>, MemoryError> {
        let offset = self.alloc.allocate(size, alignment)?;
        let result = Ok(unsafe { CudaDevicePtr::from_raw(offset as _) });
        debug!("device allocate {}: {:?}", size, result);
        result
    }

    pub(crate) fn deallocate(&mut self, ptr: CudaDevicePtr<u8>, size: usize) {
        debug!("device deallocate {}: {:?}", size, ptr);
        self.alloc.deallocate(ptr.raw() as usize, size);
    }
}

#[derive(Debug)]
struct Block {
    alloc: Allocator,
    mem: CudaPinnedMem<u8>,
}

#[derive(Debug)]
pub(crate) struct HostMemoryPool {
    context: CudaContextHandle,
    max_capacity: usize,
    min_block_size: usize,
    current_block_idx: usize,
    blocks: Vec<Block>,
}

impl HostMemoryPool {
    pub(crate) fn new(
        context: CudaContextHandle,
        min_block_size: usize,
        max_capacity: usize,
    ) -> Self {
        Self {
            context,
            max_capacity,
            min_block_size,
            current_block_idx: 0,
            blocks: vec![],
        }
    }

    pub(crate) fn allocate(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<*mut u8, MemoryError> {
        use MemoryError::MemoryExhausted;

        let n = self.blocks.len();
        for _ in 0..n {
            if let Ok(offset) = self.blocks[self.current_block_idx]
                .alloc
                .allocate(size, alignment)
            {
                debug!(
                    "host allocate {}: {:?} in block {}",
                    size, offset, self.current_block_idx
                );
                return Ok(offset as *mut u8);
            }

            self.current_block_idx = (self.current_block_idx + 1) % n;
        }

        loop {
            // pool is full, we must allocate a new block of memory. Try to allocate more memory.
            let mut mem = None;
            let block_size = usize::max(self.min_block_size, size + alignment);
            if self.capacity() + block_size <= self.max_capacity {
                use cuda_driver_sys::cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY;

                mem = match self
                    .context
                    .try_with(|| -> CudaResult<_> { CudaPinnedMem::empty(block_size) })
                {
                    Ok(m) => Some(m),
                    Err(e) if e.raw() == CUDA_ERROR_OUT_OF_MEMORY => None,
                    Err(e) => panic!("unexpected error: {}", e),
                };
            }

            // Allocation successfull. Allocate memory from new block and insert block into pool.
            if let Some(mut mem) = mem {
                let mut block = Block {
                    alloc: Allocator::new(mem.as_mut_ptr() as usize, mem.size_in_bytes()),
                    mem,
                };

                // Allocate needed memory. This should not fail.
                let ptr = block.alloc.allocate(size, alignment).unwrap();

                // Insert block into pool. Position should be such that blocks are sorted by ptr.
                let index = match self.find_block_for_ptr(ptr as *const u8) {
                    Some(i) => i + 1, // Insert _after_ block i.
                    None => 0,
                };

                self.blocks.insert(index, block);
                self.current_block_idx = index;

                debug!(
                    "host allocate {} bytes ({:x}) in new block {}",
                    size, ptr, index
                );
                return Ok(ptr as *mut u8);
            }

            // Deallocate block that is not in use and try again.
            if let Some(index) = self.blocks.iter().position(|b| b.alloc.in_use == 0) {
                self.blocks.remove(index);
                continue;
            }

            // There is nothing we can do any more.
            break Err(MemoryExhausted);
        }
    }

    pub(crate) fn deallocate(&mut self, ptr: *mut u8, size: usize) {
        let index = self.find_block_for_ptr(ptr as *const u8).unwrap();
        debug!(
            "host deallocate {} bytes ({:x}) from block {}",
            size, ptr as usize, index
        );
        self.blocks[index].alloc.deallocate(ptr as usize, size);
    }

    #[inline]
    fn find_block_for_ptr(&self, ptr: *const u8) -> Option<usize> {
        // binary search to find corresponding block.
        let mut index = 0;
        let mut len = self.blocks.len();
        while len > 1 {
            let half = len / 2;
            let mid = index + half;

            if ptr >= self.blocks[mid].mem.as_host().as_ptr() {
                index = mid;
                len -= half;
            } else {
                len = half;
            }
        }

        if index >= self.blocks.len() {
            None
        } else if ptr < self.blocks[index].mem.as_host().as_ptr() {
            None
        } else {
            Some(index)
        }
    }

    pub(crate) fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    pub(crate) fn capacity(&self) -> usize {
        self.blocks.iter().map(|b| b.mem.size_in_bytes()).sum()
    }
}

#[cfg(test)]
mod test {
    use super::MemoryError::MemoryExhausted;
    use super::*;

    #[test]
    fn test_allocator() {
        let begin = 123;
        let len = 512;
        let range = begin..(begin + len);
        let mut alloc = Allocator::new(begin, len);

        // Should fail
        alloc.allocate(1000, 1).unwrap_err();

        // Should succeed
        let a = alloc.allocate(16, 8).unwrap();
        assert!(range.contains(&a));

        let b = alloc.allocate(100, 2).unwrap();
        assert!(range.contains(&b));

        let c = alloc.allocate(300, 4).unwrap();
        assert!(range.contains(&c));

        alloc.deallocate(b, 100);

        // Should fail. There should be gap from b which is too small to fit.
        alloc.allocate(102, 2).unwrap_err();

        // Should fit into the hole left by b.
        let b2 = alloc.allocate(100, 2).unwrap();
        assert_eq!(b, b2);

        // Should succeed
        let d = alloc.allocate(10, 1).unwrap();
        assert!(range.contains(&d));

        alloc.deallocate(a, 16);
        alloc.deallocate(c, 300);

        let a = alloc.allocate(16, 8).unwrap();
        assert!(range.contains(&a));

        // Zero-sized allocation should not fail.
        let ptr = alloc.allocate(0, 4).unwrap();
        alloc.deallocate(ptr, 0);

        alloc.deallocate(a, 16);
        alloc.deallocate(b, 100);
        alloc.deallocate(d, 10);

        // Sanity checks
        assert_eq!(alloc.in_use, 0);
        assert_eq!(alloc.blocks.len(), 0);
        assert_eq!(
            alloc.gaps_by_size.iter().collect::<Vec<_>>(),
            vec![&(len, begin)]
        );
    }
}
