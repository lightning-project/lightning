use smallvec::SmallVec;

use super::annotations::{AccessMode, AffineAccessPattern};
use super::task::CudaLaunchTasklet;
use super::{ArrayId, PlannerStage};
use crate::planner::stage::UnmapAction;
use crate::prelude::*;
use crate::types::dag::OperationChunk;
use crate::types::{
    Affine, CudaArg, CudaKernelId, DataValue, Dim, Dim3, ExecutorId, MemoryId, Point, Rect,
    Transform, Translate, MAX_DIMS,
};

pub(crate) enum CudaLauncherArg {
    Value {
        value: DataValue,
    },
    Array {
        id: ArrayId,
        domain: Rect,
        transform: Affine,
        access_mode: AccessMode,
        access_patterns: SmallVec<[AffineAccessPattern; 1]>,
    },
}

pub(crate) struct CudaLauncher {
    pub(crate) kernel_id: CudaKernelId,
    pub(crate) block_size: Dim3,
    pub(crate) args: Vec<CudaLauncherArg>,
}

impl CudaLauncher {
    pub fn submit_launch(
        &self,
        planner: &mut PlannerStage,
        executor: ExecutorId,
        block_offset: Point,
        block_count: Dim,
    ) -> Result {
        if block_count.is_empty() {
            return Ok(());
        }

        let place = executor.best_affinity_memory();
        let n = self.args.len();
        let mut chunks = Vec::with_capacity(n);
        let mut actions = Vec::with_capacity(n);
        let mut new_args = Vec::with_capacity(self.args.len());

        for arg in &self.args {
            let new_arg = match arg {
                CudaLauncherArg::Value { value } => CudaArg::Value(value.clone()),
                &CudaLauncherArg::Array {
                    id,
                    domain,
                    ref transform,
                    access_mode,
                    ref access_patterns,
                } => self.build_array_arg(
                    planner,
                    place,
                    block_offset,
                    block_count,
                    id,
                    domain,
                    transform,
                    access_mode,
                    access_patterns,
                    &mut chunks,
                    &mut actions,
                )?,
            };

            new_args.push(new_arg);
        }

        let tasklet = CudaLaunchTasklet {
            kernel_id: self.kernel_id,
            block_size: self.block_size,
            block_offset,
            block_count,
            shared_mem: 0,
            args: new_args,
        };

        let exe_op =
            planner
                .plan
                .add_tasklet(executor.node_id(), executor.kind(), &tasklet, chunks)?;
        planner.plan.add_terminal(exe_op);

        for action in actions {
            planner.unmap_array(exe_op, action)?;
        }

        Ok(())
    }

    fn build_array_arg<'a>(
        &self,
        planner: &mut PlannerStage<'_, 'a, '_>,
        place: MemoryId,
        block_offset: Point,
        block_count: Dim,
        id: ArrayId,
        array_domain: Rect,
        array_transform: &Affine,
        access_mode: AccessMode,
        access_patterns: &[AffineAccessPattern],
        chunks: &mut Vec<OperationChunk>,
        actions: &mut Vec<UnmapAction<'a>>,
    ) -> Result<CudaArg> {
        use AccessMode::*;
        if access_patterns.len() < 1
            || (access_patterns.len() > 1 && !matches!(access_mode, Read | Atomic(_)))
        {
            bail!("invalid number of access patterns");
        }

        let dependent_blocks = access_patterns[0].dependent_block_factors();

        // TODO: This needs more attention. For Write/ReadWrite the access pattern must
        // fill the entire region. For Reduce, the access must be such that each block fills
        // the entire region.
        let subdomain = match access_mode {
            Read | Atomic(_) => {
                let mut bounds = Rect::default();

                for pattern in access_patterns {
                    bounds = Rect::union(
                        bounds,
                        pattern.compute_bounds(
                            self.block_size,
                            block_offset,
                            block_count,
                            array_domain,
                        ),
                    );
                }

                bounds
            }
            Write | ReadWrite => access_patterns[0].compute_bounds(
                self.block_size,
                block_offset,
                block_count,
                array_domain,
            ),
            Reduce(_) => access_patterns[0].compute_bounds_exact(
                self.block_size,
                block_offset,
                block_count,
                array_domain,
            )?,
        };

        // TODO: This is not ok if the array_transform is non-regular?!
        let region = array_transform.apply_bounds(subdomain);

        let (chunk, chunk_offset, unmap_action) = match access_mode {
            Read => planner.map_read(place, id, region),
            ReadWrite => planner.map_readwrite(place, id, region),
            Write => planner.map_write(place, id, region),
            Atomic(reduction) => planner.map_reduce(place, id, region, reduction),
            Reduce(reduction) => {
                let n = (0..MAX_DIMS)
                    .filter(|&i| !dependent_blocks[i])
                    .map(|i| block_count[i])
                    .product();

                planner.map_replicated_reduce(place, id, region, reduction, n)
            }
        }?;

        chunks.push(chunk);
        actions.push(unmap_action);

        let transform = Affine::combine(
            array_transform,
            &Affine::from(Translate::combine(
                &Translate::add_offset(chunk_offset),
                &Translate::sub_offset(region.low()),
            )),
        );

        let array_index = chunks.len() - 1;
        let ndims = access_patterns[0].len();

        if !matches!(access_mode, Reduce(_)) {
            Ok(CudaArg::array(ndims, array_index, subdomain, transform))
        } else {
            let mut n = subdomain.extents()[MAX_DIMS - 1];
            let mut matrix = [[0; MAX_DIMS]; MAX_DIMS];

            for i in 0..MAX_DIMS {
                if !dependent_blocks[i] {
                    matrix[MAX_DIMS - 1][i] = n as i64;
                    n *= block_count[i];
                }
            }

            Ok(CudaArg::array_per_block(
                ndims,
                array_index,
                Transform::new(matrix),
                subdomain,
                transform,
            ))
        }
    }
}
