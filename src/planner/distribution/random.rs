use super::*;
use rand::prelude::*;

pub struct RandomDist<D> {
    inner: D,
}

impl<D> RandomDist<D> {
    pub fn new(inner: D) -> Self {
        Self { inner }
    }
}

impl<D: IntoWorkDistribution> IntoWorkDistribution for RandomDist<D> {
    fn into_work_distribution(
        self,
        system: &SystemInfo,
        size: Dim,
    ) -> Result<Arc<dyn WorkDistribution>> {
        let inner = self.inner.into_work_distribution(system, size)?;

        Ok(Arc::new(RandomDistribution { inner }))
    }
}

pub struct RandomDistribution {
    inner: Arc<dyn WorkDistribution>,
}

impl WorkDistribution for RandomDistribution {
    fn query_point(&self, p: Point) -> ExecutorId {
        self.inner.query_point(p)
    }

    fn query_region(&self, region: Rect) -> Vec<(ExecutorId, Rect)> {
        let mut results = self.inner.query_region(region);
        results.shuffle(&mut SmallRng::seed_from_u64(0));

        results
    }
}
