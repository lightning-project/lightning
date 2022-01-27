use lightning_core::util::array;
use std::num::NonZeroI64;
use std::ops::Range;

use super::Var;
use crate::planner::annotations::{
    AccessPattern, AffineAccessPattern, AffineExpr, AffineSliceExpr, EvalError, Expr, SliceExpr,
};
use crate::prelude::*;
use crate::types::{Dim, Dim3, Point3, Rect3, MAX_DIMS};

const ONE: NonZeroI64 = unsafe { NonZeroI64::new_unchecked(1) };

struct Vars<'a> {
    block_size: Dim,
    arguments: &'a dyn Fn(&str) -> Option<i64>,
}

impl AccessPattern {
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn rewrite(
        &self,
        block_size: Dim,
        arguments: &dyn Fn(&str) -> Option<i64>,
    ) -> Result<AffineAccessPattern, EvalError> {
        let old_slices = &self.0;
        let mut new_slices = Vec::with_capacity(old_slices.len());

        let vars = &Vars {
            block_size,
            arguments,
        };

        for old_slice in &**old_slices {
            let new_slice = match old_slice {
                SliceExpr::Index { index } => AffineSliceExpr {
                    start: index.to_affine(vars)?,
                    step_size: ONE,
                    num_steps: ONE,
                },
                SliceExpr::Range { start, end, step } => {
                    let (start, len) = match (start, end) {
                        (Some(start), Some(end)) => {
                            let start = start.to_affine(vars)?;
                            let end = end.to_affine(vars)?;
                            let len = AffineExpr::distance(start, end)?;

                            (start, len)
                        }
                        (Some(start), None) => {
                            let start = start.to_constant(vars)?.max(0);
                            (AffineExpr::constant(start), i64::MAX)
                        }
                        (None, Some(end)) => {
                            let end = end.to_constant(vars)?;
                            (AffineExpr::zero(), end)
                        }
                        (None, None) => (AffineExpr::zero(), i64::MAX),
                    };

                    let step_size = match step {
                        Some(step) => match NonZeroI64::new(step.to_constant(vars)?) {
                            Some(n) => n,
                            None => return Err(EvalError),
                        },
                        None => ONE,
                    };

                    let num_steps = match NonZeroI64::new(div_ceil(len, step_size.get())) {
                        Some(v) if v.get() > 0 => v,
                        _ => return Err(EvalError),
                    };

                    AffineSliceExpr {
                        start,
                        num_steps,
                        step_size,
                    }
                }
            };

            new_slices.push(new_slice);
        }

        Ok(AffineAccessPattern(new_slices.into_boxed_slice()))
    }
}

impl Expr {
    fn to_affine(&self, vars: &Vars) -> Result<AffineExpr, EvalError> {
        match self {
            Expr::Var(v) => Ok(AffineExpr::from_var(*v, vars.block_size)),
            Expr::Immediate(c) => Ok(AffineExpr::constant(*c)),
            Expr::Add(terms) => {
                AffineExpr::addition(terms[0].to_affine(vars)?, terms[1].to_affine(vars)?)
            }
            Expr::Mul(terms) => {
                let lhs = terms[0].to_affine(vars)?;
                let rhs = terms[1].to_affine(vars)?;

                if let Some(lhs) = lhs.to_constant() {
                    AffineExpr::multiply(rhs, lhs)
                } else if let Some(rhs) = rhs.to_constant() {
                    AffineExpr::multiply(lhs, rhs)
                } else {
                    Err(EvalError)
                }
            }
            Expr::Div(terms) => {
                AffineExpr::division(terms[0].to_affine(vars)?, terms[1].to_constant(vars)?)
            }
            Expr::Arg(name) => match (vars.arguments)(&**name) {
                Some(v) => Ok(AffineExpr::constant(v)),
                None => Err(EvalError),
            },
        }
    }

    fn to_constant(&self, vars: &Vars) -> Result<i64, EvalError> {
        Ok(match self {
            Expr::Var(v) => match *v {
                Var::BlockSize(axis) => vars.block_size[axis as usize] as i64,
                _ => return Err(EvalError),
            },
            Expr::Immediate(c) => *c,
            Expr::Add(terms) => terms[0].to_constant(vars)? + terms[1].to_constant(vars)?,
            Expr::Mul(terms) => terms[0].to_constant(vars)? * terms[1].to_constant(vars)?,
            Expr::Div(terms) => {
                i64::div_euclid(terms[0].to_constant(vars)?, terms[1].to_constant(vars)?)
            }
            Expr::Arg(name) => match (vars.arguments)(&**name) {
                Some(v) => v,
                None => return Err(EvalError),
            },
        })
    }
}

impl AffineSliceExpr {
    fn to_bounds(&self, block_size: Dim3, block_offset: Point3, block_count: Dim3) -> Range<i64> {
        let start = self.start;
        let mut constant = start.constant;

        for i in 0..MAX_DIMS {
            constant += start.block_factors[i] * (block_offset[i] as i64);
        }

        let (mut lo, mut hi) = (constant, constant);

        for i in 0..MAX_DIMS {
            let factor = start.thread_factors[i];
            if factor > 0 {
                hi += factor * (block_size[i] as i64 - 1);
            } else if factor < 0 {
                lo += factor * (block_size[i] as i64 - 1);
            }
        }

        for i in 0..MAX_DIMS {
            let factor = start.block_factors[i];
            if factor > 0 {
                hi += factor * (block_count[i] as i64 - 1);
            } else if factor < 0 {
                lo += factor * (block_count[i] as i64 - 1);
            }
        }

        if start.divisor != ONE {
            lo = i64::div_euclid(lo, start.divisor.get());
            hi = i64::div_euclid(hi, start.divisor.get());
        }

        let hi = hi
            .saturating_add((self.num_steps.get() - 1) * self.step_size.get())
            .saturating_add(1);
        lo..hi
    }

    fn to_bounds_exact(
        &self,
        block_size: Dim3,
        block_offset: Point3,
        block_count: Dim3,
    ) -> Result<Range<i64>> {
        let start = self.start;
        let mut constant = start.constant;

        for i in 0..MAX_DIMS {
            constant += start.block_factors[i] * (block_offset[i] as i64);
        }

        let mut factors = [(0, 0); 2 * MAX_DIMS + 1];
        let mut nfactors = 0;

        if self.num_steps.get() > 1 {
            factors[nfactors] = (self.step_size.get(), self.num_steps.get());
            nfactors += 1;
        }

        for i in 0..MAX_DIMS {
            if block_size[i] > 1 && start.thread_factors[i] != 0 {
                factors[nfactors] = (start.thread_factors[i], block_size[i] as i64);
                nfactors += 1;
            }

            if block_count[i] > 1 && start.block_factors[i] != 0 {
                factors[nfactors] = (start.block_factors[i], block_count[i] as i64);
                nfactors += 1;
            }
        }

        factors[..nfactors].sort_by_key(|(f, _)| f.abs());

        let mut current_factor = 1;
        let (mut lo, mut hi) = (constant, constant);

        for &(factor, steps) in &factors[..nfactors] {
            if current_factor != factor.abs() {
                bail!("access pattern is not exact: {:?}", self); // Improve error handling
            }

            current_factor = i64::saturating_mul(factor.abs(), steps.abs());

            if (factor > 0 && steps > 0) || (factor < 0 && steps < 0) {
                hi = lo.saturating_add(current_factor);
            } else {
                lo = hi.saturating_sub(current_factor);
            }
        }

        if start.divisor != ONE {
            lo = i64::div_euclid(lo, start.divisor.get());
            hi = i64::div_euclid(hi, start.divisor.get());
        }

        hi = hi.saturating_add(1);
        Ok(lo..hi)
    }
}

impl AffineAccessPattern {
    pub(crate) fn compute_bounds(
        &self,
        block_size: Dim3,
        block_offset: Point3,
        block_count: Dim3,
        bounds: Rect3,
    ) -> Rect3 {
        let (mut lo, mut hi) = (bounds.low(), bounds.high());

        if block_size.is_empty() || block_count.is_empty() || bounds.is_empty() {
            return Rect3::default();
        }

        for (i, slice) in enumerate(&*self.0) {
            let range = slice.to_bounds(block_size, block_offset, block_count);

            lo[i] = u64::clamp(range.start.max(0) as u64, lo[i], hi[i]);
            hi[i] = u64::clamp(range.end.max(0) as u64, lo[i], hi[i]);
        }

        Rect3::from_bounds(lo, hi)
    }

    pub(crate) fn compute_bounds_exact(
        &self,
        block_size: Dim3,
        block_offset: Point3,
        block_count: Dim3,
        bounds: Rect3,
    ) -> Result<Rect3> {
        let (mut lo, mut hi) = (bounds.low(), bounds.high());

        if block_size.is_empty() || block_count.is_empty() || bounds.is_empty() {
            return Ok(Rect3::default());
        }

        for (i, slice) in enumerate(&*self.0) {
            let range = slice.to_bounds_exact(block_size, block_offset, block_count)?;

            lo[i] = u64::clamp(range.start.max(0) as u64, lo[i], hi[i]);
            hi[i] = u64::clamp(range.end.max(0) as u64, lo[i], hi[i]);
        }

        Ok(Rect3::from_bounds(lo, hi))
    }

    pub(crate) fn dependent_block_factors(&self) -> [bool; MAX_DIMS] {
        let mut dependent = [false; MAX_DIMS];
        for slice in &*self.0 {
            for i in 0..MAX_DIMS {
                dependent[i] |= slice.start.block_factors[i] != 0;
            }
        }

        dependent
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

impl AffineExpr {
    fn from_var(var: Var, block_size: Dim3) -> Self {
        let mut result = AffineExpr::zero();
        match var {
            Var::GlobalIndex(axis) => {
                result.block_factors[axis as usize] = block_size[axis as usize] as i64;
                result.thread_factors[axis as usize] = 1;
            }
            Var::LocalIndex(axis) => {
                result.thread_factors[axis as usize] = 1;
            }
            Var::BlockIndex(axis) => {
                result.block_factors[axis as usize] = 1;
            }
            Var::BlockSize(axis) => {
                result.constant = block_size[axis as usize] as i64;
            }
        }
        result
    }

    fn constant(constant: i64) -> Self {
        Self {
            block_factors: [0; MAX_DIMS],
            thread_factors: [0; MAX_DIMS],
            constant,
            divisor: ONE,
        }
    }

    fn zero() -> Self {
        Self::constant(0)
    }

    fn to_constant(&self) -> Option<i64> {
        if self.block_factors == [0; MAX_DIMS]
            && self.thread_factors == [0; MAX_DIMS]
            && self.divisor == ONE
        {
            Some(self.constant)
        } else {
            None
        }
    }

    fn distance(lhs: Self, rhs: Self) -> Result<i64, EvalError> {
        if lhs.block_factors != rhs.block_factors
            || lhs.thread_factors != rhs.thread_factors
            || lhs.divisor != rhs.divisor
        {
            return Err(EvalError);
        }

        let dist = rhs.constant - lhs.constant;
        let divisor = lhs.divisor.get(); // == rhs.divisor

        if dist % divisor != 0 {
            return Err(EvalError);
        }

        Ok(i64::div_euclid(dist, divisor))
    }

    fn addition_constant(mut expr: Self, value: i64) -> Self {
        expr.constant += value * expr.divisor.get();
        expr
    }

    fn addition(lhs: Self, rhs: Self) -> Result<Self, EvalError> {
        if lhs.divisor != ONE || rhs.divisor != ONE {
            return if let Some(rhs) = rhs.to_constant() {
                Ok(Self::addition_constant(lhs, rhs))
            } else if let Some(lhs) = lhs.to_constant() {
                Ok(Self::addition_constant(rhs, lhs))
            } else {
                Err(EvalError)
            };
        }

        let add = |x, y| x + y;
        Ok(Self {
            block_factors: array::zip(lhs.block_factors, rhs.block_factors, add),
            thread_factors: array::zip(lhs.thread_factors, rhs.thread_factors, add),
            constant: add(lhs.constant, rhs.constant),
            divisor: lhs.divisor, // == rhs.divisor
        })
    }

    fn multiply(expr: Self, factor: i64) -> Result<Self, EvalError> {
        if expr.divisor != ONE && (factor != 1 || factor != 0 || factor != -1) {
            return Err(EvalError);
        }

        let mult = move |x| factor * x;
        Ok(Self {
            block_factors: array::map(expr.block_factors, mult),
            thread_factors: array::map(expr.thread_factors, mult),
            constant: mult(expr.constant),
            divisor: ONE,
        })
    }

    fn division(mut numer: Self, denom: i64) -> Result<Self, EvalError> {
        let divisor = match NonZeroI64::new(numer.divisor.get() * denom) {
            Some(v) => v,
            None => return Err(EvalError),
        };

        numer.divisor = divisor;
        numer.simplify_fraction();
        Ok(numer)
    }

    /// Attempts to simplify a fraction by removing the denominator. For example, "(4x + 6) / 2"
    /// can be rewritten into "2x + 3". But "(3x + 1) / 2" cannot be rewritten. We can simplify a
    /// fraction if all factors are divisible by the denominator.
    fn simplify_fraction(&mut self) {
        let denom = match self.divisor.get() {
            1 => return,
            c => c,
        };

        let mut is_divisible = true;

        for i in 0..MAX_DIMS {
            is_divisible &= self.thread_factors[i] == 0 || self.thread_factors[i] % denom == 0;
            is_divisible &= self.block_factors[i] == 0 || self.block_factors[i] % denom == 0;
        }

        if !is_divisible {
            return;
        }

        self.constant = i64::div_euclid(self.constant, denom);
        self.divisor = ONE;

        for i in 0..MAX_DIMS {
            self.thread_factors[i] = i64::div_euclid(self.thread_factors[i], denom);
            self.block_factors[i] = i64::div_euclid(self.block_factors[i], denom);
        }
    }
}

#[cfg(test)]
mod test {
    use super::super::parse_rules;
    use super::*;

    fn parse_pattern(input: &str, block_size: Dim3) -> AffineAccessPattern {
        let args = |_: &str| None;
        parse_rules(input)
            .unwrap()
            .swap_remove(0)
            .access_pattern
            .rewrite(block_size, &args)
            .unwrap()
    }

    #[test]
    fn test_simple() {
        let n = 1000;
        let block_size = Dim3::from(128);
        let bounds = Rect3::new(Point3::zeros(), Dim3::repeat(n));
        let rule = parse_pattern("global i => read A[i]", block_size);

        let got = rule.compute_bounds(block_size, Point3::zeros(), Dim3::one(), bounds);
        let expect = Rect3::from((0..128, 0..n, 0..n));
        assert_eq!(got, expect);

        let got = rule.compute_bounds(block_size, Point3::ones(), Dim3::one(), bounds);
        let expect = Rect3::from((128..256, 0..n, 0..n));
        assert_eq!(got, expect);

        let got = rule.compute_bounds(block_size, Point3::from(6), Dim3::from(2), bounds);
        let expect = Rect3::from((768..1000, 0..n, 0..n));
        assert_eq!(got, expect);
    }

    #[test]
    fn test_negative() {
        let n = 1000;
        let block_size = Dim3::from(128);
        let bounds = Rect3::new(Point3::zeros(), Dim3::repeat(n));
        let rule = parse_pattern("global i => read A[1000-i-1]", block_size);

        let got = rule.compute_bounds(block_size, Point3::zeros(), Dim3::one(), bounds);
        let expect = Rect3::from((872..1000, 0..n, 0..n));
        assert_eq!(got, expect);

        let got = rule.compute_bounds(block_size, Point3::ones(), Dim3::one(), bounds);
        let expect = Rect3::from((744..872, 0..n, 0..n));
        assert_eq!(got, expect);

        let got = rule.compute_bounds(block_size, Point3::from(6), Dim3::from(2), bounds);
        let expect = Rect3::from((0..232, 0..n, 0..n));
        assert_eq!(got, expect);
    }

    #[test]
    fn test_stencil() {
        let n = 1000;
        let block_size = Dim3::from(128);
        let bounds = Rect3::new(Point3::zeros(), Dim3::repeat(n));
        let rule = parse_pattern("global i => read A[i-1:i+2]", block_size);

        let got = rule.compute_bounds(block_size, Point3::zeros(), Dim3::one(), bounds);
        let expect = Rect3::from((0..129, 0..n, 0..n));
        assert_eq!(got, expect);

        let got = rule.compute_bounds(block_size, Point3::ones(), Dim3::one(), bounds);
        let expect = Rect3::from((127..257, 0..n, 0..n));
        assert_eq!(got, expect);

        let got = rule.compute_bounds(block_size, Point3::from(6), Dim3::from(2), bounds);
        let expect = Rect3::from((767..1000, 0..n, 0..n));
        assert_eq!(got, expect);
    }

    #[test]
    fn test_division() {
        let n = 1000;
        let block_size = Dim3::from(128);
        let bounds = Rect3::new(Point3::zeros(), Dim3::repeat(n));
        let rule = parse_pattern("global i => read A[i / 16:i/16 + 2]", block_size);

        let got = rule.compute_bounds(block_size, Point3::zeros(), Dim3::one(), bounds);
        let expect = Rect3::from((0..9, 0..n, 0..n));
        assert_eq!(got, expect);

        let got = rule.compute_bounds(block_size, Point3::ones(), Dim3::one(), bounds);
        let expect = Rect3::from((8..17, 0..n, 0..n));
        assert_eq!(got, expect);

        let got = rule.compute_bounds(block_size, Point3::from(123), Dim3::from(2), bounds);
        let expect = Rect3::from((984..1000, 0..n, 0..n));
        assert_eq!(got, expect);
    }
}
