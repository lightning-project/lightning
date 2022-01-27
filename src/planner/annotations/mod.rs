#[macro_use]
mod eval;
mod parser;

use std::fmt::{self, Debug};
use std::num::NonZeroI64;

pub(crate) use self::parser::parse_rules;
use crate::types::{ReductionFunction, MAX_DIMS};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum AccessMode {
    Read,
    Write,
    ReadWrite,
    Reduce(ReductionFunction),
    Atomic(ReductionFunction),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum Var {
    GlobalIndex(u8),
    LocalIndex(u8),
    BlockIndex(u8),
    BlockSize(u8),
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) enum Expr {
    Var(Var),
    Arg(Box<str>),
    Immediate(i64),
    Add(Box<[Expr; 2]>),
    Mul(Box<[Expr; 2]>),
    Div(Box<[Expr; 2]>),
}

impl Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Var(v) => write!(f, "{:?}", v),
            Expr::Immediate(v) => write!(f, "{:?}", v),
            Expr::Add(t) => write!(f, "({:?} + {:?})", &t[0], &t[1]),
            Expr::Mul(t) => write!(f, "({:?} * {:?})", &t[0], &t[1]),
            Expr::Div(t) => write!(f, "({:?} / {:?})", &t[0], &t[1]),
            Expr::Arg(v) => write!(f, "${}", v),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SliceExpr {
    Index {
        index: Box<Expr>,
    },
    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        step: Option<Box<Expr>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AccessPattern(Box<[SliceExpr]>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AnnotationRule {
    pub(crate) name: String,
    pub(crate) access_mode: AccessMode,
    pub(crate) access_pattern: AccessPattern,
}

// Linear equation of the form:
// (thread_factor[0] * thread_idx[0] + .. + thread_factor[n] * thread_idx[n] +
//    block_factor[0] * block_idx[0] + .. + block_factor[n] * block_idx[n] +
//    constant) / divisor
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct AffineExpr {
    block_factors: [i64; MAX_DIMS],
    thread_factors: [i64; MAX_DIMS],
    constant: i64,
    divisor: NonZeroI64,
}

// Slice of the form `start + i * step` for integers `0 <= i < num_steps`
#[derive(Debug, Clone, PartialEq, Eq)]
struct AffineSliceExpr {
    start: AffineExpr,
    num_steps: NonZeroI64,
    step_size: NonZeroI64,
}

#[derive(Debug, Clone)]
pub(crate) struct AffineAccessPattern(Box<[AffineSliceExpr]>);

#[derive(Debug)]
pub(crate) struct EvalError;
