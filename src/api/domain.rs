use crate::types::{
    Dim, Dim1, Dim2, Dim3, Point, Point1, Point2, Point3, Rect1, Rect2, Rect3, MAX_DIMS,
};
use std::fmt::Debug;
use std::ops::{Bound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

pub trait Shape {
    fn into_dim(self) -> Dim;
    fn from_dim(this: Dim) -> Self;
}

impl Shape for u64 {
    fn into_dim(self) -> Dim {
        Dim::from([self])
    }

    fn from_dim(this: Dim) -> Self {
        assert_eq!(&this[1..], &[1; MAX_DIMS - 1]);
        this[0]
    }
}

impl Shape for () {
    fn into_dim(self) -> Dim {
        Dim::from(self)
    }

    fn from_dim(this: Dim) -> Self {
        assert_eq!(&this[..], &[1; MAX_DIMS]);
        ()
    }
}

impl Shape for (u64,) {
    fn into_dim(self) -> Dim {
        Dim::from(self)
    }

    fn from_dim(this: Dim) -> Self {
        assert_eq!(&this[1..], &[1; MAX_DIMS - 1]);
        (this[0],)
    }
}

impl Shape for (u64, u64) {
    fn into_dim(self) -> Dim {
        Dim::from(self)
    }

    fn from_dim(this: Dim) -> Self {
        assert_eq!(&this[2..], &[1; MAX_DIMS - 2]);
        (this[0], this[1])
    }
}

impl Shape for (u64, u64, u64) {
    fn into_dim(self) -> Dim {
        Dim::from(self)
    }

    fn from_dim(this: Dim) -> Self {
        assert_eq!(&this[3..], &[1; MAX_DIMS - 3]);
        (this[0], this[1], this[2])
    }
}

impl Shape for [u64; 0] {
    fn into_dim(self) -> Dim {
        Dim::one()
    }

    fn from_dim(this: Dim) -> Self {
        let () = Shape::from_dim(this);
        []
    }
}

impl Shape for [u64; 1] {
    fn into_dim(self) -> Dim {
        Dim::from(self)
    }

    fn from_dim(this: Dim) -> Self {
        let (x,) = Shape::from_dim(this);
        [x]
    }
}

impl Shape for [u64; 2] {
    fn into_dim(self) -> Dim {
        Dim::from(self)
    }

    fn from_dim(this: Dim) -> Self {
        let (x, y) = Shape::from_dim(this);
        [x, y]
    }
}

impl Shape for [u64; 3] {
    fn into_dim(self) -> Dim {
        Dim::from(self)
    }

    fn from_dim(this: Dim) -> Self {
        let (x, y, z) = Shape::from_dim(this);
        [x, y, z]
    }
}

impl Shape for Dim1 {
    fn into_dim(self) -> Dim {
        Dim::from(self)
    }

    fn from_dim(this: Dim) -> Self {
        let (x,) = Shape::from_dim(this);
        Dim1::new(x)
    }
}

impl Shape for Dim2 {
    fn into_dim(self) -> Dim {
        Dim::from(self)
    }

    fn from_dim(this: Dim) -> Self {
        let (x, y) = Shape::from_dim(this);
        Dim2::new(x, y)
    }
}

impl Shape for Dim3 {
    fn into_dim(self) -> Dim {
        Dim::from(self)
    }

    fn from_dim(this: Dim) -> Self {
        let (x, y, z) = Shape::from_dim(this);
        Dim3::new(x, y, z)
    }
}

impl Shape for Vec<u64> {
    fn into_dim(self) -> Dim {
        assert!(self.len() < MAX_DIMS);
        let x = self.get(0).copied().unwrap_or(1);
        let y = self.get(1).copied().unwrap_or(1);
        let z = self.get(2).copied().unwrap_or(1);
        Dim::new(x, y, z)
    }

    fn from_dim(this: Dim) -> Self {
        this.to_vec()
    }
}

impl Shape for Box<[u64]> {
    fn into_dim(self) -> Dim {
        self.to_vec().into_dim()
    }

    fn from_dim(this: Dim) -> Self {
        this.to_vec().into_boxed_slice()
    }
}

pub trait ArrayIndex {
    fn to_point(self) -> Point;
}

impl ArrayIndex for u64 {
    fn to_point(self) -> Point {
        Point::from([self])
    }
}

impl ArrayIndex for () {
    fn to_point(self) -> Point {
        Point::zeros()
    }
}

impl ArrayIndex for (u64,) {
    fn to_point(self) -> Point {
        Point::from(self)
    }
}

impl ArrayIndex for (u64, u64) {
    fn to_point(self) -> Point {
        Point::from(self)
    }
}

impl ArrayIndex for (u64, u64, u64) {
    fn to_point(self) -> Point {
        Point::from(self)
    }
}

impl ArrayIndex for [u64; 0] {
    fn to_point(self) -> Point {
        Point::zeros()
    }
}

impl ArrayIndex for [u64; 1] {
    fn to_point(self) -> Point {
        Point::from(self)
    }
}

impl ArrayIndex for [u64; 2] {
    fn to_point(self) -> Point {
        Point::from(self)
    }
}

impl ArrayIndex for [u64; 3] {
    fn to_point(self) -> Point {
        Point::from(self)
    }
}

impl ArrayIndex for Point1 {
    fn to_point(self) -> Point {
        Point::from(self)
    }
}

impl ArrayIndex for Point2 {
    fn to_point(self) -> Point {
        Point::from(self)
    }
}

impl ArrayIndex for Point3 {
    fn to_point(self) -> Point {
        Point::from(self)
    }
}

impl ArrayIndex for &[u64] {
    fn to_point(self) -> Point {
        Point::new(
            self.get(0).copied().unwrap_or(0),
            self.get(1).copied().unwrap_or(0),
            self.get(2).copied().unwrap_or(0),
        )
    }
}

#[derive(Copy, Clone, Debug)]
pub enum SliceDescriptor {
    Range(Bound<u64>, Bound<u64>),
    Index(u64),
    NewAxis,
}

pub trait SliceRange {
    fn into_range(self) -> SliceDescriptor;
}

impl SliceRange for Range<u64> {
    fn into_range(self) -> SliceDescriptor {
        SliceDescriptor::Range(Bound::Included(self.start), Bound::Excluded(self.end))
    }
}

impl SliceRange for RangeFrom<u64> {
    fn into_range(self) -> SliceDescriptor {
        SliceDescriptor::Range(Bound::Included(self.start), Bound::Unbounded)
    }
}

impl SliceRange for RangeTo<u64> {
    fn into_range(self) -> SliceDescriptor {
        SliceDescriptor::Range(Bound::Unbounded, Bound::Excluded(self.end))
    }
}

impl SliceRange for RangeFull {
    fn into_range(self) -> SliceDescriptor {
        SliceDescriptor::Range(Bound::Unbounded, Bound::Unbounded)
    }
}

impl SliceRange for RangeInclusive<u64> {
    fn into_range(self) -> SliceDescriptor {
        let (start, end) = self.into_inner();
        SliceDescriptor::Range(Bound::Included(start), Bound::Included(end))
    }
}

impl SliceRange for RangeToInclusive<u64> {
    fn into_range(self) -> SliceDescriptor {
        SliceDescriptor::Range(Bound::Unbounded, Bound::Included(self.end))
    }
}

pub trait ArraySlice {
    type Slices: AsRef<[SliceDescriptor]>;

    fn into_slices(self) -> Self::Slices;
}

impl ArraySlice for Rect1 {
    type Slices = [SliceDescriptor; 1];

    fn into_slices(self) -> Self::Slices {
        let (lo, hi) = (self.low(), self.high());
        (lo[0]..hi[0]).into_slices()
    }
}

impl ArraySlice for Rect2 {
    type Slices = [SliceDescriptor; 2];

    fn into_slices(self) -> Self::Slices {
        let (lo, hi) = (self.low(), self.high());
        (lo[0]..hi[0], lo[1]..hi[1]).into_slices()
    }
}

impl ArraySlice for Rect3 {
    type Slices = [SliceDescriptor; 3];

    fn into_slices(self) -> Self::Slices {
        let (lo, hi) = (self.low(), self.high());
        (lo[0]..hi[0], lo[1]..hi[1], lo[2]..hi[2]).into_slices()
    }
}

impl ArraySlice for u64 {
    type Slices = [SliceDescriptor; 1];

    fn into_slices(self) -> Self::Slices {
        [SliceDescriptor::Index(self)]
    }
}

impl<I: SliceRange> ArraySlice for I {
    type Slices = [SliceDescriptor; 1];

    fn into_slices(self) -> Self::Slices {
        [self.into_range()]
    }
}

impl ArraySlice for [u64; 0] {
    type Slices = [SliceDescriptor; 0];

    fn into_slices(self) -> Self::Slices {
        []
    }
}

impl ArraySlice for [u64; 1] {
    type Slices = [SliceDescriptor; 1];

    fn into_slices(self) -> Self::Slices {
        [SliceDescriptor::Index(self[0])]
    }
}

impl ArraySlice for [u64; 2] {
    type Slices = [SliceDescriptor; 2];

    fn into_slices(self) -> Self::Slices {
        [
            SliceDescriptor::Index(self[0]),
            SliceDescriptor::Index(self[1]),
        ]
    }
}

impl ArraySlice for [u64; 3] {
    type Slices = [SliceDescriptor; 3];

    fn into_slices(self) -> Self::Slices {
        [
            SliceDescriptor::Index(self[0]),
            SliceDescriptor::Index(self[1]),
            SliceDescriptor::Index(self[2]),
        ]
    }
}

impl ArraySlice for () {
    type Slices = [SliceDescriptor; 0];

    fn into_slices(self) -> Self::Slices {
        []
    }
}

impl ArraySlice for (u64,) {
    type Slices = [SliceDescriptor; 1];

    fn into_slices(self) -> Self::Slices {
        [SliceDescriptor::Index(self.0)]
    }
}

impl ArraySlice for (u64, u64) {
    type Slices = [SliceDescriptor; 2];

    fn into_slices(self) -> Self::Slices {
        [
            SliceDescriptor::Index(self.0),
            SliceDescriptor::Index(self.1),
        ]
    }
}

impl ArraySlice for (u64, u64, u64) {
    type Slices = [SliceDescriptor; 3];

    fn into_slices(self) -> Self::Slices {
        [
            SliceDescriptor::Index(self.0),
            SliceDescriptor::Index(self.1),
            SliceDescriptor::Index(self.2),
        ]
    }
}

impl<I: SliceRange> ArraySlice for (I,) {
    type Slices = [SliceDescriptor; 1];

    fn into_slices(self) -> Self::Slices {
        [self.0.into_range()]
    }
}

impl<I: SliceRange> ArraySlice for (I, u64) {
    type Slices = [SliceDescriptor; 2];

    fn into_slices(self) -> Self::Slices {
        [self.0.into_range(), SliceDescriptor::Index(self.1)]
    }
}

impl<J: SliceRange> ArraySlice for (u64, J) {
    type Slices = [SliceDescriptor; 2];

    fn into_slices(self) -> Self::Slices {
        [SliceDescriptor::Index(self.0), self.1.into_range()]
    }
}

impl<I: SliceRange> ArraySlice for (I, u64, u64) {
    type Slices = [SliceDescriptor; 3];

    fn into_slices(self) -> Self::Slices {
        [
            self.0.into_range(),
            SliceDescriptor::Index(self.1),
            SliceDescriptor::Index(self.2),
        ]
    }
}

impl<J: SliceRange> ArraySlice for (u64, J, u64) {
    type Slices = [SliceDescriptor; 3];

    fn into_slices(self) -> Self::Slices {
        [
            SliceDescriptor::Index(self.0),
            self.1.into_range(),
            SliceDescriptor::Index(self.2),
        ]
    }
}

impl<K: SliceRange> ArraySlice for (u64, u64, K) {
    type Slices = [SliceDescriptor; 3];

    fn into_slices(self) -> Self::Slices {
        [
            SliceDescriptor::Index(self.0),
            SliceDescriptor::Index(self.1),
            self.2.into_range(),
        ]
    }
}

impl<I: SliceRange, J: SliceRange> ArraySlice for (I, J) {
    type Slices = [SliceDescriptor; 2];

    fn into_slices(self) -> Self::Slices {
        [self.0.into_range(), self.1.into_range()]
    }
}

impl<I: SliceRange, J: SliceRange> ArraySlice for (I, J, u64) {
    type Slices = [SliceDescriptor; 3];

    fn into_slices(self) -> Self::Slices {
        [
            self.0.into_range(),
            self.1.into_range(),
            SliceDescriptor::Index(self.2),
        ]
    }
}

impl<J: SliceRange, K: SliceRange> ArraySlice for (u64, J, K) {
    type Slices = [SliceDescriptor; 3];

    fn into_slices(self) -> Self::Slices {
        [
            SliceDescriptor::Index(self.0),
            self.1.into_range(),
            self.2.into_range(),
        ]
    }
}

impl<I: SliceRange, K: SliceRange> ArraySlice for (I, u64, K) {
    type Slices = [SliceDescriptor; 3];

    fn into_slices(self) -> Self::Slices {
        [
            self.0.into_range(),
            SliceDescriptor::Index(self.1),
            self.2.into_range(),
        ]
    }
}

impl<I: SliceRange, J: SliceRange, K: SliceRange> ArraySlice for (I, J, K) {
    type Slices = [SliceDescriptor; 3];

    fn into_slices(self) -> Self::Slices {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
        ]
    }
}
