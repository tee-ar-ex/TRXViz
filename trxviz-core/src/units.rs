use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct Millimeters(pub f32);

impl Millimeters {
    pub const ZERO: Self = Self(0.0);

    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }
}

impl Default for Millimeters {
    fn default() -> Self {
        Self::ZERO
    }
}

impl From<f32> for Millimeters {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<Millimeters> for f32 {
    fn from(value: Millimeters) -> Self {
        value.0
    }
}

impl Add for Millimeters {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Millimeters {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul<f32> for Millimeters {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Div<f32> for Millimeters {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl fmt::Display for Millimeters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct ParcelId(pub u32);

impl From<u32> for ParcelId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<ParcelId> for u32 {
    fn from(value: ParcelId) -> Self {
        value.0
    }
}

impl fmt::Display for ParcelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct StreamlineIndex(pub u32);

impl From<u32> for StreamlineIndex {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<StreamlineIndex> for u32 {
    fn from(value: StreamlineIndex) -> Self {
        value.0
    }
}

impl fmt::Display for StreamlineIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct GroupId(pub u32);

impl From<u32> for GroupId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<GroupId> for u32 {
    fn from(value: GroupId) -> Self {
        value.0
    }
}

impl fmt::Display for GroupId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
