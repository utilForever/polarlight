use std::fmt::Debug;
use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul};

extern crate num;
use self::num::traits::Zero;

pub trait TensorTrait<T>:
    AddAssign<T> + Zero + Clone + Display + Debug + Add<T, Output = T> + Mul<T, Output = T>
{
}
impl<T> TensorTrait<T> for T where
    T: AddAssign<T> + Zero + Clone + Display + Debug + Add<T, Output = T> + Mul<T, Output = T>
{
}
