#![allow(dead_code)]

pub trait Module {
    fn init(&self, in_feautures: i32, out_features: i32) -> &Self;
    fn forward(&self, inputs: tensor::Tensor) -> tensor::Tensor;
}