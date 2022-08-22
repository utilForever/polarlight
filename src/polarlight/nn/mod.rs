pub mod traits;

pub struct Linear {
    weights: tensor::Tensor,
    bias: tensor::Tensor,
}

impl traits::Module for Linear {
    fn init(&self, in_feautures: i32, out_features: i32) -> &self {
        &self;
    }

    fn forward(&self, inputs: tensor::Tensor) -> tensor::Tensor {
        inputs
    }
}