use super::tensor::Tensor;
use super::tensor::traits;

use std::fmt;

// ===== Module =====

pub trait Module <T: traits::TensorTrait<T>> {
    fn forward(&self, inputs: &Tensor<T>) -> &Tensor<T>;
}

// ===== Linear =====

pub struct Linear <T: traits::TensorTrait<T>> {
    pub in_features: i32,
    pub out_features: i32,
    pub weights: Tensor<T>,
    pub bias: Tensor<T>,
    pub module_name: String,
}

// impl Module for Linear {
//     pub fn forward(&self, inputs: &Tensor) {
        
//     }
// }

impl<T: traits::TensorTrait<T>> fmt::Display for Linear <T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "module_name: {}\nin_features: {}\nout_features:{}", self.module_name, self.in_features, self.out_features)
    }
}

// impl<T: traits::TensorTrait<T>> Linear <T> {
//     pub fn build(in_features: i32, out_features: i32) -> Self {
//         let weights_shape = vec![in_features, out_features];
//         let weights_size = (in_features * out_features) as usize;
//         let weights_components = Vec::with_capacity(weights_size);

//         for i in 0..weights_size {
//             let j: T = i as T;
//             weights_components.push(j);
//         }

//         let bias_shape = vec![out_features];
//         let bias_size = out_features as usize;
//         let bias_components = Vec::with_capacity(bias_size);

//         for i in 0..bias_size {
//             let j: T = i as T;
//             bias_components.push(j);
//         }

//         Linear {
//             in_features: in_features,
//             out_features: out_features,
//             weights: Tensor::build(weights_shape, weights_components),
//             bias: Tensor::build(bias_shape, bias_components),
//             module_name: String::from("Linear"),
//         }
//     }
// }

// ===== ReLU =====

pub struct ReLU {
    pub module_name: String,
}

// impl Module for ReLU {
//     pub fn forward(&self, inputs: &Tensor) -> &Tensor {

//     }
// }

impl fmt::Display for ReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "module_name: {}", self.module_name)
    }
}

// impl ReLU {
//     pub fn build() -> Self {
//         ReLU {
//             module_name: String::from("ReLU"),
//         }
//     }
// }
