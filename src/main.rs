extern crate polarlight;

use polarlight::nn;
use polarlight::tensor::Tensor;

fn main() {
    let weights = Tensor::build(
        vec![2, 2],
        vec![
            1.0, 2.0,
            3.0, 4.0
        ]);
    
    let bias = Tensor::build(
        vec![2],
        vec![0.5, 0.5]);

    
    let linear1 = nn::Linear {
        in_features: 2,
        out_features: 2,
        weights: weights,
        bias: bias,
        module_name: String::from("Linear : 1")
    };

    println!("{}\n", linear1);

    let linear2 = nn::Linear::build(3, 3);

    println!("{}\n", linear2);

    let relu = nn::ReLU{
        module_name: String::from("ReLU : 1")
    };

    println!("{}", relu);
}