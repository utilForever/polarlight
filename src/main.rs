use polarlight::tensor::Tensor;
use polarlight::nn;

fn main() {
    a = Tensor::build(vec![2, 3], vec![1, 2, 3, 1, 2, 3]);

    linear = nn::Linear::init(2, 3);

    b = linear.forward(a);
}
