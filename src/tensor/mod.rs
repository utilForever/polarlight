use core::panicking::panic;
#[allow(dead_code)]
use crate::traits;


//generalized tensor
pub struct Tensor<T: traits::TensorTrait<T>> {
    shape: Vec<i32>,
    components: Vec<T>,
}

impl<T: traits::TensorTrait<T>> Tensor<T> {
    pub fn build(shape: Vec<i32>, components: Vec<T>) -> Self {
        let mut len = 1;
        for i in &shape {
            len *= i;
        }

        if len != components.len() as i32 {
            panic!("The length of components is not equal to the length of shape");
        }

        Tensor {
            shape: shape,
            components: components,
        }
    }
    pub fn get(&self, index: Vec<i32>) -> T {
        let mut index = index;
        let mut i: i32 = 0;
        for x in 0..self.dim() {
            let mut tmp = 1;
            for y in x + 1..self.dim() {
                tmp *= self.shape[y as usize];
            }

            i += tmp * index[x as usize];
        }

        return self.components[i as usize].clone();
    }

    pub fn dim(&self) -> i32 {
        self.shape.len() as i32
    }

    pub fn add(&self, other: &Tensor<T>) -> Tensor<T>
        where
            T: std::ops::Add<Output=T>,
    {
        if self.dim() != other.dim() {
            panic!("The dimension of two tensors are not equal");
        }
        let mut vec = Vec::new();
        for i in 0..self.components.len() {
            let mut tmp1 = self.components[i as usize].clone();
            let mut tmp2 = other.components[i as usize].clone();
            vec.push(tmp1 + tmp2);
        }
        Tensor::build(self.shape.clone(), vec)
    }

    pub fn sub(&self, other: &Tensor<T>) -> Tensor<T>
        where
            T: std::ops::Sub<Output=T>,
    {
        if self.dim() != other.dim() {
            panic!("The dimension of two tensors are not equal");
        }
        let mut vec = Vec::new();
        for i in 0..self.components.len() {
            let mut tmp1 = self.components[i as usize].clone();
            let mut tmp2 = other.components[i as usize].clone();
            vec.push(tmp1 - tmp2);
        }
        Tensor::build(self.shape.clone(), vec)
    }

    pub fn div(&self, other: &Tensor<T>) -> Tensor<T>
        where
            T: std::ops::Div<Output=T>,
    {
        if self.dim() != other.dim() {
            panic!("The dimension of two tensors are not equal");
        }
        let mut vec = Vec::new();
        for i in 0..self.components.len() {
            let mut tmp1 = self.components[i as usize].clone();
            let mut tmp2 = other.components[i as usize].clone();
            // println!("{}", tmp1 / tmp2);
            vec.push(tmp1 / tmp2);
        }
        Tensor::build(self.shape.clone(), vec)
    }

    pub fn mul(&self, other: &Tensor<T>) -> Tensor<T>
        where
            T: std::ops::Mul<Output=T>,
    {
        if self.dim() != other.dim() {
            panic!("The dimension of two tensors are not equal");
        }
        let mut vec = Vec::new();
        for i in 0..self.components.len() {
            let mut tmp1 = self.components[i as usize].clone();
            let mut tmp2 = other.components[i as usize].clone();
            vec.push(tmp1 * tmp2);
        }
        Tensor::build(self.shape.clone(), vec)
    }

    pub fn matmul(&self, other: &Tensor<T>) -> Tensor<T>
        where
            T: std::ops::Mul<Output=T>,
    {
        // TODO : 3d, 4d matmul
        if self.dim() != other.dim() {
            panic!("The dimension of two tensors are not equal");
        }

        if self.dim() == 2 { self.matmul2d(other) } else { panic!("Not implemented"); }
    }

    fn matmul2d(&self, other: &Tensor<T>) -> Tensor<T>
        where
            T: std::ops::Mul<Output=T>,
            T: std::ops::Add<Output=T>,
    {
        if self.shape[1] != other.shape[0] {
            panic!("{:?} @ {:?} The dimension of two tensors are not equal", self.shape, other.shape);
        }
        let new_shape = vec![self.shape[0], other.shape[1]];
        let mut vec = Vec::new();
        for k in 0..self.shape[0] {
            for i in 0..other.shape[1] {
                let mut tmp: T = T::zero();
                for j in 0..self.shape[1] {
                    tmp = tmp + self.get(vec![k, j]) * other.get(vec![j, i]);
                }
                vec.push(tmp);
            }
        }
        Tensor::build(new_shape, vec)
    }

    fn matmul3d(&self, other: &Tensor<T>) -> Tensor<T>
        where
            T: std::ops::Mul<Output=T>,
            T: std::ops::Add<Output=T>,
    {
        // TODO : Not Implemented
        panic!("Not implemented");
    }

    pub fn transpose(&self) -> Tensor<T> {
        // TODO : 3d, 4d transpose

        let mut vec = Vec::new();
        for i in 0..self.shape[1] {
            for j in 0..self.shape[0] {
                vec.push(self.get(vec![j, i]));
            }
        }
        Tensor::build(vec![self.shape[1], self.shape[0]], vec)
    }

    pub fn reshape(&self, shape: Vec<i32>) -> Tensor<T> {
        // TODO
        panic!("Not implemented");
        let mut vec = Vec::new();
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                vec.push(self.get(vec![i, j]));
            }
        }
        Tensor::build(shape, vec)
    }

    pub fn print(&self) {
        println!("shape {:?}", self.shape);
        self.print_helper();
        println!("\n");
    }

    fn print_helper(&self) {
        if self.dim() == 1 {
            println!("{:?}", self.components);
        } else if self.dim() == 2 {
            print!("[");
            for i in 0..self.shape[0] {
                print!("[");
                for j in 0..self.shape[1] {
                    if j == self.shape[1] - 1 { print!("{}", self.get(vec![i, j])); } else { print!("{}, ", self.get(vec![i, j])); }
                }
                if i == self.shape[0] - 1 { print!("]"); } else { println!("], "); }
            }
            print!("]");
        } else if self.dim() == 3 {
            print!("[");
            for i in 0..self.shape[0] {
                print!("[");
                for j in 0..self.shape[1] {
                    print!("[");
                    for k in 0..self.shape[2] {
                        if k == self.shape[2] - 1 { print!("{}", self.get(vec![i, j, k])); } else { print!("{}, ", self.get(vec![i, j, k])); }
                    }
                    if j == self.shape[1] - 1 { print!("]"); } else { println!("], "); }
                }
                if i == self.shape[0] - 1 { print!("]"); } else { println!("], "); }
            }
            print!("]");
        } else if self.dim() == 4 {
            print!("[");
            for i in 0..self.shape[0] {
                print!("[");
                for j in 0..self.shape[1] {
                    print!("[");
                    for k in 0..self.shape[2] {
                        print!("[");
                        for l in 0..self.shape[3] {
                            if l == self.shape[3] - 1 { print!("{}", self.get(vec![i, j, k, l])); } else { print!("{}, ", self.get(vec![i, j, k, l])); }
                        }
                        if k == self.shape[2] - 1 { print!("]"); } else { println!("], "); }
                    }
                    if j == self.shape[1] - 1 { print!("]"); } else { println!("], "); }
                }
                if i == self.shape[0] - 1 { print!("]"); } else { println!("], "); }
            }
            print!("]");
        }
    }
}