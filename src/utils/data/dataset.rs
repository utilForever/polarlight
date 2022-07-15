/// Map-style Dataset
pub trait Dataset {
    /// Returns the number of data in this dataset
    ///
    /// # Arguments
    /// * `self` - A reference to itself
    fn len(&self) -> i32;

    /// Returns the datum corresponding to the index
    ///
    /// # Arguments
    /// * `self` - A reference to itself
    /// * `idx` - An index that corresponds to a datum
    fn get_item<T>(&self, idx: i32) -> T;
}

/// Consists of builtin datasets (MNIST etc.)
pub mod builtin {
    use super::Dataset;
    use std::path::PathBuf;
    use crate::utils::data::download_from_url;

    /// MNIST dataset
    ///
    /// # Parameters
    /// * `root` - Root directory of dataset
    pub struct MNIST {
        // TODO make this root optional.
        root: PathBuf,
    }
    impl MNIST {
        /// Constructor of MNIST dataset
        ///
        /// # Parameters
        /// * `root` - Root directory of dataset
        /// * `train` - If True, downloads training set, otherwise downloads test set
        fn new(mut root: PathBuf) -> MNIST {
            // validate root
            // create one if one does not exist
            if let Ok(_) = std::fs::create_dir_all(root.as_path()) {}

            // Download dataset from official MNIST distribution website
            // The download process will only happen when file path does not exist
            download_from_url(&mut root, "mnist_train_img", "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz");
            download_from_url(&mut root, "mnist_train_label", "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz");
            download_from_url(&mut root, "mnist_test_img", "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz");
            download_from_url(&mut root, "mnist_test_label", "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz");

            MNIST { root }
        }
    }
    impl Dataset for MNIST {
        fn len(&self) -> i32 {
            todo!()
        }

        fn get_item<T>(&self, idx: i32) -> T {
            todo!()
        }
    }

    /// Test code for builtin datasets
    #[cfg(test)]
    mod tests{
        use std::path::PathBuf;
        use super::{MNIST};

        #[test]
        fn generate_mnist_dataset() {
            let mnist_train = MNIST::new(PathBuf::from("raw"));
        }
    }
}