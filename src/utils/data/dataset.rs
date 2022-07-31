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
    use std::{error, fs};
    use super::Dataset;
    use std::path::{PathBuf};
    use crate::utils::data::download_from_url;

    /// On Memory Data struct
    ///
    /// # Parameters
    /// * `file_name` - name of downloaded file
    /// * `source` - file online source
    /// * `raw` - raw data on memory
    struct OnMemData {
        root: PathBuf,
        file_name: &'static str,
        source: &'static str,
        raw: Vec<u8>
    }
    impl OnMemData {
        /// Constructor
        fn new(mut root: PathBuf, file_name: &'static str, source: &'static str) -> Result<OnMemData, Box<dyn error::Error>> {
            // Download if file does not exist
            download_from_url(&mut root, file_name, source)?;

            // load file to memory
            root.push(file_name);
            let raw = fs::read(&root)?;
            root.pop();

            Ok(OnMemData {
                root,
                file_name,
                source,
                raw
            })
        }
    }

    /// MNIST dataset
    ///
    /// # Parameters
    /// * `root` - Root directory of dataset
    pub struct MNIST {
        test: bool,
        root: PathBuf,
        train_img: Option<OnMemData>,
        train_label: Option<OnMemData>,
        test_img: Option<OnMemData>,
        test_label: Option<OnMemData>
    }
    impl MNIST {
        /// Constructor of MNIST dataset
        ///
        /// # Parameters
        /// * `root` - Root directory of dataset
        /// * `train` - If True, downloads training set, otherwise downloads test set
        fn new(mut root: PathBuf, test: bool) -> Result<MNIST, Box<dyn error::Error>> {
            // validate root
            // create one if one does not exist
            if let Ok(_) = std::fs::create_dir_all(root.as_path()) {}

            if test {
                let test_img = Some(OnMemData::new(root.clone(), "mnist_test_img",
                                              "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")?);
                let test_label = Some(OnMemData::new(root.clone(), "mnist_test_label",
                                                "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")?);
                Ok(MNIST { test, root, train_img: None, train_label: None, test_img, test_label })
            } else {
                let train_img = Some(OnMemData::new(root.clone(), "mnist_train_img",
                                                   "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")?);
                let train_label = Some(OnMemData::new(root.clone(), "mnist_train_label",
                                                 "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")?);
                Ok(MNIST { test, root, train_img, train_label, test_img: None, test_label: None })
            }
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
            if let Ok(mnist_train) = MNIST::new(PathBuf::from("raw"), true) {
                // TODO test mnist dataset
            }
        }
    }
}