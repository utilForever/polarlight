use std::{error, fmt};

// TODO Make DatasetItem Batchable
/// DatasetItem.
/// Every return type of get_item() should implement this
pub trait DatasetItem: fmt::Debug {}

/// Map-style Dataset
pub trait Dataset<T: DatasetItem> {
    /// Returns the number of data in this dataset
    ///
    /// # Arguments
    /// * `self` - A reference to itself
    fn len(&self) -> u32;

    /// Returns the datum corresponding to the index
    ///
    /// # Arguments
    /// * `self` - A reference to itself
    /// * `idx` - An index that corresponds to a datum
    fn get_item(&self, idx: u32) -> Result<T, Box<dyn error::Error>>;
}

/// Consists of builtin datasets (MNIST etc.)
pub mod builtin {
    use super::Dataset;
    use crate::utils::data::dataset::DatasetItem;
    use crate::utils::data::{byte_arr_to_u32, download_from_url};
    use std::path::PathBuf;
    use std::{error, fmt, fs};

    /// On Memory Data struct
    ///
    /// # Parameters
    /// * `root` - Root directory
    /// * `file_name` - name of downloaded file
    /// * `source` - file online source
    /// * `raw` - raw data on memory
    struct OnMemData {
        raw: Vec<u8>,
    }
    impl OnMemData {
        /// Constructor
        ///
        /// # Arguments
        /// * `root` - Root directory
        /// * `file_name` - name of downloaded file
        /// * `source` - file online source
        fn new(
            mut root: PathBuf,
            file_name: &'static str,
            source: &'static str,
        ) -> Result<OnMemData, Box<dyn error::Error>> {
            // Download if file does not exist
            download_from_url(&mut root, file_name, source, true)?;

            // load file to memory
            root.push(file_name);
            let raw = fs::read(&root)?;
            root.pop();

            Ok(OnMemData { raw })
        }
    }

    /// Mnist Item
    ///
    /// # Parameters
    /// * `image` - image vector
    /// * `label` - label (ranging from 0 to 9)
    /// * `row` - row size
    /// * `col` - col size
    pub struct MnistItem {
        image: Vec<u8>,
        label: u8,
        row: u32,
    }
    impl DatasetItem for MnistItem {}
    impl fmt::Debug for MnistItem {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let label_str: String = format!("label: {}\n", self.label);
            let mut image_str: String = format!("image:\n");

            for (i, pixel) in self.image.iter().enumerate() {
                if *pixel == 0 {
                    image_str.push_str(" ");
                } else {
                    image_str.push_str("*");
                }
                if (i + 1) as u32 % self.row == 0 {
                    image_str.push_str("\n");
                }
            }
            write!(f, "{}{}", label_str, image_str)
        }
    }

    /// MNIST dataset
    ///
    /// # Parameters
    /// * `root` - Root directory of dataset
    pub struct MNIST {
        test: bool,
        train_img: Option<OnMemData>,
        train_label: Option<OnMemData>,
        test_img: Option<OnMemData>,
        test_label: Option<OnMemData>,
    }
    impl MNIST {
        /// Constructor of MNIST dataset
        ///
        /// # Arguments
        /// * `root` - Root directory of dataset
        /// * `train` - If True, downloads training set, otherwise downloads test set
        fn new(root: PathBuf, test: bool) -> Result<MNIST, Box<dyn error::Error>> {
            // validate root
            // create one if one does not exist
            if let Ok(_) = std::fs::create_dir_all(root.as_path()) {}

            if test {
                let test_img = Some(OnMemData::new(
                    root.clone(),
                    "mnist_test_img",
                    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                )?);
                let test_label = Some(OnMemData::new(
                    root.clone(),
                    "mnist_test_label",
                    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                )?);
                Ok(MNIST {
                    test,
                    train_img: None,
                    train_label: None,
                    test_img,
                    test_label,
                })
            } else {
                let train_img = Some(OnMemData::new(
                    root.clone(),
                    "mnist_train_img",
                    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                )?);
                let train_label = Some(OnMemData::new(
                    root.clone(),
                    "mnist_train_label",
                    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                )?);
                Ok(MNIST {
                    test,
                    train_img,
                    train_label,
                    test_img: None,
                    test_label: None,
                })
            }
        }

        /// Returns dimension of each image.
        /// Returns [N_ROW, N_COL] vector
        fn get_dim(&self) -> Vec<u32> {
            let mut image_data: Option<&OnMemData> = None;

            if self.test {
                if let Some(image_) = &self.test_img {
                    image_data = Some(image_);
                }
            } else {
                if let Some(image_) = &self.train_img {
                    image_data = Some(image_);
                }
            }

            let mut n_rows: u32 = 0;
            let mut n_cols: u32 = 0;
            if let Some(image_) = image_data {
                n_rows = byte_arr_to_u32(&image_.raw, 8, true);
                n_cols = byte_arr_to_u32(&image_.raw, 12, true);
            }
            vec![n_rows, n_cols]
        }
    }
    impl Dataset<MnistItem> for MNIST {
        fn len(&self) -> u32 {
            let mut label: Option<&OnMemData> = None;
            if self.test {
                if let Some(label_) = &self.test_label {
                    label = Some(label_);
                }
            } else {
                if let Some(label_) = &self.train_label {
                    label = Some(label_);
                }
            }

            // big endian
            // extract number of labels
            if let Some(label_data) = label {
                byte_arr_to_u32(&label_data.raw, 4, true)
            } else {
                0
            }
        }

        fn get_item(&self, idx: u32) -> Result<MnistItem, Box<dyn error::Error>> {
            if self.len() <= idx {
                panic!("Index out of bound");
            }

            let mut label_data: Option<&OnMemData> = None;
            let mut image_data: Option<&OnMemData> = None;

            if self.test {
                if let Some(label_) = &self.test_label {
                    label_data = Some(label_);
                }
                if let Some(image_) = &self.test_img {
                    image_data = Some(image_);
                }
            } else {
                if let Some(label_) = &self.train_label {
                    label_data = Some(label_);
                }
                if let Some(image_) = &self.train_img {
                    image_data = Some(image_);
                }
            }

            let dim = self.get_dim();
            let n_rows: u32 = dim[0];
            let n_cols: u32 = dim[1];

            // build images vector
            let mut image = Vec::new();
            if let Some(image_) = image_data {
                let size = n_rows * n_cols;
                let start_idx = 16 + size * idx;
                for idx in start_idx..start_idx + size {
                    image.push(image_.raw[idx as usize]);
                }
            }

            // get label
            let mut label = 0;
            if let Some(label_) = label_data {
                label = label_.raw[(8 + idx) as usize];
            }

            // build item
            let item = MnistItem {
                image,
                label,
                row: n_rows,
            };
            Ok(item)
        }
    }

    /// Test code for builtin datasets
    #[cfg(test)]
    mod tests {
        use super::MNIST;
        use crate::utils::data::dataset::Dataset;
        use std::path::PathBuf;

        #[test]
        fn generate_mnist_dataset() {
            if let Ok(mnist_test) = MNIST::new(PathBuf::from("raw"), true) {
                let dataset_length = mnist_test.len();
                assert_eq!(dataset_length, 10000);
                if let Ok(dataset_item) = mnist_test.get_item(0) {
                    println!("{:?}", dataset_item);
                }
            }
        }
    }
}
