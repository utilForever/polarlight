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
    use crate::utils::data::{convert_byte_arr_to_u32, download_from_url};
    use std::path::PathBuf;
    use std::{error, fmt, fs};

    /// On Memory Data struct
    ///
    /// # Parameters
    /// * `raw` - raw data on memory
    struct OnMemData {
        raw: Vec<u8>,
    }

    impl OnMemData {
        /// Constructor
        ///
        /// # Arguments
        /// * `root` - The root directory
        /// * `file_name` - The downloaded file's name
        /// * `source` - The source URL to download
        fn new(
            mut root: PathBuf,
            file_name: &'static str,
            source: &'static str,
        ) -> Result<OnMemData, Box<dyn error::Error>> {
            // Download if file does not exist
            download_from_url(&mut root, file_name, source, true)?;

            // Load file to memory
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
    /// * `test` - true, The testing dataset is loaded, false: The training dataset is loaded
    /// * `train_img` - The on memory data of train images.
    /// * `train_label` - The on memory data of train labels.
    /// * `test_img` - The on memory data of test images.
    /// * `test_label` - The on memory data of test labels.
    pub struct MNIST {
        is_test: bool,
        train_img: Option<OnMemData>,
        train_label: Option<OnMemData>,
        test_img: Option<OnMemData>,
        test_label: Option<OnMemData>,
    }

    impl MNIST {
        /// Constructor of MNIST dataset
        ///
        /// # Arguments
        /// * `root` - The root directory of dataset
        /// * `is_test` - true: downloads test set, false: downloads train set
        pub fn new(root: PathBuf, is_test: bool) -> Result<MNIST, Box<dyn error::Error>> {
            // validate root
            // create one if one does not exist
            if let Ok(_) = fs::create_dir_all(root.as_path()) {}

            if is_test {
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
                    is_test: is_test,
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
                    is_test: is_test,
                    train_img,
                    train_label,
                    test_img: None,
                    test_label: None,
                })
            }
        }

        /// Returns a vector that contains number of rows and columns for each image
        fn get_dim(&self) -> Vec<u32> {
            let mut image_data: Option<&OnMemData> = None;

            if self.is_test {
                if let Some(image) = &self.test_img {
                    image_data = Some(image);
                }
            } else {
                if let Some(image) = &self.train_img {
                    image_data = Some(image);
                }
            }

            let mut n_rows: u32 = 0;
            let mut n_cols: u32 = 0;
            if let Some(image) = image_data {
                n_rows = convert_byte_arr_to_u32(&image.raw, 8, true);
                n_cols = convert_byte_arr_to_u32(&image.raw, 12, true);
            }

            vec![n_rows, n_cols]
        }
    }

    impl Dataset<MnistItem> for MNIST {
        fn len(&self) -> u32 {
            let mut label: Option<&OnMemData> = None;

            if self.is_test {
                if let Some(test_label) = &self.test_label {
                    label = Some(test_label);
                }
            } else {
                if let Some(train_label) = &self.train_label {
                    label = Some(train_label);
                }
            }

            // big endian
            // extract number of labels
            if let Some(label_data) = label {
                convert_byte_arr_to_u32(&label_data.raw, 4, true)
            } else {
                0
            }
        }

        /// Returns the datum corresponding to the index
        ///
        /// # Arguments
        /// * `self` - A reference to itself
        /// * `idx` - An index that corresponds to a datum
        fn get_item(&self, idx: u32) -> Result<MnistItem, Box<dyn error::Error>> {
            if self.len() <= idx {
                panic!("Index out of bound");
            }

            let mut label_data: Option<&OnMemData> = None;
            let mut image_data: Option<&OnMemData> = None;

            if self.is_test {
                if let Some(label) = &self.test_label {
                    label_data = Some(label);
                }
                if let Some(image) = &self.test_img {
                    image_data = Some(image);
                }
            } else {
                if let Some(label) = &self.train_label {
                    label_data = Some(label);
                }
                if let Some(image) = &self.train_img {
                    image_data = Some(image);
                }
            }

            let dim = self.get_dim();
            let n_rows: u32 = dim[0];
            let n_cols: u32 = dim[1];

            // build images vector
            let mut image_vec = Vec::new();

            if let Some(image) = image_data {
                let size = n_rows * n_cols;
                let start_idx = 16 + size * idx;

                for idx in start_idx..start_idx + size {
                    image_vec.push(image.raw[idx as usize]);
                }
            }

            // get label
            let mut item_label = 0;
            if let Some(label) = label_data {
                item_label = label.raw[(8 + idx) as usize];
            }

            // build item
            let item = MnistItem {
                image: image_vec,
                label: item_label,
                row: n_rows,
            };

            Ok(item)
        }
    }

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
