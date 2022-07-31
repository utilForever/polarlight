use std::{error, fmt};

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
    use std::{error, fmt, fs};
    use std::fmt::{format, write};
    use super::Dataset;
    use std::path::{PathBuf};
    use crate::utils::data::{bytearr_to_u32, download_from_url};
    use crate::utils::data::dataset::DatasetItem;

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
            download_from_url(&mut root, file_name, source, true)?;

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

    // TODO need to make this Batchable
    struct MnistItem {
        image: Vec<u8>,
        label: u8,
        row: u32,
        col: u32
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
                if (i+1) as u32 % self.row == 0 {
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

        fn get_dim(&self) -> Vec<u32> {
            let mut image_data: Option<&OnMemData> = None;

            if self.test {
                if let Some(image_) = &self.test_img {
                    image_data = Some(image_);
                }
            }else {
                if let Some(image_) = &self.train_img {
                    image_data = Some(image_);
                }
            }

            let mut n_rows: u32 = 0;
            let mut n_cols: u32 = 0;
            if let Some(image_) = image_data {
                n_rows = bytearr_to_u32(&image_.raw, 8, true);
                n_cols = bytearr_to_u32(&image_.raw, 12, true);
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
            }else {
                if let Some(label_) = &self.train_label {
                    label = Some(label_);
                }
            }

            // big endian
            // extract number of labels
            if let Some(label_data) = label {
                bytearr_to_u32(&label_data.raw, 4, true)
            } else {
                0
            }
        }

        fn get_item(&self, idx: u32) -> Result<MnistItem, Box<dyn error::Error>>
        {
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
            }else {
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
                for idx in start_idx..start_idx+size {
                    image.push(image_.raw[idx as usize]);
                }
            }

            // get label
            let mut label = 0;
            if let Some(label_) = label_data {
                label = label_.raw[(8 + idx) as usize];
            }

            // build item
            let item = MnistItem { image, label, row: n_rows, col: n_cols};
            Ok(item)
        }
    }

    /// Test code for builtin datasets
    #[cfg(test)]
    mod tests{
        use std::path::PathBuf;
        use crate::utils::data::dataset::Dataset;
        use super::{MNIST};

        #[test]
        fn generate_mnist_dataset() {
            if let Ok(mnist_test) = MNIST::new(PathBuf::from("raw"), true) {
                let dataset_length = mnist_test.len();
                print!("length: {}", dataset_length);
                if let Ok(dataset_item) = mnist_test.get_item(1) {
                    println!("{:?}", dataset_item);
                }
            }
        }
    }
}