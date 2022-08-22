use crate::utils::data::dataset::{Dataset, DatasetItem};

type U = DatasetItem;
pub struct Dataloader<T>
where
    T: DatasetItem
{
    next_idx: u32,
    dataset: Box<dyn Dataset<T>>
}

impl<T> Dataloader<T>
where
    T: DatasetItem
{
    fn new(dataset: Box<dyn Dataset<T>>) -> Dataloader<T> {
        Dataloader {
            next_idx: 0,
            dataset
        }
    }
}

impl<T> Iterator for Dataloader<T>
where
    T: DatasetItem
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        todo!("Make batch of DatasetItem")
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use crate::utils::data::dataloader::Dataloader;
    use crate::utils::data::dataset::builtin::{MNIST, MnistItem};

    #[test]
    fn dataloader_initialization() {
        let dataset = match MNIST::new(PathBuf::from("raw"), true) {
            Ok(dataset) => dataset,
            Err(e) => {
                panic!("Error occurred when initializing MNIST dataset: {:?}", e);
            }
        };

        let dataloader: Dataloader<MnistItem> = Dataloader::new(Box::new(dataset));

        // TODO iterate through dataset using dataloader
    }
}
