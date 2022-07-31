use std::error;
use std::io::Cursor;
use std::path::PathBuf;

pub mod dataset;


/// Downloads file from url to designated root path.
/// Downloading process will only be proceeded when file path does not exist
///
/// # Parameters
/// * `root` - Root directory of dataset
/// * `file_name` - Name of file
/// * `url` - Url where original data is stored
fn download_from_url(root: &mut PathBuf, file_name: &str, url: &str) -> Result<(), Box<dyn error::Error>>
{
    // append file_name to root to construct file path
    root.push(file_name);
    if !root.exists() {
        let resp = reqwest::blocking::get(url)?;
        // create file
        let mut file = std::fs::File::create(&root)?;
        // get response bytes
        let byt = resp.bytes()?;
        // write bytes to file
        let mut content = Cursor::new(byt);
        std::io::copy(&mut content, &mut file)?;
    } else {
        panic!("Root directory [{:?}] does not exist", root);
    }
    // convert back to original data root path
    root.pop();
    Ok(())
}
