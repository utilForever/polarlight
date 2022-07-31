use std::any::Any;
use std::error;
use std::io::{Cursor, Read};
use std::path::PathBuf;
use flate2::read::GzDecoder;

pub mod dataset;


/// Downloads file from url to designated root path.
/// Downloading process will only be proceeded when file path does not exist
///
/// # Parameters
/// * `root` - Root directory of dataset
/// * `file_name` - Name of file
/// * `url` - Url where original data is stored
fn download_from_url(root: &mut PathBuf, file_name: &str, url: &str, decompress: bool) -> Result<(), Box<dyn error::Error>>
{
    // append file_name to root to construct file path
    root.push(file_name);
    // download only when file does not exist
    if !root.exists() {
        // get request
        let resp = reqwest::blocking::get(url)?;

        // create file
        let mut file = std::fs::File::create(&root)?;

        if decompress {
            // Decompress response and store its bytes in buffer
            let mut gz = GzDecoder::new(resp);
            let mut buf: Vec<u8> = Vec::new();
            gz.read_to_end(&mut buf);

            // write bytes to file
            let mut content = Cursor::new(buf);
            std::io::copy(&mut content, &mut file)?;
        } else {
            // write bytes to file
            let mut content = Cursor::new(resp.bytes()?);
            std::io::copy(&mut content, &mut file)?;
        }

    }
    // convert back to original data root path
    root.pop();
    Ok(())
}
