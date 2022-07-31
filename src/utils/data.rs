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
fn download_from_url(root: &mut PathBuf, file_name: &str, url: &str) {
    // append file_name to root to construct file path
    // TODO check hash
    root.push(file_name);
    // TODO throw error
    if !root.exists() {
        if let Ok(resp) = reqwest::blocking::get(url) {
            if let Some(file_path_str) = root.to_str() {
                // create file
                if let Ok(mut file) = std::fs::File::create(String::from(file_path_str)) {
                    // get response bytes
                    if let Ok(byt) = resp.bytes() {
                        // write bytes to file
                        let mut content = Cursor::new(byt);
                        if let Ok(_) = std::io::copy(&mut content, &mut file) {}
                    }
                }
            }
        }
    }
    // convert back to original data root path
    root.pop();
}
