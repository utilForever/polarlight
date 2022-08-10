use flate2::read::GzDecoder;
use std::error;
use std::io::{Cursor, Read};
use std::path::PathBuf;

pub mod dataset;

/// Downloads file from url to designated root path.
/// Download process will be proceeded when the file path does not exist.
///
/// # Arguments
/// * `root` - The root directory of dataset
/// * `file_name` - The file name to download
/// * `url` - URL where the original data is stored
pub fn download_from_url(
    root: &mut PathBuf,
    file_name: &str,
    url: &str,
    is_decompress: bool,
) -> Result<(), Box<dyn error::Error>> {
    // append file_name to root to construct file path
    root.push(file_name);
    // download only when file does not exist
    if !root.exists() {
        // get request
        let resp = reqwest::blocking::get(url)?;

        // create file
        let mut file = std::fs::File::create(&root)?;

        if is_decompress {
            // Decompress response and store its bytes in buffer
            let mut gz = GzDecoder::new(resp);
            let mut buf: Vec<u8> = Vec::new();
            gz.read_to_end(&mut buf)?;

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

/// Convert byte array to u32
///
/// # Arguments
/// * `byt` : byte array
/// * `offset` : offset of byte array to convert
/// * `big_endian` : Boolean for if data is stored as big_endian
fn byte_arr_to_u32(byt: &Vec<u8>, offset: u32, big_endian: bool) -> u32 {
    let mut val: u32 = 0;
    for i in 0..4 {
        let tmp = byt[(offset + i) as usize];
        if big_endian {
            val += (tmp as u32) << (24 - 8 * i);
        } else {
            val += (tmp as u32) << (8 * i);
        }
    }
    val
}
