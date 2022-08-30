use flate2::read::GzDecoder;
use std::error;
use std::io::{Cursor, Read};
use std::path::{PathBuf};

pub mod dataset;

/// Downloads file from url to designated root path.
/// Download process will be proceeded when the file path does not exist.
///
/// # Arguments
/// * `root` - The root directory of dataset
/// * `file_name` - The file name to download
/// * `url` - URL where the original data is stored
/// * `is_decompress` - true: decompress the downloaded file, false: does not decompress the downloaded file
pub fn download_from_url(
    root: &mut PathBuf,
    file_name: &str,
    url: &str,
    is_decompress: bool,
) -> Result<(), Box<dyn error::Error>> {
    let mut file_path = root.to_path_buf();
    file_path.push(file_name);

    if !file_path.exists() {
        let response = reqwest::blocking::get(url)?;
        let mut file = std::fs::File::create(&file_path)?;
        let mut cursor: Box<dyn Read>;

        if is_decompress {
            let mut gz = GzDecoder::new(response);
            let mut buf: Vec<u8> = Vec::new();
            gz.read_to_end(&mut buf)?;
            cursor = Box::new(Cursor::new(buf));
        } else {
            let bytes = response.bytes()?;
            cursor = Box::new(Cursor::new(bytes));
        }

        std::io::copy(&mut cursor, &mut file)?;
    }

    Ok(())
}

/// Convert byte array to u32
///
/// # Arguments
/// * `byte_arr` : The original byte array
/// * `offset` : The offset of the byte array to convert
/// * `is_big_endian` : A flag that indicates if the data is stored as big endian
fn convert_byte_arr_to_u32(byte_arr: &[u8], offset: u32, is_big_endian: bool) -> u32 {
    let mut val: u32 = 0;

    for i in 0..4 {
        let tmp = byte_arr[(offset + i) as usize];
        if is_big_endian {
            val += (tmp as u32) << (24 - 8 * i);
        } else {
            val += (tmp as u32) << (8 * i);
        }
    }

    val
}
