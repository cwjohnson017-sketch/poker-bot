//! Minimal reader for 1-D integer `.npy` files (bucket tables).

use std::fs;

fn field<'a>(header: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("'{name}':");
    let start = header.find(&key)? + key.len();
    Some(header[start..].trim_start())
}

/// Read a 1-D little-endian (or byte-sized) integer array as `u32`. Supported
/// dtypes: `u1 u2 u4 u8 i1 i2 i4 i8`. Negative or too-large values are errors.
pub fn read_npy_u32(path: &str) -> Result<Vec<u32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    parse_npy_u32(&bytes).map_err(|e| format!("{path}: {e}"))
}

pub fn parse_npy_u32(bytes: &[u8]) -> Result<Vec<u32>, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let major = bytes[6];
    let (hlen, hstart) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        if bytes.len() < 12 {
            return Err("truncated header".into());
        }
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
    };
    let header =
        std::str::from_utf8(bytes.get(hstart..hstart + hlen).ok_or("truncated header")?).map_err(|_| "bad header")?;
    let descr = field(header, "descr").ok_or("no descr")?;
    let descr = descr.trim_start_matches(['\'', '"']);
    let descr: String = descr.chars().take_while(|&c| c != '\'' && c != '"').collect();
    let fortran = field(header, "fortran_order").ok_or("no fortran_order")?;
    if fortran.starts_with("True") {
        return Err("fortran-ordered arrays are not supported".into());
    }
    let shape = field(header, "shape").ok_or("no shape")?;
    let shape: String = shape.trim_start_matches('(').chars().take_while(|&c| c != ')').collect();
    let dims: Vec<usize> = shape
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|_| "bad shape")?;
    if dims.len() != 1 {
        return Err(format!("expected a 1-D array, got shape {dims:?}"));
    }
    let n = dims[0];
    let (endian, kind) = descr.split_at(1);
    if endian == ">" {
        return Err("big-endian arrays are not supported".into());
    }
    let data = &bytes[hstart + hlen..];
    let width: usize = kind[1..].parse().map_err(|_| format!("unsupported dtype {descr}"))?;
    let signed = match &kind[..1] {
        "u" => false,
        "i" => true,
        _ => return Err(format!("unsupported dtype {descr}")),
    };
    if data.len() < n * width {
        return Err("truncated data".into());
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = &data[i * width..(i + 1) * width];
        let v: i128 = match (width, signed) {
            (1, false) => b[0] as i128,
            (1, true) => b[0] as i8 as i128,
            (2, false) => u16::from_le_bytes([b[0], b[1]]) as i128,
            (2, true) => i16::from_le_bytes([b[0], b[1]]) as i128,
            (4, false) => u32::from_le_bytes(b.try_into().unwrap()) as i128,
            (4, true) => i32::from_le_bytes(b.try_into().unwrap()) as i128,
            (8, false) => u64::from_le_bytes(b.try_into().unwrap()) as i128,
            (8, true) => i64::from_le_bytes(b.try_into().unwrap()) as i128,
            _ => return Err(format!("unsupported dtype {descr}")),
        };
        if !(0..=u32::MAX as i128).contains(&v) {
            return Err(format!("value {v} at {i} does not fit in u32"));
        }
        out.push(v as u32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_u2() {
        let header = "{'descr': '<u2', 'fortran_order': False, 'shape': (3,), }";
        let mut h = header.to_string();
        while !(10 + h.len() + 1).is_multiple_of(64) {
            h.push(' ');
        }
        h.push('\n');
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        b.extend((h.len() as u16).to_le_bytes());
        b.extend(h.as_bytes());
        for v in [1u16, 2, 65535] {
            b.extend(v.to_le_bytes());
        }
        assert_eq!(parse_npy_u32(&b).unwrap(), vec![1, 2, 65535]);
    }
}
