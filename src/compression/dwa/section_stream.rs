// The four on-disk DWA payload sections (UNKNOWN, AC, DC, RLE): splitting the
// chunk body into them, the zlib inflate wrapper they share, and the
// differencing transform applied to the DC stream.

use super::chunk_header::{AcCompression, DwaHeader};
use crate::{
    compression::{piz::huffman, reused_slice, Inflater},
    error::{Error, Result},
};

/// Split the data after header + rules into the four sections, in on-disk
/// order. Errors on truncation like the C parser.
pub(super) fn split_sections<'d>(data: &'d [u8], header: &DwaHeader) -> Result<[&'d [u8]; 4]> {
    let mut rest = data;
    let mut take = |length: usize| -> Result<&'d [u8]> {
        if length > rest.len() {
            return Err(Error::invalid("truncated DWA section"));
        }
        let (section, remaining) = rest.split_at(length);
        rest = remaining;
        Ok(section)
    };

    Ok([
        take(header.unknown_compressed_size)?,
        take(header.ac_compressed_size)?,
        take(header.dc_compressed_size)?,
        take(header.rle_compressed_size)?,
    ])
}

/// Inflate a section of exactly the expected size into the start of the
/// buffer, which grows to that size if it is shorter.
fn inflate<'b>(
    inflater: &mut Inflater,
    compressed: &[u8],
    expected_size: usize,
    buffer: &'b mut Vec<u8>,
) -> Result<&'b mut [u8]> {
    // The expected size comes from the chunk header. Deflate expands at
    // most about 1032:1; refuse anything larger before making room for it.
    if expected_size > compressed.len().saturating_mul(1032) {
        return Err(Error::invalid("DWA zlib data size mismatch"));
    }

    let inflated = reused_slice(buffer, expected_size);
    let inflated_size = inflater
        .inflate(compressed, inflated)
        .map_err(|_| Error::invalid("DWA zlib data malformed"))?;

    if inflated_size != expected_size {
        return Err(Error::invalid("DWA zlib data size mismatch"));
    }
    Ok(inflated)
}

/// UNKNOWN section: raw (non-DCT-compressible) channel data,
/// zlib-compressed, planar in channel order.
pub(super) fn decode_unknown_section<'b>(
    section: &[u8],
    header: &DwaHeader,
    inflater: &mut Inflater,
    planar: &'b mut Vec<u8>,
) -> Result<&'b [u8]> {
    if header.unknown_uncompressed_size == 0 {
        return Ok(&[]);
    }
    Ok(inflate(inflater, section, header.unknown_uncompressed_size, planar)?)
}

/// AC section: RLE DCT coefficients as u16, entropy coded with either the
/// PIZ static Huffman coder or zlib.
pub(super) fn decode_ac_section<'b>(
    section: &[u8],
    header: &DwaHeader,
    inflater: &mut Inflater,
    huffman: &mut huffman::DecodeScratch,
    inflated: &mut Vec<u8>,
    values: &'b mut Vec<u16>,
) -> Result<&'b [u16]> {
    if header.ac_count == 0 {
        return Ok(&[]);
    }

    match header.ac_compression {
        AcCompression::StaticHuffman => {
            // The count comes from the chunk header, so refuse a count that
            // the section cannot decode to before making room for it.
            if header.ac_count > huffman::max_decompressed_len(section) {
                return Err(Error::invalid("DWA AC data size mismatch"));
            }

            let values = reused_slice(values, header.ac_count);
            huffman::decompress_into(section, huffman, values)?;
            Ok(values)
        }
        AcCompression::Deflate => {
            let byte_size = header
                .ac_count
                .checked_mul(2)
                .ok_or_else(|| Error::invalid("DWA zlib data size mismatch"))?;
            let bytes = inflate(inflater, section, byte_size, inflated)?;

            let values = reused_slice(values, header.ac_count);
            for (value, pair) in values.iter_mut().zip(bytes.chunks_exact(2)) {
                *value = u16::from_le_bytes([pair[0], pair[1]]);
            }
            Ok(values)
        }
    }
}

/// DC section: one u16 (half bits) per 8x8 block, zlib-compressed after
/// the "zip reconstruct" transform (differencing + byte deinterleave).
pub(super) fn decode_dc_section<'b>(
    section: &[u8],
    header: &DwaHeader,
    inflater: &mut Inflater,
    inflated: &mut Vec<u8>,
    values: &'b mut Vec<u16>,
) -> Result<&'b [u16]> {
    if header.dc_count == 0 {
        return Ok(&[]);
    }

    let byte_size = header
        .dc_count
        .checked_mul(2)
        .ok_or_else(|| Error::invalid("DWA zlib data size mismatch"))?;
    let bytes = inflate(inflater, section, byte_size, inflated)?;

    let values = reused_slice(values, header.dc_count);
    undo_zip_reconstruct(bytes, values);
    Ok(values)
}

/// RLE section: zlib, then classic byte-oriented RLE. Result is planar per
/// channel, each channel further split into byte planes.
pub(super) fn decode_rle_section<'b>(
    section: &[u8],
    header: &DwaHeader,
    inflater: &mut Inflater,
    inflated: &mut Vec<u8>,
    planar: &'b mut Vec<u8>,
) -> Result<&'b [u8]> {
    if header.rle_raw_size == 0 {
        return Ok(&[]);
    }

    // Every token unpacks to at most 127 bytes and takes at least two bytes;
    // refuse a size that the tokens cannot unpack to before making room.
    if header.rle_raw_size > header.rle_uncompressed_size.saturating_mul(64) {
        return Err(Error::invalid("DWA RLE data size mismatch"));
    }

    let tokens = inflate(inflater, section, header.rle_uncompressed_size, inflated)?;
    let planar = reused_slice(planar, header.rle_raw_size);
    let length = crate::compression::rle::unpack_rle_tokens(tokens, planar, false)?;
    Ok(&planar[..length])
}

/// Ports "internal_zip_reconstruct_bytes" for an even number of bytes, and
/// reads the result as little-endian values: undo differencing, then
/// interleave the two buffer halves.
fn undo_zip_reconstruct(source: &mut [u8], values: &mut [u16]) {
    debug_assert_eq!(source.len(), values.len() * 2);

    crate::compression::optimize_bytes::differences_to_samples(source);

    let (first_half, second_half) = source.split_at(values.len());
    for ((value, &low), &high) in values.iter_mut().zip(first_half).zip(second_half) {
        *value = u16::from_le_bytes([low, high]);
    }
}

/// Encoder-side companion applied to the DC byte stream before zlib:
/// byte-fragment separation followed by successive differencing.
pub(super) fn zip_deconstruct_bytes(bytes: &mut [u8]) {
    crate::compression::optimize_bytes::separate_bytes_fragments(bytes);
    crate::compression::optimize_bytes::samples_to_differences(bytes);
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};

    use super::*;

    const SEED: [u8; 32] = [
        19, 240, 8, 91, 3, 128, 9, 44, 201, 17, 88, 6, 255, 61, 30, 11, 2, 121, 99, 1, 250, 77, 33,
        7, 42, 13, 200, 176, 22, 5, 66, 100,
    ];

    /// The encoder-side DC transform (`zip_deconstruct_bytes`) and the
    /// decoder-side reconstruction (`undo_zip_reconstruct`) must be inverses.
    #[test]
    fn zip_deconstruct_reconstruct_roundtrip() {
        let mut random = rand::rngs::StdRng::from_seed(SEED);

        for length in [0usize, 2, 4, 6, 18, 64, 130] {
            let original: Vec<u8> = (0..length).map(|_| random.gen()).collect();

            let mut deconstructed = original.clone();
            zip_deconstruct_bytes(&mut deconstructed);

            let mut values = vec![0_u16; length / 2];
            undo_zip_reconstruct(&mut deconstructed, &mut values);
            let reconstructed: Vec<u8> =
                values.iter().flat_map(|value| value.to_le_bytes()).collect();

            assert_eq!(reconstructed, original, "failed at length {length}");
        }
    }
}
