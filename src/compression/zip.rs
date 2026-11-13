// see https://github.com/openexr/openexr/blob/master/OpenEXR/IlmImf/ImfCompressor.cpp

use super::{optimize_bytes::*, *};
use crate::error::{Result, UnitResult};

// scanline decompression routine, see https://github.com/openexr/openexr/blob/master/OpenEXR/IlmImf/ImfScanLineInputFile.cpp
// 1. Uncompress the data, if necessary (If the line is uncompressed, it's in
//    XDR format, regardless of the compressor's output format.)
// 3. Convert one scan line's worth of pixel data back from the
//    machine-independent representation
// 4. Fill the frame buffer with pixel data, respective to sampling and whatnot

pub fn decompress_bytes_into(
    channels: &ChannelList,
    data_le: Bytes<'_>,
    rectangle: IntegerBounds,
    _pedantic: bool,
    scratch: &mut DecodeScratch,
    decompressed_ne: &mut [u8],
) -> UnitResult {
    let packed_le = reused_slice(&mut scratch.packed_bytes, decompressed_ne.len());
    if scratch.inflater.inflate(data_le, packed_le)? != packed_le.len() {
        return Err(Error::invalid("decompressed data"));
    }

    differences_to_samples(packed_le);
    interleave_byte_blocks(packed_le, decompressed_ne);

    super::convert_slice_little_endian_to_current(decompressed_ne, channels, rectangle)
}

pub fn compress_bytes(
    channels: &ChannelList,
    uncompressed_ne: ByteVec,
    rectangle: IntegerBounds,
) -> Result<ByteVec> {
    // see https://github.com/AcademySoftwareFoundation/openexr/blob/3bd93f85bcb74c77255f28cdbb913fdbfbb39dfe/OpenEXR/IlmImf/ImfTiledOutputFile.cpp#L750-L842
    let mut packed_le = convert_current_to_little_endian(uncompressed_ne, channels, rectangle)?;

    separate_bytes_fragments(&mut packed_le);
    samples_to_differences(&mut packed_le);

    super::zlib_compress(&packed_le, 4)
}
