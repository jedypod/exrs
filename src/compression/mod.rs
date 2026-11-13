//! Contains the compression attribute definition
//! and methods to compress and decompress data.

// private modules make non-breaking changes easier
mod b44;

// public only for benchmarking
#[doc(hidden)]
pub mod dwa;

mod piz;
mod pxr24;
mod rle;
mod zip;

use std::convert::TryInto;

use crate::{
    error::{usize_to_i32, Error, Result, UnitResult},
    meta::{
        attribute::{ChannelList, IntegerBounds, SampleType},
        header::Header,
    },
};

/// A byte vector.
pub type ByteVec = Vec<u8>;

/// A byte slice.
pub type Bytes<'s> = &'s [u8];

/// The temporary memory a decoder needs for one block.
///
/// Decoding a block with [`Compression::decompress_image_section_from_le_into`]
/// borrows every buffer it needs from this value instead of allocating it.
/// The buffers grow to fit the largest block decoded so far and are kept, so
/// that decoding allocates nothing once a block of each size has been seen.
///
/// Create one with `DecodeScratch::default()`, which allocates nothing, and
/// keep it for as long as blocks are being decoded: one per thread that
/// decodes. Any scratch can decode any block of any file.
#[derive(Default)]
pub struct DecodeScratch {
    inflater: Inflater,

    /// The bytes of a block before they are rearranged into the output
    /// layout, for the methods that have such a stage.
    packed_bytes: ByteVec,

    b44_channels: Vec<b44::ChannelData>,
    piz: piz::DecodeScratch,
    dwa: dwa::DecodeScratch,
}

impl std::fmt::Debug for DecodeScratch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("DecodeScratch").finish_non_exhaustive()
    }
}

thread_local! {
    // The scratch of the functions that are not given one. Allocating
    // memory is cheap, but faulting in fresh pages for every block is not.
    static THREAD_SCRATCH: std::cell::Cell<DecodeScratch> =
        std::cell::Cell::new(DecodeScratch::default());
}

/// Lend the scratch of this thread to the function, for the functions that
/// are not given a scratch. A thread keeps its scratch until it exits.
fn with_thread_scratch<T>(decode: impl FnOnce(&mut DecodeScratch) -> T) -> T {
    THREAD_SCRATCH.with(|thread_scratch| {
        let mut scratch = thread_scratch.take();
        let result = decode(&mut scratch);
        thread_scratch.set(scratch);
        result
    })
}

/// The first `length` elements of a reused buffer, growing it if it is
/// shorter. The contents are whatever an earlier use left there.
fn reused_slice<T: Copy + Default>(buffer: &mut Vec<T>, length: usize) -> &mut [T] {
    reused_slice_of(buffer, length, T::default())
}

/// The first `length` elements of a reused buffer, replacing it with a
/// longer one of the zero value if it is shorter.
fn reused_slice_of<T: Copy>(buffer: &mut Vec<T>, length: usize, zero: T) -> &mut [T] {
    if buffer.len() < length {
        // The contents need not be kept, so the longer buffer is requested
        // zeroed from the allocator. That is cheaper than growing the buffer
        // and filling the new part: memory that is never written to, as when
        // a damaged file claims a huge size, is then never touched at all.
        *buffer = Vec::new();
        *buffer = vec![zero; length];
    }

    &mut buffer[..length]
}

/// Inflates zlib streams, keeping the inflate state between streams.
#[derive(Default)]
struct Inflater {
    /// Created on first use, as it allocates.
    state: Option<zlib_rs::Inflate>,
}

impl Inflater {
    /// Inflate a whole zlib stream into `output`, returning the number of
    /// bytes written. It is an error if the stream is damaged, incomplete,
    /// or holds more than `output` can take.
    fn inflate(&mut self, compressed: &[u8], output: &mut [u8]) -> Result<usize> {
        const WINDOW_BITS: u8 = 15;

        let state = match &mut self.state {
            Some(state) => {
                state.reset(true);
                state
            }
            none => none.insert(zlib_rs::Inflate::new(true, WINDOW_BITS)),
        };

        match state.decompress(compressed, output, zlib_rs::InflateFlush::Finish) {
            Ok(zlib_rs::Status::StreamEnd) => Ok(state.total_out() as usize),
            _ => Err(Error::invalid("zlib-compressed data malformed")),
        }
    }
}

/// Specifies which compression method to use.
///
/// Use uncompressed data for fastest loading and writing speeds.
/// Use RLE compression for fast loading and writing with slight memory savings.
/// Use ZIP compression for slow processing with large memory savings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Compression {
    /// Store uncompressed values.
    /// Produces large files that can be read and written very quickly.
    /// Consider using RLE instead, as it provides some compression with almost
    /// equivalent speed.
    Uncompressed,

    /// Produces slightly smaller files
    /// that can still be read and written rather quickly.
    /// The compressed file size is usually between 60 and 75 percent of the
    /// uncompressed size. Works best for images with large flat areas, such
    /// as masks and abstract graphics. This compression method is lossless.
    RLE,

    /// Uses ZIP compression to compress each line. Slowly produces small images
    /// which can be read with moderate speed. This compression method is
    /// lossless. Might be slightly faster but larger than `ZIP16´.
    ZIP1, /* TODO ZIP { individual_lines: bool, compression_level: Option<u8> }  // TODO
           * specify zip compression level? */
    /// Uses ZIP compression to compress blocks of 16 lines. Slowly produces
    /// small images which can be read with moderate speed. This compression
    /// method is lossless. Might be slightly slower but smaller than
    /// `ZIP1´.
    ZIP16, // TODO collapse with ZIP1

    /// PIZ compression works well for noisy and natural images. Works better
    /// with larger tiles. Only supported for flat images, but not for deep
    /// data. This compression method is lossless.
    // A wavelet transform is applied to the pixel data, and the result is Huffman-
    // encoded. This scheme tends to provide the best compression ratio for the types of
    // images that are typically processed at Industrial Light & Magic. Files are
    // compressed and decompressed at roughly the same speed. For photographic
    // images with film grain, the files are reduced to between 35 and 55 percent of their
    // uncompressed size.
    // PIZ compression works well for scan-line based files, and also for tiled files with
    // large tiles, but small tiles do not shrink much. (PIZ-compressed data start with a
    // relatively long header; if the input to the compressor is short, adding the header
    // tends to offset any size reduction of the input.)
    PIZ,

    /// Like `ZIP1`, but reduces precision of `f32` images to `f24`.
    /// Therefore, this is lossless compression for `f16` and `u32` data, lossy
    /// compression for `f32` data. This compression method works well for
    /// depth buffers and similar images, where the possible range of values
    /// is very large, but where full 32-bit floating-point accuracy is not
    /// necessary. Rounding improves compression significantly by
    /// eliminating the pixels' 8 least significant bits, which tend to be
    /// very noisy, and therefore difficult to compress. This produces
    /// really small image files. Only supported for flat images, not for deep
    /// data.
    // After reducing 32-bit floating-point data to 24 bits by rounding (while leaving 16-bit
    // floating-point data unchanged), differences between horizontally adjacent pixels
    // are compressed with zlib, similar to ZIP. PXR24 compression preserves image
    // channels of type HALF and UINT exactly, but the relative error of FLOAT data
    // increases to about ???.
    PXR24, // TODO specify zip compression level?

    /// This is a lossy compression method for f16 images.
    /// It's the predecessor of the `B44A` compression,
    /// which has improved compression rates for uniformly colored areas.
    /// You should probably use `B44A` instead of the plain `B44`.
    ///
    /// Only supported for flat images, not for deep data.
    // lossy 4-by-4 pixel block compression,
    // flat fields are compressed more
    // Channels of type HALF are split into blocks of four by four pixels or 32 bytes. Each
    // block is then packed into 14 bytes, reducing the data to 44 percent of their
    // uncompressed size. When B44 compression is applied to RGB images in
    // combination with luminance/chroma encoding (see below), the size of the
    // compressed pixels is about 22 percent of the size of the original RGB data.
    // Channels of type UINT or FLOAT are not compressed.
    // Decoding is fast enough to allow real-time playback of B44-compressed OpenEXR
    // image sequences on commodity hardware.
    // The size of a B44-compressed file depends on the number of pixels in the image,
    // but not on the data in the pixels. All images with the same resolution and the same
    // set of channels have the same size. This can be advantageous for systems that
    // support real-time playback of image sequences; the predictable file size makes it
    // easier to allocate space on storage media efficiently.
    // B44 compression is only supported for flat images.
    B44, // TODO B44 { optimize_uniform_areas: bool }

    /// This is a lossy compression method for f16 images.
    /// All f32 and u32 channels will be stored without compression.
    /// All the f16 pixels are divided into 4x4 blocks.
    /// Each block is then compressed as a whole.
    ///
    /// The 32 bytes of a block will require only ~14 bytes after compression,
    /// independent of the actual pixel contents. With chroma subsampling,
    /// a block will be compressed to ~7 bytes.
    /// Uniformly colored blocks will be compressed to ~3 bytes.
    ///
    /// The 512 bytes of an f32 block will not be compressed at all.
    ///
    /// Should be fast enough for realtime playback.
    /// Only supported for flat images, not for deep data.
    B44A, // TODO collapse with B44

    /// Lossy DCT-based compression (DreamWorks Animation), 32 scanlines per
    /// block. Partial buffer access friendly.
    /// Decoding support is implemented. Encoding is not implemented and returns
    /// an error.
    DWAA(Option<f32>),

    /// Lossy DCT-based compression (DreamWorks Animation), 256 scanlines per
    /// block. Better compression ratio for full frames.
    /// Decoding support is implemented. Encoding is not implemented and returns
    /// an error.
    DWAB(Option<f32>),

    /// __This lossy compression is not yet supported by this implementation.__
    // High-Throughput JPEG 2000 (32 lines)
    HTJ2K32,

    /// __This lossy compression is not yet supported by this implementation.__
    // High-Throughput JPEG 2000 (256 lines)
    HTJ2K256,
}

impl std::fmt::Display for Compression {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} compression",
            match self {
                Self::Uncompressed => "no",
                Self::RLE => "rle",
                Self::ZIP1 => "zip line",
                Self::ZIP16 => "zip block",
                Self::B44 => "b44",
                Self::B44A => "b44a",
                Self::DWAA(_) => "dwaa",
                Self::DWAB(_) => "dwab",
                Self::PIZ => "piz",
                Self::PXR24 => "pxr24",
                Self::HTJ2K32 => "ht j2k 32",
                Self::HTJ2K256 => "ht j2k 256",
            }
        )
    }
}

impl Compression {
    /// Compress the image section, converting from native endian into with
    /// little-endian format.
    pub fn compress_image_section_to_le(
        self,
        header: &Header,
        uncompressed_native_endian: ByteVec,
        pixel_section: IntegerBounds,
    ) -> Result<ByteVec> {
        let max_tile_size = header.max_block_pixel_size();

        assert!(
            pixel_section.validate(Some(max_tile_size)).is_ok(),
            "decompress tile coordinate bug"
        );
        if header.deep {
            assert!(self.supports_deep_data());
        }

        use self::Compression::*;
        let compressed_little_endian = match self {
            Uncompressed => {
                return convert_current_to_little_endian(
                    uncompressed_native_endian,
                    &header.channels,
                    pixel_section,
                );
            }

            // we need to clone here, because we might have to fallback to the
            // uncompressed data later (when compressed data is larger than raw
            // data)
            ZIP16 => zip::compress_bytes(
                &header.channels,
                uncompressed_native_endian.clone(),
                pixel_section,
            ),
            ZIP1 => zip::compress_bytes(
                &header.channels,
                uncompressed_native_endian.clone(),
                pixel_section,
            ),
            RLE => rle::compress_bytes(
                &header.channels,
                uncompressed_native_endian.clone(),
                pixel_section,
            ),
            PIZ => {
                piz::compress(&header.channels, uncompressed_native_endian.clone(), pixel_section)
            }
            PXR24 => {
                pxr24::compress(&header.channels, uncompressed_native_endian.clone(), pixel_section)
            }
            B44 => b44::compress(
                &header.channels,
                uncompressed_native_endian.clone(),
                pixel_section,
                false,
            ),
            B44A => b44::compress(
                &header.channels,
                uncompressed_native_endian.clone(),
                pixel_section,
                true,
            ),
            DWAA(level) | DWAB(level) => dwa::compress(
                &header.channels,
                uncompressed_native_endian.clone(),
                pixel_section,
                level,
            ),
            _ => {
                return Err(Error::unsupported(format!(
                    "yet unimplemented compression method: {self}"
                )));
            }
        };

        let compressed_little_endian = compressed_little_endian
            .map_err(|_| Error::invalid(format!("pixels cannot be compressed ({self})")))?;

        if self == Uncompressed || compressed_little_endian.len() < uncompressed_native_endian.len()
        {
            // only write compressed if it actually is smaller than raw
            Ok(compressed_little_endian)
        } else {
            // if we do not use compression, manually convert uncompressed data
            convert_current_to_little_endian(
                uncompressed_native_endian,
                &header.channels,
                pixel_section,
            )
        }
    }

    /// Decompress the image section from bytes of little-endian format,
    /// returning native-endian format.
    ///
    /// The temporary memory of the decoder is kept by the calling thread
    /// until the thread exits, so that the next section does not allocate
    /// it again. Use `decompress_image_section_from_le_into` to decide
    /// where that memory lives and when it is freed.
    pub fn decompress_image_section_from_le(
        self,
        header: &Header,
        compressed_le: ByteVec,
        pixel_section: IntegerBounds,
        pedantic: bool,
    ) -> Result<ByteVec> {
        let expected_byte_size = self.validate_image_section(header, pixel_section);

        // note: always true where self == Uncompressed
        if compressed_le.len() == expected_byte_size {
            // the compressed data was larger than the raw data, so the small
            // raw data has been written: the bytes are moved all the way
            return convert_little_endian_to_current(
                compressed_le,
                &header.channels,
                pixel_section,
            );
        }

        // the size comes from the header: do not allocate what cannot be filled
        self.validate_compressed_size(compressed_le.len(), expected_byte_size)?;

        let mut decompressed_ne = vec![0_u8; expected_byte_size];
        with_thread_scratch(|scratch| {
            self.decompress_image_section_from_le_into(
                header,
                &compressed_le,
                pixel_section,
                pedantic,
                scratch,
                &mut decompressed_ne,
            )
        })?;

        Ok(decompressed_ne)
    }

    /// Decompress the image section from bytes of little-endian format
    /// into the start of `decompressed_ne`, in native-endian format,
    /// and return the number of bytes written, which is the pixel count
    /// of the section times the bytes per pixel of the header.
    ///
    /// The bytes have the layout that `decompress_image_section_from_le`
    /// returns. It is an error if `decompressed_ne` is too short.
    ///
    /// All temporary memory is taken from `scratch`, so no memory is
    /// allocated once the scratch has grown to fit the largest section.
    pub fn decompress_image_section_from_le_into(
        self,
        header: &Header,
        compressed_le: Bytes<'_>,
        pixel_section: IntegerBounds,
        pedantic: bool,
        scratch: &mut DecodeScratch,
        decompressed_ne: &mut [u8],
    ) -> Result<usize> {
        let expected_byte_size = self.validate_image_section(header, pixel_section);
        let channels = &header.channels;

        let decompressed_ne = decompressed_ne
            .get_mut(..expected_byte_size)
            .ok_or_else(|| Error::invalid("decompressed block exceeds the destination size"))?;

        // note: always true where self == Uncompressed
        if compressed_le.len() == expected_byte_size {
            // the compressed data was larger than the raw data, so the small
            // raw data has been written
            decompressed_ne.copy_from_slice(compressed_le);
            convert_slice_little_endian_to_current(decompressed_ne, channels, pixel_section)?;
            return Ok(expected_byte_size);
        }

        self.validate_compressed_size(compressed_le.len(), expected_byte_size)?;

        use self::Compression::*;
        let result = match self {
            // uncompressed data always has the expected size
            Uncompressed => Err(Error::invalid("decompressed data")),
            ZIP16 | ZIP1 => zip::decompress_bytes_into(
                channels,
                compressed_le,
                pixel_section,
                pedantic,
                scratch,
                decompressed_ne,
            ),
            RLE => rle::decompress_bytes_into(
                channels,
                compressed_le,
                pixel_section,
                pedantic,
                scratch,
                decompressed_ne,
            ),
            PIZ => piz::decompress_into(
                channels,
                compressed_le,
                pixel_section,
                pedantic,
                &mut scratch.piz,
                decompressed_ne,
            ),
            PXR24 => pxr24::decompress_into(
                channels,
                compressed_le,
                pixel_section,
                pedantic,
                scratch,
                decompressed_ne,
            ),
            B44 | B44A => b44::decompress_into(
                channels,
                compressed_le,
                pixel_section,
                pedantic,
                scratch,
                decompressed_ne,
            ),
            DWAA(_) | DWAB(_) => dwa::decompress_into(
                channels,
                compressed_le,
                pixel_section,
                pedantic,
                scratch,
                decompressed_ne,
            ),
            _ => {
                return Err(Error::unsupported(format!(
                    "yet unimplemented compression method: {self}"
                )));
            }
        };

        // map all errors to compression errors
        result.map_err(|decompression_error| match decompression_error {
            Error::NotSupported(message) => Error::unsupported(format!(
                "yet unimplemented compression special case ({message})"
            )),

            error => Error::invalid(format!("compressed {self:?} data ({error})")),
        })?;

        Ok(expected_byte_size)
    }

    /// Refuse a section that is larger than any that so few compressed bytes
    /// can decompress to. The size of a section comes from the header, and a
    /// damaged header can claim any size, so this is checked before making
    /// room for the section.
    fn validate_compressed_size(
        self,
        compressed_byte_size: usize,
        expected_byte_size: usize,
    ) -> UnitResult {
        // deflate codes at most 258 bytes with two bits
        const ZIP: usize = 1032;

        use self::Compression::*;
        let max_expansion = match self {
            // a token of two bytes repeats a byte at most 128 times
            RLE => 64,

            ZIP1 | ZIP16 => ZIP,

            // decoding widens three inflated bytes to an `f32`
            PXR24 => ZIP * 2,

            // a bit decodes to at most 29 values of two bytes
            PIZ => 8 * 29 * 2,

            // three bytes unpack to at most 16 values of two bytes
            B44 | B44A => 11,

            // an empty chunk decodes to zeros
            DWAA(_) | DWAB(_) if compressed_byte_size == 0 => return Ok(()),

            // two inflated bytes hold the DC value of a block of 64 samples
            // of up to four bytes, with any number of AC values being zero
            DWAA(_) | DWAB(_) => ZIP * 128,

            // not stored raw, or not implemented, which is reported later
            Uncompressed | HTJ2K32 | HTJ2K256 => return Ok(()),
        };

        if expected_byte_size > compressed_byte_size.saturating_mul(max_expansion) {
            return Err(Error::invalid("block size exceeds what its data can decompress to"));
        }

        Ok(())
    }

    /// The byte size of the uncompressed section.
    /// Panics for a section that the header cannot contain.
    fn validate_image_section(self, header: &Header, pixel_section: IntegerBounds) -> usize {
        let max_tile_size = header.max_block_pixel_size();

        assert!(
            pixel_section.validate(Some(max_tile_size)).is_ok(),
            "decompress tile coordinate bug"
        );
        if header.deep {
            assert!(self.supports_deep_data());
        }

        pixel_section.size.area() * header.channels.bytes_per_pixel // FIXME this needs to account for subsampling anywhere
    }

    /// For scan line images and deep scan line images, one or more scan lines
    /// may be stored together as a scan line block. The number of scan
    /// lines per block depends on how the pixel data are compressed.
    pub const fn scan_lines_per_block(self) -> usize {
        use self::Compression::*;
        match self {
            Uncompressed | RLE | ZIP1 => 1,
            ZIP16 | PXR24 => 16,
            PIZ | B44 | B44A | DWAA(_) | HTJ2K32 => 32,
            DWAB(_) | HTJ2K256 => 256,
        }
    }

    /// Deep data can only be compressed using RLE or ZIP compression.
    pub const fn supports_deep_data(self) -> bool {
        use self::Compression::*;
        match self {
            Uncompressed | RLE | ZIP1 => true,

            ZIP16 | PXR24 | PIZ | B44 | B44A | DWAA(_) | DWAB(_) | HTJ2K256 | HTJ2K32 => false,
        }
    }

    /// Most compression methods will reconstruct the exact pixel bytes,
    /// but some might throw away unimportant data for specific types of
    /// samples.
    pub fn is_lossless_for(self, sample_type: SampleType) -> bool {
        use self::Compression::*;
        match self {
            PXR24 => sample_type != SampleType::F32, // pxr reduces f32 to f24
            // B44 only compresses f16 values; other sample types are left
            // uncompressed.
            B44 | B44A => sample_type != SampleType::F16,
            Uncompressed | RLE | ZIP1 | ZIP16 | PIZ | HTJ2K32 | HTJ2K256 => true,
            DWAB(_) | DWAA(_) => false,
        }
    }

    /// Most compression methods will reconstruct the exact pixel bytes,
    /// but some might throw away unimportant data in some cases.
    pub fn may_loose_data(self) -> bool {
        use self::Compression::*;
        match self {
            Uncompressed | RLE | ZIP1 | ZIP16 | PIZ | HTJ2K32 | HTJ2K256 => false,
            PXR24 | B44 | B44A | DWAB(_) | DWAA(_) => true,
        }
    }

    /// Most compression methods will reconstruct the exact pixel bytes,
    /// but some might replace NaN with zeroes.
    /// This might also depend on the sample type of the pixels.
    /// Even a compression method that supports NaN might change the bit
    /// patterns of those NaNs.
    pub fn supports_nan(self) -> bool {
        use self::Compression::*;
        match self {
            B44A | DWAB(_) | DWAA(_) => false,
            Uncompressed | PXR24 | RLE | ZIP1 | ZIP16 | PIZ | B44 | HTJ2K32 | HTJ2K256 => true,
        }
    }

    /// Most compression methods will reconstruct the exact pixel and NaN bits,
    /// but some might replace NaN bits with other NaN bits.
    /// This might also depend on the sample type of the pixels.
    pub fn preserves_nan_bits(self) -> bool {
        use self::Compression::*;
        match self {
            B44A | PXR24 | DWAB(_) | DWAA(_) => false,
            B44 | Uncompressed | RLE | ZIP1 | ZIP16 | PIZ | HTJ2K32 | HTJ2K256 => true,
        }
    }
}

/// Zlib-compress a block with zlib-rs, which deflates about twice as fast as miniz_oxide.
fn zlib_compress(bytes: &[u8], level: i32) -> Result<ByteVec> {
    let mut compressed = vec![0_u8; zlib_rs::compress_bound(bytes.len())];
    let (written, status) =
        zlib_rs::compress_slice(&mut compressed, bytes, zlib_rs::DeflateConfig::new(level));
    if status != zlib_rs::ReturnCode::Ok {
        return Err(Error::invalid("zlib compression failed"));
    }
    let length = written.len();
    compressed.truncate(length);
    Ok(compressed)
}

// see https://github.com/AcademySoftwareFoundation/openexr/blob/6a9f8af6e89547bcd370ae3cec2b12849eee0b54/OpenEXR/IlmImf/ImfMisc.cpp#L1456-L1541

#[allow(unused)] // allows the extra parameters to be unused
fn convert_current_to_little_endian(
    mut bytes: ByteVec,
    channels: &ChannelList,
    rectangle: IntegerBounds,
) -> Result<ByteVec> {
    #[cfg(target_endian = "big")]
    reverse_block_endianness(&mut bytes, channels, rectangle)?;

    Ok(bytes)
}

#[allow(unused)] // allows the extra parameters to be unused
fn convert_little_endian_to_current(
    mut bytes: ByteVec,
    channels: &ChannelList,
    rectangle: IntegerBounds,
) -> Result<ByteVec> {
    #[cfg(target_endian = "big")]
    reverse_block_endianness(&mut bytes, channels, rectangle)?;

    Ok(bytes)
}

#[allow(unused)] // allows the extra parameters to be unused
fn convert_slice_little_endian_to_current(
    bytes: &mut [u8],
    channels: &ChannelList,
    rectangle: IntegerBounds,
) -> UnitResult {
    #[cfg(target_endian = "big")]
    reverse_block_endianness(bytes, channels, rectangle)?;

    Ok(())
}

#[allow(unused)] // unused when on little endian system
fn reverse_block_endianness(
    bytes: &mut [u8],
    channels: &ChannelList,
    rectangle: IntegerBounds,
) -> UnitResult {
    let mut remaining_bytes: &mut [u8] = bytes;

    for y in rectangle.position.y()..rectangle.end().y() {
        for channel in &channels.list {
            let line_is_subsampled = mod_p(y, usize_to_i32(channel.sampling.y(), "sampling")?) != 0;
            if line_is_subsampled {
                continue;
            }

            let sample_count = rectangle.size.width() / channel.sampling.x();

            match channel.sample_type {
                SampleType::F16 => {
                    remaining_bytes =
                        convert_byte_chunks(reverse_2_bytes, 2, remaining_bytes, sample_count);
                }

                SampleType::F32 => {
                    remaining_bytes =
                        convert_byte_chunks(reverse_4_bytes, 4, remaining_bytes, sample_count);
                }

                SampleType::U32 => {
                    remaining_bytes =
                        convert_byte_chunks(reverse_4_bytes, 4, remaining_bytes, sample_count);
                }
            }
        }
    }

    // Converts groups of bytes (e.g. 2 bytes), as many groups as specified.
    // Returns a slice of the remaining bytes.
    #[inline]
    fn convert_byte_chunks(
        convert_single_value: fn(&mut [u8]),
        batch_size: usize,
        bytes: &mut [u8],
        batch_count: usize,
    ) -> &mut [u8] {
        let (line_bytes, rest) = bytes.split_at_mut(batch_count * batch_size);
        let value_byte_chunks = line_bytes.chunks_exact_mut(batch_size);

        for value_bytes in value_byte_chunks {
            convert_single_value(value_bytes);
        }

        rest
    }

    debug_assert!(remaining_bytes.is_empty(), "not all bytes were converted to little endian");
    Ok(())
}

#[inline]
fn reverse_2_bytes(bytes: &mut [u8]) {
    // this code seems like it could be optimized easily by the compiler
    let two_bytes: [u8; 2] = bytes.try_into().expect("invalid byte count");
    bytes.copy_from_slice(&[two_bytes[1], two_bytes[0]]);
}

#[inline]
fn reverse_4_bytes(bytes: &mut [u8]) {
    let four_bytes: [u8; 4] = bytes.try_into().expect("invalid byte count");
    bytes.copy_from_slice(&[four_bytes[3], four_bytes[2], four_bytes[1], four_bytes[0]]);
}

#[inline]
const fn div_p(x: i32, y: i32) -> i32 {
    if x >= 0 {
        if y >= 0 {
            x / y
        } else {
            -(x / -y)
        }
    } else if y >= 0 {
        -((y - 1 - x) / y)
    } else {
        (-y - 1 - x) / -y
    }
}

#[inline]
const fn mod_p(x: i32, y: i32) -> i32 {
    x - y * div_p(x, y)
}

/// A collection of functions used to prepare data for compression.
mod optimize_bytes {
    /// Integrate over all differences to the previous value in order to
    /// reconstruct sample values.
    pub fn differences_to_samples(buffer: &mut [u8]) {
        // The naive implementation is very simple:
        //
        // for index in 1..buffer.len() {
        //    buffer[index] = (buffer[index - 1] as i32 + buffer[index] as i32 -
        // 128) as u8; }
        //
        // But we process elements in pairs to take advantage of
        // instruction-level parallelism. When computations within a
        // pair do not depend on each other, they can be processed in
        // parallel. Since this function is responsible for a very large
        // chunk of execution time, this tweak alone improves decoding
        // performance of RLE images by 20%.
        if let Some(first) = buffer.first() {
            let mut previous = i16::from(*first);
            for chunk in &mut buffer[1..].chunks_exact_mut(2) {
                // no bounds checks here due to indices and chunk size being
                // constant
                let diff0 = i16::from(chunk[0]);
                let diff1 = i16::from(chunk[1]);
                // these two computations do not depend on each other, unlike in
                // the naive version, so they can be executed by
                // the CPU in parallel via instruction-level
                // parallelism
                let sample0 = (previous + diff0 - 128) as u8;
                let sample1 = (previous + diff0 + diff1 - 128 * 2) as u8;
                chunk[0] = sample0;
                chunk[1] = sample1;
                previous = i16::from(sample1);
            }
            // handle the remaining element at the end not processed by the loop
            // over pairs, if present
            for elem in &mut buffer[1..].chunks_exact_mut(2).into_remainder().iter_mut() {
                let sample = (previous + i16::from(*elem) - 128) as u8;
                *elem = sample;
                previous = i16::from(sample);
            }
        }
    }

    /// Derive over all values in order to produce differences to the previous
    /// value.
    pub fn samples_to_differences(buffer: &mut [u8]) {
        // naive version:
        // for index in (1..buffer.len()).rev() {
        //     buffer[index] = (buffer[index] as i32 - buffer[index - 1] as i32
        // + 128) as u8; }
        //
        // But we process elements in batches to take advantage of
        // autovectorization. If the target platform has no vector
        // instructions (e.g. 32-bit ARM without `-C target-cpu=native`)
        // this will instead take advantage of instruction-level
        // parallelism.
        if let Some(first) = buffer.first() {
            let mut previous = i16::from(*first);
            // Chunk size is 16 because we process bytes (8 bits),
            // and 8*16 = 128 bits is the size of a typical SIMD register.
            // Even WASM has 128-bit SIMD registers.
            for chunk in &mut buffer[1..].chunks_exact_mut(16) {
                // no bounds checks here due to indices and chunk size being
                // constant
                let sample0 = i16::from(chunk[0]);
                let sample1 = i16::from(chunk[1]);
                let sample2 = i16::from(chunk[2]);
                let sample3 = i16::from(chunk[3]);
                let sample4 = i16::from(chunk[4]);
                let sample5 = i16::from(chunk[5]);
                let sample6 = i16::from(chunk[6]);
                let sample7 = i16::from(chunk[7]);
                let sample8 = i16::from(chunk[8]);
                let sample9 = i16::from(chunk[9]);
                let sample10 = i16::from(chunk[10]);
                let sample11 = i16::from(chunk[11]);
                let sample12 = i16::from(chunk[12]);
                let sample13 = i16::from(chunk[13]);
                let sample14 = i16::from(chunk[14]);
                let sample15 = i16::from(chunk[15]);
                // Unlike in decoding, computations in here are truly
                // independent from each other, which enables
                // the compiler to vectorize this loop.
                // Even if the target platform has no vector instructions,
                // so using more parallelism doesn't imply doing more work,
                // and we're not really limited in how wide we can go.
                chunk[0] = (sample0 - previous + 128) as u8;
                chunk[1] = (sample1 - sample0 + 128) as u8;
                chunk[2] = (sample2 - sample1 + 128) as u8;
                chunk[3] = (sample3 - sample2 + 128) as u8;
                chunk[4] = (sample4 - sample3 + 128) as u8;
                chunk[5] = (sample5 - sample4 + 128) as u8;
                chunk[6] = (sample6 - sample5 + 128) as u8;
                chunk[7] = (sample7 - sample6 + 128) as u8;
                chunk[8] = (sample8 - sample7 + 128) as u8;
                chunk[9] = (sample9 - sample8 + 128) as u8;
                chunk[10] = (sample10 - sample9 + 128) as u8;
                chunk[11] = (sample11 - sample10 + 128) as u8;
                chunk[12] = (sample12 - sample11 + 128) as u8;
                chunk[13] = (sample13 - sample12 + 128) as u8;
                chunk[14] = (sample14 - sample13 + 128) as u8;
                chunk[15] = (sample15 - sample14 + 128) as u8;
                previous = sample15;
            }
            // Handle the remaining element at the end not processed by the loop
            // over batches, if present This is what the
            // iterator-based version of this function would look
            // like without vectorization
            for elem in &mut buffer[1..].chunks_exact_mut(16).into_remainder().iter_mut() {
                let diff = (i16::from(*elem) - previous + 128) as u8;
                previous = i16::from(*elem);
                *elem = diff;
            }
        }
    }

    use std::cell::Cell;
    thread_local! {
        // A buffer for reusing between invocations of interleaving and deinterleaving.
        // Allocating memory is cheap, but zeroing or otherwise initializing it is not.
        // Doing it hundreds of times (once per block) would be expensive.
        // This optimization brings down the time spent in interleaving from 15% to 5%.
        static SCRATCH_SPACE: Cell<Vec<u8>> = const { Cell::new(Vec::new()) };
    }

    fn with_reused_buffer<F>(length: usize, mut func: F)
    where
        F: FnMut(&mut [u8]),
    {
        SCRATCH_SPACE.with(|scratch_space| {
            // reuse a buffer if we've already initialized one
            let mut buffer = scratch_space.take();
            if buffer.len() < length {
                // Efficiently create a zeroed Vec by requesting zeroed memory
                // from the OS. This is slightly faster than a
                // `memcpy()` plus `memset()` that would happen
                // otherwise, but is not a big deal either way since it's not a
                // hot codepath.
                buffer = vec![0u8; length];
            }

            // call the function
            func(&mut buffer[..length]);

            // save the internal buffer for reuse
            scratch_space.set(buffer);
        });
    }

    /// Interleave the bytes of both halves of `separated` into `interleaved`,
    /// which must have the same length: every other byte comes from the
    /// second half.
    pub fn interleave_byte_blocks(separated: &[u8], interleaved: &mut [u8]) {
        debug_assert_eq!(separated.len(), interleaved.len());

        // Split the two halves that we are going to interleave.
        let (first_half, second_half) = separated.split_at((separated.len() + 1) / 2);
        // The first half can be 1 byte longer than the second if the length
        // of the input is odd, but the loop below only
        // processes numbers in pairs. To handle it, preserve
        // the last element of the first slice, to be handled
        // after the loop.
        let first_half_last = first_half.last();
        // Truncate the first half to match the lenght of the second one;
        // more optimizer-friendly
        let first_half_iter = &first_half[..second_half.len()];

        // Main loop that performs the interleaving
        for ((first, second), interleaved) in
            first_half_iter.iter().zip(second_half.iter()).zip(interleaved.chunks_exact_mut(2))
        {
            // The length of each chunk is known to be 2 at compile time,
            // and each index is also a constant.
            // This allows the compiler to remove the bounds checks.
            interleaved[0] = *first;
            interleaved[1] = *second;
        }

        // If the length of the slice was odd, restore the last element of
        // the first half that we saved
        if interleaved.len() % 2 == 1 {
            if let Some(value) = first_half_last {
                // we can unwrap() here because we just checked that the
                // lenght is non-zero: `% 2 == 1` will
                // fail for zero
                *interleaved.last_mut().unwrap() = *value;
            }
        }
    }

    /// Separate the bytes such that the second half contains every other byte.
    /// This performs deinterleaving - the inverse of interleaving.
    pub fn separate_bytes_fragments(source: &mut [u8]) {
        with_reused_buffer(source.len(), |separated| {
            // Split the two halves that we are going to interleave.
            let (first_half, second_half) = separated.split_at_mut((source.len() + 1) / 2);
            // The first half can be 1 byte longer than the second if the length
            // of the input is odd, but the loop below only
            // processes numbers in pairs. To handle it, preserve
            // the last element of the input, to be handled after the
            // loop.
            let last = source.last();
            let first_half_iter = &mut first_half[..second_half.len()];

            // Main loop that performs the deinterleaving
            for ((first, second), interleaved) in
                first_half_iter.iter_mut().zip(second_half.iter_mut()).zip(source.chunks_exact(2))
            {
                // The length of each chunk is known to be 2 at compile time,
                // and each index is also a constant.
                // This allows the compiler to remove the bounds checks.
                *first = interleaved[0];
                *second = interleaved[1];
            }

            // If the length of the slice was odd, restore the last element of
            // the input that we saved
            if source.len() % 2 == 1 {
                if let Some(value) = last {
                    // we can unwrap() here because we just checked that the
                    // lenght is non-zero: `% 2 == 1` will
                    // fail for zero
                    *first_half.last_mut().unwrap() = *value;
                }
            }

            // write out the results
            source.copy_from_slice(separated);
        });
    }

    #[cfg(test)]
    pub mod test {
        #[test]
        fn roundtrip_interleave() {
            let source = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            let mut modified = source.clone();

            super::separate_bytes_fragments(&mut modified);
            let mut interleaved = vec![0; modified.len()];
            super::interleave_byte_blocks(&modified, &mut interleaved);

            assert_eq!(source, interleaved);
        }

        #[test]
        fn roundtrip_derive() {
            let source = vec![0, 1, 2, 7, 4, 5, 6, 7, 13, 9, 10];
            let mut modified = source.clone();

            super::samples_to_differences(&mut modified);
            super::differences_to_samples(&mut modified);

            assert_eq!(source, modified);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{block::samples::IntoNativeSample, meta::attribute::ChannelDescription};

    #[test]
    fn roundtrip_endianness_mixed_channels() {
        let a32 = ChannelDescription::new("A", SampleType::F32, true);
        let y16 = ChannelDescription::new("Y", SampleType::F16, true);
        let channels = ChannelList::new(smallvec![a32, y16]);

        let data = vec![
            (23582740683_f32).to_ne_bytes().as_slice(),
            (35827420683_f32).to_ne_bytes().as_slice(),
            (27406832358_f32).to_f16().to_ne_bytes().as_slice(),
            (74062358283_f32).to_f16().to_ne_bytes().as_slice(),
            (52582740683_f32).to_ne_bytes().as_slice(),
            (45827420683_f32).to_ne_bytes().as_slice(),
            (15406832358_f32).to_f16().to_ne_bytes().as_slice(),
            (65062358283_f32).to_f16().to_ne_bytes().as_slice(),
        ]
        .into_iter()
        .flatten()
        .copied()
        .collect();

        roundtrip_convert_endianness(data, &channels, IntegerBounds::from_dimensions((2, 2)));
    }

    fn roundtrip_convert_endianness(
        current_endian: ByteVec,
        channels: &ChannelList,
        rectangle: IntegerBounds,
    ) {
        let little_endian =
            convert_current_to_little_endian(current_endian.clone(), channels, rectangle).unwrap();

        let current_endian_decoded =
            convert_little_endian_to_current(little_endian, channels, rectangle).unwrap();

        assert_eq!(current_endian, current_endian_decoded, "endianness conversion failed");
    }
}
