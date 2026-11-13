// Moving channel samples between the layouts DWA needs: the interleaved
// scanline buffer the rest of the crate uses, the per-channel byte runs, the
// planar byte planes of the UNKNOWN/RLE sections, and back again.

use std::io::Write;

use half::{f16, slice::HalfFloatSliceExt};
use lebe::io::WriteEndian;

use super::{ChannelInfo, CompressorScheme};
use crate::{
    error::{Error, Result},
    meta::attribute::{ChannelList, IntegerBounds, SampleType},
};

pub(super) fn split_scanline_channels(
    data: &[u8],
    channels: &ChannelList,
    infos: &[ChannelInfo],
    rectangle: IntegerBounds,
) -> Result<Vec<Vec<u8>>> {
    // The scanline buffer is already in channel order, but the encoder needs
    // each channel sliced back out so the per-scheme packing matches the C
    // reference's running cursors.
    let mut per_channel: Vec<Vec<u8>> = infos
        .iter()
        .map(|info| Vec::with_capacity(info.width * info.height * info.bytes_per_sample))
        .collect();
    let mut input = data;

    for y in rectangle.position.y()..rectangle.end().y() {
        for (index, channel) in channels.list.iter().enumerate() {
            let sampling_y = channel.sampling.y().max(1) as i32;
            if y % sampling_y != 0 {
                continue;
            }

            let row_length = infos[index].width * infos[index].bytes_per_sample;
            if row_length > input.len() {
                return Err(Error::invalid("DWA input data truncated"));
            }
            let (row, rest) = input.split_at(row_length);
            per_channel[index].extend_from_slice(row);
            input = rest;
        }
    }

    if !input.is_empty() {
        return Err(Error::invalid("DWA input data size mismatch"));
    }

    Ok(per_channel)
}

pub(super) fn pack_unknown_channels(
    infos: &[ChannelInfo],
    channel_bytes: &[Vec<u8>],
    scheme: CompressorScheme,
) -> Vec<u8> {
    // UNKNOWN channels stay planar and are concatenated in channel order
    // before the zlib step.
    let total_len = infos
        .iter()
        .zip(channel_bytes)
        .filter(|(info, _)| info.scheme == scheme)
        .map(|(_, bytes)| bytes.len())
        .sum();
    let mut out = Vec::with_capacity(total_len);

    for (info, bytes) in infos.iter().zip(channel_bytes) {
        if info.scheme == scheme {
            out.extend_from_slice(bytes);
        }
    }
    out
}

pub(super) fn pack_rle_channels(infos: &[ChannelInfo], channel_bytes: &[Vec<u8>]) -> Vec<u8> {
    // RLE channels are repacked into byte planes first, then byte-RLE'd and
    // zlib-compressed by the caller.
    let total_len = infos
        .iter()
        .zip(channel_bytes)
        .filter(|(info, _)| info.scheme == CompressorScheme::Rle)
        .map(|(_, bytes)| bytes.len())
        .sum();
    let mut out = Vec::with_capacity(total_len);

    for (info, bytes) in infos.iter().zip(channel_bytes) {
        if info.scheme == CompressorScheme::Rle {
            out.extend_from_slice(&separate_byte_planes(bytes, info.bytes_per_sample));
        }
    }
    out
}

fn separate_byte_planes(interleaved: &[u8], bytes_per_sample: usize) -> Vec<u8> {
    let sample_count = interleaved.len() / bytes_per_sample;
    let mut planar = vec![0u8; interleaved.len()];

    for byte in 0..bytes_per_sample {
        for sample in 0..sample_count {
            planar[byte * sample_count + sample] = interleaved[sample * bytes_per_sample + byte];
        }
    }

    planar
}

pub(super) fn u16s_to_le_bytes(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// Find where the bytes of each channel of the given scheme start in the
/// planar buffer of that scheme, in channel order (mirrors
/// "DwaCompressor_setupChannelData"s running per-scheme cursor).
/// The starts of the channels of other schemes are left as they are.
pub(super) fn locate_planar_channels(
    infos: &[ChannelInfo],
    scheme: CompressorScheme,
    planar: &[u8],
    channel_starts: &mut [usize],
) -> Result<()> {
    let mut cursor = 0;

    for (channel_start, info) in channel_starts.iter_mut().zip(infos) {
        if info.scheme != scheme {
            continue;
        }

        *channel_start = cursor;
        cursor += info.width * info.height * info.bytes_per_sample;
        if cursor > planar.len() {
            return Err(Error::invalid("truncated DWA channel data"));
        }
    }
    Ok(())
}

/// Restore per-sample byte order from byte planes
pub(super) fn interleave_byte_planes(
    planar: &[u8],
    bytes_per_sample: usize,
    interleaved: &mut [u8],
) {
    debug_assert_eq!(planar.len(), interleaved.len());
    let sample_count = planar.len() / bytes_per_sample;
    for sample in 0..sample_count {
        for byte in 0..bytes_per_sample {
            interleaved[sample * bytes_per_sample + byte] = planar[byte * sample_count + sample];
        }
    }
}

/// Interleave the per-channel decoded data into the scanline layout the
/// rest of exrs expects: rows of "y" ascending, channels in list order
/// within each row, samples little-endian.
///
/// The UNKNOWN and RLE channels start at their `channel_starts`
/// in the buffer of their scheme.
pub(super) fn write_scanlines(
    channels: &ChannelList,
    infos: &[ChannelInfo],
    rectangle: IntegerBounds,
    lossy_samples: &[Vec<f16>],
    unknown_bytes: &[u8],
    rle_bytes: &[u8],
    channel_starts: &[usize],
    mut out: &mut [u8],
) -> Result<()> {
    // Reassemble the per-channel decoded data into the scanline layout the
    // rest of the crate expects: rows in ascending y, channels in list order.
    for y in rectangle.position.y()..rectangle.end().y() {
        for (index, channel) in channels.list.iter().enumerate() {
            let sampling_y = channel.sampling.y().max(1) as i32;
            if y % sampling_y != 0 {
                continue;
            }

            let info = &infos[index];
            let row = ((y - rectangle.position.y()) / sampling_y) as usize;

            match info.scheme {
                CompressorScheme::LossyDct => {
                    let row_samples = &lossy_samples[index][row * info.width..][..info.width];
                    match info.sample_type {
                        SampleType::F16 => {
                            out.write_as_little_endian(row_samples.reinterpret_cast())?;
                        }
                        SampleType::F32 => {
                            // Batch half conversions without allocating a
                            // temporary buffer for the entire scanline.
                            let mut floats = [0.0f32; 64];
                            for samples in row_samples.chunks(floats.len()) {
                                let floats = &mut floats[..samples.len()];
                                samples.convert_to_f32_slice(floats);
                                out.write_as_little_endian(&*floats)?;
                            }
                        }
                        // rejected before decoding
                        SampleType::U32 => {
                            return Err(Error::unsupported(
                                "DWA lossy DCT compression of u32 channels",
                            ));
                        }
                    }
                }

                CompressorScheme::Unknown | CompressorScheme::Rle => {
                    let bytes = if info.scheme == CompressorScheme::Unknown {
                        unknown_bytes
                    } else {
                        rle_bytes
                    };
                    let row_length = info.width * info.bytes_per_sample;
                    let row_start = channel_starts[index] + row * row_length;
                    out.write_all(&bytes[row_start..][..row_length])?;
                }
            }
        }
    }

    if !out.is_empty() {
        return Err(Error::invalid("DWA decoded size mismatch"));
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};

    use super::*;

    const SEED: [u8; 32] = [
        3, 128, 9, 44, 201, 17, 88, 6, 255, 61, 30, 11, 2, 121, 99, 1, 250, 77, 33, 7, 42, 13, 200,
        176, 22, 5, 66, 100, 19, 240, 8, 91,
    ];

    fn random_bytes(random: &mut impl Rng, count: usize) -> Vec<u8> {
        (0..count).map(|_| random.gen()).collect()
    }

    fn channel_info(scheme: CompressorScheme, width: usize, height: usize) -> ChannelInfo {
        // Use F16 (2 bytes) as a representative sample type; the layout code
        // only cares about the byte counts, not the semantics.
        ChannelInfo {
            scheme,
            width,
            height,
            bytes_per_sample: SampleType::F16.bytes_per_sample(),
            sample_type: SampleType::F16,
            quantize_linearly: false,
        }
    }

    /// Splitting interleaved samples into byte planes and interleaving them
    /// back must be the identity, for several samples-per-byte widths.
    #[test]
    fn byte_planes_roundtrip() {
        let mut random = rand::rngs::StdRng::from_seed(SEED);

        for bytes_per_sample in [2usize, 4] {
            for sample_count in [0usize, 1, 5, 37] {
                let original = random_bytes(&mut random, sample_count * bytes_per_sample);
                let planar = separate_byte_planes(&original, bytes_per_sample);
                let mut interleaved = vec![0_u8; planar.len()];
                interleave_byte_planes(&planar, bytes_per_sample, &mut interleaved);
                assert_eq!(interleaved, original);
            }
        }
    }

    /// The bytes of each channel, from where the channel is located.
    fn split_planar_channels(
        infos: &[ChannelInfo],
        scheme: CompressorScheme,
        planar: &[u8],
    ) -> Vec<Vec<u8>> {
        let mut channel_starts = vec![0; infos.len()];
        locate_planar_channels(infos, scheme, planar, &mut channel_starts).unwrap();

        // a planar buffer that lacks the last byte is refused
        let truncated = &planar[..planar.len() - 1];
        assert!(locate_planar_channels(infos, scheme, truncated, &mut channel_starts).is_err());

        infos
            .iter()
            .zip(channel_starts)
            .map(|(info, start)| {
                planar[start..start + info.width * info.height * info.bytes_per_sample].to_vec()
            })
            .collect()
    }

    /// Packing UNKNOWN channels into one planar buffer and locating them in
    /// it must reproduce every channel's bytes exactly.
    #[test]
    fn pack_split_unknown_roundtrip() {
        let mut random = rand::rngs::StdRng::from_seed(SEED);

        let infos = vec![
            channel_info(CompressorScheme::Unknown, 4, 3),
            channel_info(CompressorScheme::Unknown, 5, 2),
        ];
        let channel_bytes: Vec<Vec<u8>> = infos
            .iter()
            .map(|info| random_bytes(&mut random, info.width * info.height * info.bytes_per_sample))
            .collect();

        let packed = pack_unknown_channels(&infos, &channel_bytes, CompressorScheme::Unknown);
        let split = split_planar_channels(&infos, CompressorScheme::Unknown, &packed);

        assert_eq!(split, channel_bytes);
    }

    /// RLE channels are packed via byte-plane separation; locating them in
    /// the planar buffer and interleaving each plane back must recover the
    /// input bytes.
    #[test]
    fn pack_split_rle_roundtrip() {
        let mut random = rand::rngs::StdRng::from_seed(SEED);

        let infos = vec![
            channel_info(CompressorScheme::Rle, 4, 3),
            channel_info(CompressorScheme::Rle, 6, 2),
        ];
        let channel_bytes: Vec<Vec<u8>> = infos
            .iter()
            .map(|info| random_bytes(&mut random, info.width * info.height * info.bytes_per_sample))
            .collect();

        let packed = pack_rle_channels(&infos, &channel_bytes);
        // Mirror `mod.rs::decompress_into`: locate each channel in the planar
        // buffer, then interleave its byte planes back to sample order.
        let planar_per_channel = split_planar_channels(&infos, CompressorScheme::Rle, &packed);
        let decoded: Vec<Vec<u8>> = infos
            .iter()
            .zip(&planar_per_channel)
            .map(|(info, planar)| {
                let mut interleaved = vec![0_u8; planar.len()];
                interleave_byte_planes(planar, info.bytes_per_sample, &mut interleaved);
                interleaved
            })
            .collect();

        assert_eq!(decoded, channel_bytes);
    }
}
