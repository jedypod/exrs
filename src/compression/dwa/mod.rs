// DWA / DWAB (lossy DCT) compression and decompression, ported from
// OpenEXRCores internal_dwa_compressor.h and alike.
//
// A DWA chunk is: 11 u64 counters, the channel rules (version >= 2),
// then four sections (UNKNOWN, AC, DC, RLE). Channels are classified by
// name suffix + sample type into a compression scheme; R/G/B triplets
// sharing a prefix are additionally CSC'd (Y'CbCr). All LOSSY_DCT
// channel groups of a chunk consume the same planar AC/DC streams.
//
// This module holds the two public entry points (`compress`/`decompress`) and
// the cross-cutting channel classification both directions share; each stage of
// the pipeline lives in its own submodule.

use crate::{
    compression::{reused_slice, ByteVec, Bytes},
    error::{Error, Result, UnitResult},
    meta::attribute::{ChannelList, IntegerBounds, SampleType},
};

mod channel_layout;
mod channel_rules;
mod chunk_header;
mod lossy_dct;
mod section_stream;

// public only for benchmarking
#[doc(hidden)]
pub mod discrete_cosine_transform;

#[cfg(test)]
mod tests;

use channel_layout::{
    interleave_byte_planes, locate_planar_channels, pack_rle_channels, pack_unknown_channels,
    split_scanline_channels, u16s_to_le_bytes, write_scanlines,
};
use channel_rules::{
    default_channel_rules, legacy_channel_rules, parse_channel_rules, write_relevant_channel_rules,
    Rule,
};
use chunk_header::{AcCompression, DwaHeader};
use lossy_dct::{decode_lossy_channels, encode_lossy_channels};
use section_stream::{
    decode_ac_section, decode_dc_section, decode_rle_section, decode_unknown_section,
    split_sections, zip_deconstruct_bytes,
};

#[derive(Debug, Clone, Copy, PartialEq)]
enum CompressorScheme {
    Unknown,
    LossyDct,
    Rle,
}

/// The part of a channel name after the last '.'
fn channel_suffix(name: &[u8]) -> &[u8] {
    match name.iter().rposition(|&byte| byte == b'.') {
        Some(dot) => &name[dot + 1..],
        None => name,
    }
}

#[derive(Debug, Clone)]
struct ChannelInfo {
    scheme: CompressorScheme,
    width: usize,
    height: usize,
    bytes_per_sample: usize,
    sample_type: SampleType,
    quantize_linearly: bool,
}

/// The channels that share the part of their name before the suffix,
/// and which of them holds each slot of an R/G/B triplet.
#[derive(Debug, Clone)]
struct ChannelPrefix {
    /// The first channel with this prefix.
    channel: usize,
    /// The number of bytes of that channel's name that are the prefix.
    length: usize,
    slots: [Option<usize>; 3],
}

/// The channels of a chunk as the compressor sees them, kept to classify
/// the channels of the next chunk without allocating again.
#[derive(Default)]
struct ChannelClasses {
    infos: Vec<ChannelInfo>,
    prefixes: Vec<ChannelPrefix>,
    csc_groups: Vec<[usize; 3]>,
}

/// Classify every channel and find the CSC'd R/G/B triplets, mirroring
/// "DwaCompressor_classifyChannels": a prefix map (in order of first
/// appearance, which determines group decode order) collects which channel
/// holds each triplet slot, and a group forms when all three slots are
/// filled with LOSSY_DCT channels of identical sampling.
fn classify_channels(
    channels: &ChannelList,
    rectangle: IntegerBounds,
    rules: &[Rule<'_>],
    classes: &mut ChannelClasses,
) {
    // Match channel names against the rule table, then collect R/G/B triplets
    // in the order the prefixes first appear. That order controls how the
    // shared lossy streams are consumed later.
    let ChannelClasses {
        infos,
        prefixes,
        csc_groups,
    } = classes;

    infos.clear();
    prefixes.clear();
    csc_groups.clear();

    let prefix_bytes = |prefix: &ChannelPrefix| -> &[u8] {
        &channels.list[prefix.channel].name.bytes()[..prefix.length]
    };

    for (index, channel) in channels.list.iter().enumerate() {
        let name = channel.name.bytes();
        let suffix = channel_suffix(name);
        let prefix = &name[..name.len() - suffix.len()];

        if !prefixes.iter().any(|known| prefix_bytes(known) == prefix) {
            prefixes.push(ChannelPrefix {
                channel: index,
                length: prefix.len(),
                slots: [None; 3],
            });
        }

        let mut scheme = CompressorScheme::Unknown;
        for rule in rules {
            if rule.matches(suffix, channel.sample_type) {
                scheme = rule.scheme;
                if let Some(csc_index) = rule.csc_index {
                    for known in prefixes.iter_mut() {
                        if prefix_bytes(known) == prefix {
                            known.slots[csc_index] = Some(index);
                        }
                    }
                }
            }
        }

        let sampling_x = channel.sampling.x().max(1);
        let sampling_y = channel.sampling.y().max(1);
        infos.push(ChannelInfo {
            scheme,
            width: (rectangle.size.width() + sampling_x - 1) / sampling_x,
            height: (rectangle.size.height() + sampling_y - 1) / sampling_y,
            bytes_per_sample: channel.sample_type.bytes_per_sample(),
            sample_type: channel.sample_type,
            quantize_linearly: channel.quantize_linearly,
        });
    }

    csc_groups.extend(prefixes.iter().filter_map(|prefix| {
        let [Some(r), Some(g), Some(b)] = prefix.slots else {
            return None;
        };
        let all_lossy =
            [r, g, b].iter().all(|&index| infos[index].scheme == CompressorScheme::LossyDct);
        let same_sampling = channels.list[r].sampling == channels.list[g].sampling
            && channels.list[r].sampling == channels.list[b].sampling;
        (all_lossy && same_sampling).then_some([r, g, b])
    }));
}

pub fn compress(
    channels: &ChannelList,
    uncompressed_ne: ByteVec,
    rectangle: IntegerBounds,
    compression_level: Option<f32>,
) -> Result<ByteVec> {
    if uncompressed_ne.is_empty() {
        return Ok(vec![]);
    }

    let uncompressed_le =
        crate::compression::convert_current_to_little_endian(uncompressed_ne, channels, rectangle)?;

    // The chunk is written in the same section order the decoder expects:
    // header + rules, then UNKNOWN, AC, DC and RLE payloads.
    let rules = default_channel_rules();
    let rule_bytes = write_relevant_channel_rules(&rules, channels)?;
    let mut classes = ChannelClasses::default();
    classify_channels(channels, rectangle, &rules, &mut classes);
    let (channel_infos, csc_groups) = (classes.infos, classes.csc_groups);
    let channel_bytes =
        split_scanline_channels(&uncompressed_le, channels, &channel_infos, rectangle)?;

    let unknown_planar =
        pack_unknown_channels(&channel_infos, &channel_bytes, CompressorScheme::Unknown);
    let rle_raw = pack_rle_channels(&channel_infos, &channel_bytes);

    let quant_base_error = compression_level.unwrap_or(45.0) / 100000.0;
    let (ac_values, dc_values) =
        encode_lossy_channels(&channel_infos, &csc_groups, &channel_bytes, quant_base_error)?;

    let unknown_compressed = if unknown_planar.is_empty() {
        Vec::new()
    } else {
        crate::compression::zlib_compress(&unknown_planar, 9)?
    };

    let ac_compression = AcCompression::StaticHuffman;
    let ac_compressed = if ac_values.is_empty() {
        Vec::new()
    } else {
        crate::compression::piz::huffman::compress(&ac_values)?
    };

    let dc_compressed = if dc_values.is_empty() {
        Vec::new()
    } else {
        let mut dc_bytes = u16s_to_le_bytes(&dc_values);
        zip_deconstruct_bytes(&mut dc_bytes);
        crate::compression::zlib_compress(&dc_bytes, 9)?
    };

    let (rle_uncompressed_size, rle_compressed) = if rle_raw.is_empty() {
        (0, Vec::new())
    } else {
        let rle_tokens = super::rle::pack_rle_tokens(&rle_raw);
        let compressed = crate::compression::zlib_compress(&rle_tokens, 9)?;
        (rle_tokens.len(), compressed)
    };

    let header = DwaHeader {
        version: 2,
        unknown_uncompressed_size: unknown_planar.len(),
        unknown_compressed_size: unknown_compressed.len(),
        ac_compressed_size: ac_compressed.len(),
        dc_compressed_size: dc_compressed.len(),
        rle_compressed_size: rle_compressed.len(),
        rle_uncompressed_size,
        rle_raw_size: rle_raw.len(),
        ac_count: ac_values.len(),
        dc_count: dc_values.len(),
        ac_compression,
    };

    let mut out = Vec::with_capacity(
        11 * 8
            + rule_bytes.len()
            + unknown_compressed.len()
            + ac_compressed.len()
            + dc_compressed.len()
            + rle_compressed.len(),
    );
    header.write(&mut out);
    out.extend_from_slice(&rule_bytes);
    out.extend_from_slice(&unknown_compressed);
    out.extend_from_slice(&ac_compressed);
    out.extend_from_slice(&dc_compressed);
    out.extend_from_slice(&rle_compressed);
    Ok(out)
}

/// The buffers of the decoder, kept to decode the next chunk without
/// allocating them again.
#[derive(Default)]
pub(super) struct DecodeScratch {
    classes: ChannelClasses,

    /// The output of zlib, where that is not yet the content of a section.
    inflated: Vec<u8>,
    unknown_planar: Vec<u8>,
    ac_packed: Vec<u16>,
    dc_packed: Vec<u16>,
    rle_planar: Vec<u8>,

    /// The RLE channels with the bytes of each sample together.
    rle_bytes: Vec<u8>,

    /// Where each UNKNOWN channel starts in `unknown_planar`,
    /// and each RLE channel in `rle_planar` and in `rle_bytes`.
    channel_starts: Vec<usize>,

    lossy: lossy_dct::DecodeScratch,
}

pub fn decompress(
    channels: &ChannelList,
    compressed_le: ByteVec,
    rectangle: IntegerBounds,
    expected_byte_size: usize,
    pedantic: bool,
) -> Result<ByteVec> {
    // the writer stores chunks raw when compression would not have helped
    if compressed_le.len() == expected_byte_size {
        return crate::compression::convert_little_endian_to_current(
            compressed_le,
            channels,
            rectangle,
        );
    }

    let mut decompressed_ne = vec![0_u8; expected_byte_size];
    decompress_into(
        channels,
        &compressed_le,
        rectangle,
        pedantic,
        &mut crate::compression::DecodeScratch::default(),
        &mut decompressed_ne,
    )?;

    Ok(decompressed_ne)
}

/// Decompress a chunk that is not stored raw into exactly the bytes of
/// the pixels of the rectangle.
pub(super) fn decompress_into(
    channels: &ChannelList,
    compressed_le: Bytes<'_>,
    rectangle: IntegerBounds,
    _pedantic: bool,
    scratch: &mut crate::compression::DecodeScratch,
    decompressed_ne: &mut [u8],
) -> UnitResult {
    if compressed_le.is_empty() {
        decompressed_ne.fill(0);
        return Ok(());
    }

    let crate::compression::DecodeScratch {
        inflater,
        piz,
        dwa: scratch,
        ..
    } = scratch;

    let mut input = compressed_le;
    let header = DwaHeader::parse(&mut input)?;

    let rules = if header.version < 2 {
        legacy_channel_rules()
    } else {
        parse_channel_rules(&mut input)?
    };

    classify_channels(channels, rectangle, &rules, &mut scratch.classes);
    let channel_infos = scratch.classes.infos.as_slice();
    let csc_groups = scratch.classes.csc_groups.as_slice();

    if channel_infos.iter().any(|info| {
        info.scheme == CompressorScheme::LossyDct && info.sample_type == SampleType::U32
    }) {
        return Err(Error::unsupported("DWA lossy DCT compression of u32 channels"));
    }

    let [unknown_section, ac_section, dc_section, rle_section] = split_sections(input, &header)?;

    let unknown_planar =
        decode_unknown_section(unknown_section, &header, inflater, &mut scratch.unknown_planar)?;
    let ac_packed = decode_ac_section(
        ac_section,
        &header,
        inflater,
        piz.huffman_mut(),
        &mut scratch.inflated,
        &mut scratch.ac_packed,
    )?;
    let dc_packed = decode_dc_section(
        dc_section,
        &header,
        inflater,
        &mut scratch.inflated,
        &mut scratch.dc_packed,
    )?;
    let rle_planar = decode_rle_section(
        rle_section,
        &header,
        inflater,
        &mut scratch.inflated,
        &mut scratch.rle_planar,
    )?;

    let channel_starts = &mut scratch.channel_starts;
    channel_starts.clear();
    channel_starts.resize(channel_infos.len(), 0);
    locate_planar_channels(
        channel_infos,
        CompressorScheme::Unknown,
        unknown_planar,
        channel_starts,
    )?;
    locate_planar_channels(channel_infos, CompressorScheme::Rle, rle_planar, channel_starts)?;

    let rle_bytes = reused_slice(&mut scratch.rle_bytes, rle_planar.len());
    for (info, &start) in channel_infos.iter().zip(channel_starts.iter()) {
        if info.scheme == CompressorScheme::Rle {
            let range = start..start + info.width * info.height * info.bytes_per_sample;
            interleave_byte_planes(
                &rle_planar[range.clone()],
                info.bytes_per_sample,
                &mut rle_bytes[range],
            );
        }
    }

    decode_lossy_channels(channel_infos, csc_groups, ac_packed, dc_packed, &mut scratch.lossy)?;

    write_scanlines(
        channels,
        channel_infos,
        rectangle,
        &scratch.lossy.samples,
        unknown_planar,
        rle_bytes,
        channel_starts,
        decompressed_ne,
    )?;

    crate::compression::convert_slice_little_endian_to_current(decompressed_ne, channels, rectangle)
}
