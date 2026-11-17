//! Measures what compressing a chunk with HTJ2K allocates and how long it
//! takes, for chunks that HTJ2K codes and for chunks of noise, which it
//! cannot make smaller and which are then stored raw.
//!
//! A separate test binary, because it installs a global allocator that
//! counts the allocations of each thread. Run it with
//! `cargo test --release --features htj2k --test htj2k_write_allocations --
//! --ignored --nocapture`.

#![cfg(feature = "htj2k")]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    time::{Duration, Instant},
};

use exr::{
    math::RoundingMode,
    meta::{
        attribute::{ChannelDescription, LevelMode},
        header::Header,
        BlockDescription,
    },
    prelude::*,
};

struct Counting;

thread_local! {
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

fn count_allocation(size: usize) {
    ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
    ALLOCATED_BYTES.with(|bytes| bytes.set(bytes.get() + size));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_allocation(new_size);
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The allocations and bytes that this thread made in the function.
fn count(function: impl FnOnce()) -> (usize, usize) {
    let before = (ALLOCATION_COUNT.with(Cell::get), ALLOCATED_BYTES.with(Cell::get));
    function();
    (ALLOCATION_COUNT.with(Cell::get) - before.0, ALLOCATED_BYTES.with(Cell::get) - before.1)
}

/// A bit pattern for each sample: smooth values, which HTJ2K codes, or
/// noise, which it cannot make smaller.
fn sample(noise: bool, channel: usize, x: usize, y: usize) -> u32 {
    if noise {
        let mut value = (channel as u64) << 48 ^ (y as u64) << 24 ^ x as u64;
        value = value.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        value ^= value >> 29;
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        (value >> 32) as u32
    } else {
        let value = 0.2 + 0.001 * (x + 2 * y) as f32 + 0.1 * channel as f32;
        value.to_bits()
    }
}

/// The native-endian samples of a block, in the usual layout.
fn block_samples(
    channels: &[ChannelDescription],
    noise: bool,
    rectangle: IntegerBounds,
) -> Vec<u8> {
    let mut samples = Vec::new();
    for y in 0..rectangle.size.height() {
        for (index, channel) in channels.iter().enumerate() {
            for x in 0..rectangle.size.width() {
                let bits = sample(noise, index, x + rectangle.position.x() as usize, y);
                match channel.sample_type {
                    SampleType::F16 => samples.extend_from_slice(
                        &f16::from_f32(f32::from_bits(bits)).to_bits().to_ne_bytes(),
                    ),
                    _ => samples.extend_from_slice(&bits.to_ne_bytes()),
                }
            }
        }
    }
    samples
}

#[test]
#[ignore = "a measurement, not a requirement"]
fn htj2k_compression_allocations_per_chunk() {
    let channel = |name: &str, sample_type| ChannelDescription::new(name, sample_type, false);
    let rgba_half = || {
        smallvec::smallvec![
            channel("A", SampleType::F16),
            channel("B", SampleType::F16),
            channel("G", SampleType::F16),
            channel("R", SampleType::F16),
        ]
    };
    let y_float = || smallvec::smallvec![channel("Y", SampleType::F32)];
    let tiles = BlockDescription::Tiles(TileDescription {
        tile_size: Vec2(64, 64),
        level_mode: LevelMode::Singular,
        rounding_mode: RoundingMode::Down,
    });

    let cases = [
        ("rgba half 1920x1000", rgba_half(), Vec2(1920, 1000), BlockDescription::ScanLines, false),
        ("y float 1920x1000", y_float(), Vec2(1920, 1000), BlockDescription::ScanLines, false),
        ("rgba half 200x200 64x64 tiles", rgba_half(), Vec2(200, 200), tiles, false),
        ("rgba half noise", rgba_half(), Vec2(1920, 1000), BlockDescription::ScanLines, true),
        ("y float noise", y_float(), Vec2(1920, 1000), BlockDescription::ScanLines, true),
    ];

    for (name, channels, size, blocks, noise) in cases {
        for compression in [Compression::HTJ2K32, Compression::HTJ2K256] {
            let header = Header::new("measured".into(), size, channels.clone()).with_encoding(
                compression,
                blocks,
                LineOrder::Increasing,
            );
            let blocks: Vec<_> = header
                .blocks_increasing_y_order()
                .map(|tile| {
                    let rectangle =
                        header.get_absolute_block_pixel_coordinates(tile.location).unwrap();
                    (rectangle, block_samples(&channels, noise, rectangle))
                })
                .collect();

            // The encoder may reuse its memory for a chunk of the shape of
            // the one before, and must allocate it again for a chunk of
            // another shape, such as the last chunk of an image after the
            // full ones; so the two are counted apart. The first pass warms
            // up whatever is kept, and sets the shape before the first chunk.
            let mut previous_size = None;
            let mut same = (0, 0, 0, Duration::ZERO);
            let mut other = (0, 0, 0, Duration::ZERO);
            let mut coded = 0;
            for (pass, (rectangle, samples)) in
                blocks.iter().map(|block| (0, block)).chain(blocks.iter().map(|block| (1, block)))
            {
                let samples = samples.clone();
                let uncompressed_size = samples.len();
                let mut compressed = Vec::new();
                let start = Instant::now();
                let (allocations, bytes) = count(|| {
                    compressed = compression
                        .compress_image_section_to_le(&header, samples, *rectangle)
                        .unwrap();
                });
                let time = start.elapsed();

                let tally = if previous_size == Some(rectangle.size) {
                    &mut same
                } else {
                    &mut other
                };
                if pass == 1 {
                    *tally = (tally.0 + 1, tally.1 + allocations, tally.2 + bytes, tally.3 + time);
                    coded += usize::from(compressed.len() != uncompressed_size);
                }
                previous_size = Some(rectangle.size);
            }

            let average = |(chunks, allocations, bytes, time): (usize, usize, usize, Duration)| {
                allocations.checked_div(chunks).map_or_else(
                    || "none".to_string(),
                    |allocations| {
                        format!(
                            "{} allocations, {} KiB, {} us",
                            allocations,
                            bytes / chunks / 1024,
                            time.as_micros() / chunks as u128
                        )
                    },
                )
            };
            println!(
                "{} {:?}: {} of {} chunks coded; per chunk after one of the same shape: {}; \
                 after one of another shape: {}",
                name,
                compression,
                coded,
                blocks.len(),
                average(same),
                average(other),
            );
        }
    }
}
