//! Decompressing a chunk into memory that the caller provides must produce
//! the bytes that `UncompressedBlock::decompress_chunk` returns, must not
//! allocate once its scratch is warm, and must refuse a destination that is
//! too small and sizes that a damaged chunk claims.
//!
//! A separate test binary, because it installs a global allocator that counts
//! the allocations of each thread, and refuses any single allocation over
//! 1 GiB, as a memory-constrained system would.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    convert::TryInto,
    io::Cursor,
};

use exr::{
    block::{
        chunk::{BlockLocation, Chunk, ChunkRef, CompressedBlock},
        UncompressedBlock,
    },
    compression::DecodeScratch,
    meta::MetaData,
    prelude::*,
};
use smallvec::smallvec;

struct Counting;

const MAX_ALLOCATION: usize = 1 << 30;

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
        if layout.size() > MAX_ALLOCATION {
            return std::ptr::null_mut();
        }
        count_allocation(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > MAX_ALLOCATION {
            return std::ptr::null_mut();
        }
        count_allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > MAX_ALLOCATION {
            return std::ptr::null_mut();
        }
        count_allocation(new_size);
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The number of allocations that this thread made in the function.
fn count_allocations(function: impl FnOnce()) -> usize {
    let before = ALLOCATION_COUNT.with(Cell::get);
    function();
    ALLOCATION_COUNT.with(Cell::get) - before
}

/// The number of bytes that this thread allocated in the function.
fn count_allocated_bytes(function: impl FnOnce()) -> usize {
    let before = ALLOCATED_BYTES.with(Cell::get);
    function();
    ALLOCATED_BYTES.with(Cell::get) - before
}

const ALL_COMPRESSIONS: [Compression; 10] = [
    Compression::Uncompressed,
    Compression::RLE,
    Compression::ZIP1,
    Compression::ZIP16,
    Compression::PIZ,
    Compression::PXR24,
    Compression::B44,
    Compression::B44A,
    Compression::DWAA(None),
    Compression::DWAB(None),
];

const ALL_BLOCKS: [Blocks; 2] = [Blocks::ScanLines, Blocks::Tiles(Vec2(24, 40))];

/// Smooth values with some grain, so that every method compresses them
/// and no block is stored raw.
fn sample(channel: usize, index: usize, width: usize) -> f32 {
    let (x, y) = (index % width, index / width);
    let grain = ((index * 2_654_435_761 + channel * 40_503) >> 7) % 16;
    0.01 * (x + 2 * y) as f32 + 0.1 * channel as f32 + 0.0005 * grain as f32
}

/// The bytes of a file with channels of every sample type. The names make
/// DWA compress the channels with each of its schemes: `R`, `G` and `B` as a
/// lossy triplet, `Y` lossy on its own, `A` with run-length encoding, and the
/// others with zlib.
fn write_image(compression: Compression, blocks: Blocks, size: Vec2<usize>) -> Vec<u8> {
    let halves = |channel: usize| {
        FlatSamples::F16(
            (0..size.area()).map(|index| f16::from_f32(sample(channel, index, size.0))).collect(),
        )
    };

    let floats = |channel: usize| {
        FlatSamples::F32((0..size.area()).map(|index| sample(channel, index, size.0)).collect())
    };

    let integers = FlatSamples::U32((0..size.area()).map(|index| (index / 5) as u32).collect());

    let channels = AnyChannels::sort(smallvec![
        AnyChannel::new("R", halves(0)),
        AnyChannel::new("G", halves(1)),
        AnyChannel::new("B", halves(2)),
        AnyChannel::new("A", halves(3)),
        AnyChannel::new("Y", floats(4)),
        AnyChannel::new("depth", floats(5)),
        AnyChannel::new("id", integers),
    ]);

    let encoding = Encoding {
        compression,
        blocks,
        line_order: LineOrder::Increasing,
    };

    let image =
        Image::from_layer(Layer::new(size, LayerAttributes::named("layer"), encoding, channels));

    let mut bytes = Vec::new();
    image.write().non_parallel().to_buffered(Cursor::new(&mut bytes)).unwrap();
    bytes
}

/// The meta data and all chunks of a file.
fn read_chunks(bytes: &[u8]) -> (MetaData, Vec<Chunk>) {
    let reader = exr::block::read(Cursor::new(bytes), false).unwrap();
    let meta_data = reader.meta_data().clone();
    let chunks = reader.all_chunks(false).unwrap().collect::<Result<Vec<Chunk>>>().unwrap();
    (meta_data, chunks)
}

fn chunk_ref(chunk: &Chunk) -> ChunkRef<'_> {
    chunk.as_flat_ref().expect("flat chunk")
}

/// The compressed pixels of a flat chunk.
fn compressed_pixels_mut(chunk: &mut Chunk) -> &mut Vec<u8> {
    match &mut chunk.compressed_block {
        CompressedBlock::ScanLine(block) => &mut block.compressed_pixels_le,
        CompressedBlock::Tile(block) => &mut block.compressed_pixels_le,
        _ => panic!("deep chunk"),
    }
}

#[test]
fn decompressing_into_a_slice_equals_decompressing_to_a_vector() {
    // one scratch for all files, as leftovers of one block must not show
    // up in the next block, whichever method either one is compressed with
    let mut scratch = DecodeScratch::default();
    let mut destination = Vec::new();

    for blocks in ALL_BLOCKS {
        for compression in ALL_COMPRESSIONS {
            let bytes = write_image(compression, blocks, Vec2(71, 277));
            let (meta_data, chunks) = read_chunks(&bytes);
            let header = &meta_data.headers[0];
            assert!(chunks.len() >= 2, "more than one block size should be tested");

            destination.resize(header.max_block_byte_size(), 0);
            let mut compressed_chunk_count = 0;

            // also decompress the chunks in reverse order,
            // so that each chunk follows a different chunk than before
            for chunk in chunks.iter().chain(chunks.iter().rev()) {
                let expected = UncompressedBlock::decompress_chunk(chunk.clone(), &meta_data, true)
                    .unwrap_or_else(|error| panic!("{} {:?}: {}", compression, blocks, error));

                // make sure that every byte is written
                destination.fill(0xa5);

                let index = UncompressedBlock::decompress_chunk_into(
                    chunk_ref(chunk),
                    &meta_data,
                    true,
                    &mut scratch,
                    &mut destination,
                )
                .unwrap_or_else(|error| panic!("{} {:?}: {}", compression, blocks, error));

                assert_eq!(index, expected.index, "{} {:?}", compression, blocks);
                assert!(
                    destination[..expected.data.len()] == expected.data[..],
                    "{} {:?}: different bytes at {:?}",
                    compression,
                    blocks,
                    index
                );
                assert!(
                    destination[expected.data.len()..].iter().all(|&byte| byte == 0xa5),
                    "{} {:?}: wrote past the block",
                    compression,
                    blocks
                );

                if chunk_ref(chunk).compressed_pixels_le.len() != expected.data.len() {
                    compressed_chunk_count += 1;
                }
            }

            if compression != Compression::Uncompressed {
                assert_ne!(compressed_chunk_count, 0, "{} {:?} is stored raw", compression, blocks);
            }
        }
    }
}

#[test]
fn a_borrowed_chunk_is_read_like_an_owned_chunk() {
    for blocks in ALL_BLOCKS {
        let bytes = write_image(Compression::ZIP16, blocks, Vec2(71, 277));
        let (meta_data, chunks) = read_chunks(&bytes);

        let mut chunk_bytes = Vec::new();
        for chunk in &chunks {
            chunk.write(&mut chunk_bytes, meta_data.headers.len()).unwrap();
        }

        let mut remaining_bytes = chunk_bytes.as_slice();
        for chunk in &chunks {
            let borrowed = ChunkRef::read(&mut remaining_bytes, &meta_data).unwrap();
            let expected = chunk_ref(chunk);

            assert_eq!(borrowed.layer_index, expected.layer_index);
            assert_eq!(borrowed.location, expected.location);
            assert_eq!(borrowed.compressed_pixels_le, expected.compressed_pixels_le);
        }

        assert!(remaining_bytes.is_empty());
    }
}

#[test]
fn decompressing_into_a_slice_does_not_allocate() {
    for blocks in ALL_BLOCKS {
        for compression in ALL_COMPRESSIONS {
            let bytes = write_image(compression, blocks, Vec2(71, 277));
            let (meta_data, chunks) = read_chunks(&bytes);

            let mut chunk_bytes = Vec::new();
            for chunk in &chunks {
                chunk.write(&mut chunk_bytes, meta_data.headers.len()).unwrap();
            }

            let mut scratch = DecodeScratch::default();
            let mut destination = vec![0_u8; meta_data.headers[0].max_block_byte_size()];

            let mut decompress_all_chunks = || {
                let mut remaining_bytes = chunk_bytes.as_slice();
                while !remaining_bytes.is_empty() {
                    let chunk = ChunkRef::read(&mut remaining_bytes, &meta_data).unwrap();
                    UncompressedBlock::decompress_chunk_into(
                        chunk,
                        &meta_data,
                        false,
                        &mut scratch,
                        &mut destination,
                    )
                    .unwrap();
                }
            };

            // the first time, the scratch grows to fit the largest block
            decompress_all_chunks();

            let allocation_count = count_allocations(decompress_all_chunks);
            assert_eq!(allocation_count, 0, "{} {:?} allocates when warm", compression, blocks);
        }
    }
}

#[test]
fn a_destination_that_is_too_small_is_an_error() {
    let mut scratch = DecodeScratch::default();

    for compression in ALL_COMPRESSIONS {
        let bytes = write_image(compression, Blocks::ScanLines, Vec2(71, 277));
        let (meta_data, chunks) = read_chunks(&bytes);
        let chunk = chunk_ref(&chunks[0]);

        let block_size = UncompressedBlock::decompress_chunk(chunks[0].clone(), &meta_data, true)
            .unwrap()
            .data
            .len();

        for size in [0, 1, block_size / 2, block_size - 1] {
            let mut destination = vec![0_u8; size];
            let result = UncompressedBlock::decompress_chunk_into(
                chunk,
                &meta_data,
                true,
                &mut scratch,
                &mut destination,
            );

            assert!(
                matches!(result, Err(Error::Invalid(_))),
                "{}: {} of {} bytes should be refused",
                compression,
                size,
                block_size
            );
        }

        let mut destination = vec![0_u8; block_size];
        UncompressedBlock::decompress_chunk_into(
            chunk,
            &meta_data,
            true,
            &mut scratch,
            &mut destination,
        )
        .unwrap();
    }
}

#[test]
fn a_chunk_size_beyond_the_bytes_is_an_error() {
    for blocks in ALL_BLOCKS {
        let bytes = write_image(Compression::ZIP16, blocks, Vec2(71, 277));
        let (meta_data, chunks) = read_chunks(&bytes);

        let mut chunk_bytes = Vec::new();
        chunks[0].write(&mut chunk_bytes, meta_data.headers.len()).unwrap();
        let pixel_size = chunk_ref(&chunks[0]).compressed_pixels_le.len();

        // the size is the last field before the pixels
        let size_position = chunk_bytes.len() - pixel_size - 4;
        let read_with_size = |size: i32, available: usize| {
            let mut damaged = chunk_bytes.clone();
            damaged[size_position..size_position + 4].copy_from_slice(&size.to_le_bytes());
            ChunkRef::read(&mut &damaged[..available], &meta_data)
                .map(|chunk| (chunk.layer_index, chunk.location, chunk.compressed_pixels_le.len()))
        };

        let (_, location, size) = read_with_size(pixel_size as i32, chunk_bytes.len()).unwrap();
        assert_eq!(size, pixel_size);
        assert!(matches!(
            (blocks, location),
            (Blocks::ScanLines, BlockLocation::ScanLine { .. })
                | (Blocks::Tiles(_), BlockLocation::Tile(_))
        ));

        for size in [pixel_size as i32 + 1, i32::MAX, -1, i32::MIN] {
            assert!(read_with_size(size, chunk_bytes.len()).is_err(), "size {}", size);
        }

        // truncated bytes, also inside of the fields before the pixels
        for available in [0, 3, size_position + 3, chunk_bytes.len() - 1] {
            assert!(read_with_size(pixel_size as i32, available).is_err(), "{} bytes", available);
        }
    }
}

/// The header of a DWA chunk states the size of each of its sections. Without
/// a limit, the decoder would make room for whatever a damaged header claims,
/// and keep that room in the scratch.
#[test]
fn damaged_dwa_section_sizes_are_an_error() {
    let mut scratch = DecodeScratch::default();

    for compression in [Compression::DWAA(None), Compression::DWAB(None)] {
        let bytes = write_image(compression, Blocks::ScanLines, Vec2(71, 277));
        let (meta_data, chunks) = read_chunks(&bytes);
        let mut destination = vec![0_u8; meta_data.headers[0].max_block_byte_size()];

        // let the scratch grow to fit the block
        UncompressedBlock::decompress_chunk_into(
            chunk_ref(&chunks[0]),
            &meta_data,
            false,
            &mut scratch,
            &mut destination,
        )
        .unwrap();

        // the chunk starts with little-endian counters, among them these
        const UNKNOWN_UNCOMPRESSED_SIZE: usize = 1;
        const RLE_UNCOMPRESSED_SIZE: usize = 6;
        const RLE_RAW_SIZE: usize = 7;
        const AC_COUNT: usize = 8;
        const DC_COUNT: usize = 9;

        for counter in
            [UNKNOWN_UNCOMPRESSED_SIZE, RLE_UNCOMPRESSED_SIZE, RLE_RAW_SIZE, AC_COUNT, DC_COUNT]
        {
            let counter_bytes = counter * 8..counter * 8 + 8;

            let mut chunk = chunks[0].clone();
            let pixels = compressed_pixels_mut(&mut chunk);
            let value = u64::from_le_bytes(pixels[counter_bytes.clone()].try_into().unwrap());
            assert_ne!(value, 0, "{}: the chunk should have data in every section", compression);

            // values so large that the allocator of this test refuses them,
            // and values that are merely larger than the chunk can hold
            for damaged_value in [1 << 62, 1 << 40, 1 << 32, value * 1000, value * 40] {
                let mut chunk = chunks[0].clone();
                compressed_pixels_mut(&mut chunk)[counter_bytes.clone()]
                    .copy_from_slice(&u64::to_le_bytes(damaged_value));

                let allocated_bytes = count_allocated_bytes(|| {
                    let result = UncompressedBlock::decompress_chunk_into(
                        chunk_ref(&chunk),
                        &meta_data,
                        false,
                        &mut scratch,
                        &mut destination,
                    );

                    // only the error message is allocated
                    assert!(
                        result.is_err(),
                        "{}: counter {} of {} should be refused",
                        compression,
                        counter,
                        damaged_value
                    );
                });

                assert!(
                    allocated_bytes < 1024,
                    "{}: counter {} of {} made the scratch grow by {} bytes",
                    compression,
                    counter,
                    damaged_value,
                    allocated_bytes
                );
            }
        }
    }
}
