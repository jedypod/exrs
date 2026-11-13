//! Decompressing a chunk into memory that the caller provides must produce
//! the bytes that `UncompressedBlock::decompress_chunk` returns, and must
//! refuse a destination that is too small and a chunk that is damaged.

use std::io::Cursor;

use exr::{
    block::{
        chunk::{BlockLocation, Chunk, ChunkRef},
        UncompressedBlock,
    },
    compression::DecodeScratch,
    meta::MetaData,
    prelude::*,
};
use smallvec::smallvec;

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
