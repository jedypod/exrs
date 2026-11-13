//! Damaged files must not make the reader reserve huge amounts of memory up
//! front. A separate test binary, because it installs a global allocator that
//! refuses any single allocation over 1 GiB, as a memory-constrained system
//! would.

use std::alloc::{GlobalAlloc, Layout, System};

use exr::{image::pixel_vec::PixelVec, prelude::*};

struct Capped;

const MAX_ALLOCATION: usize = 1 << 30;

unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > MAX_ALLOCATION {
            return std::ptr::null_mut();
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Capped = Capped;

/// The DWA header claims 2^35 AC values; the decoder used to reserve room for
/// all of them.
#[test]
fn dwa_huge_ac_count_is_an_error() {
    let result = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .all_layers()
        .all_attributes()
        .non_parallel()
        .from_file("tests/images/fuzzed/dwa_huffman_huge_ac_count.exr");

    assert!(result.is_err());
}

/// A DWA chunk header claims a zlib section of hundreds of terabytes; the
/// decoder used to pass that size to the inflater, which reserved it up front.
#[test]
fn dwa_huge_zlib_section_is_an_error() {
    let result = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .all_layers()
        .all_attributes()
        .non_parallel()
        .from_file("tests/images/fuzzed/dwa_zlib_huge_section_size.exr");

    assert!(result.is_err());
}

/// Headers claiming 654 million lines, where the file ends before the offset
/// table does. Only the file's own offsets tell it apart from a real image.
#[test]
fn huge_height_is_an_error() {
    assert_flat_and_rgba_reads_fail("tests/images/fuzzed/huge_height.exr");
}

/// A single 2^29 pixel line, which has a valid one-entry offset table: 8 GiB
/// of RGBA samples.
#[test]
fn huge_width_is_an_error() {
    assert_flat_and_rgba_reads_fail("tests/images/fuzzed/huge_width.exr");
}

/// The image would fit into memory, so only the missing offset table can
/// refuse it, and that must happen before the user's storage is created.
#[test]
fn truncated_offset_table_is_found_before_allocating() {
    let result = read_first_rgba_layer_from_file(
        "tests/images/fuzzed/truncated_offset_table.exr",
        |_, _| -> PixelVec<(f32, f32, f32, f32)> { panic!("allocated for a truncated file") },
        PixelVec::set_pixel,
    );
    assert!(result.is_err());
}

fn assert_flat_and_rgba_reads_fail(path: &str) {
    let flat = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .all_layers()
        .all_attributes()
        .non_parallel()
        .from_file(path);
    assert!(flat.is_err());

    let rgba = read_first_rgba_layer_from_file(
        path,
        PixelVec::<(f32, f32, f32, f32)>::constructor,
        PixelVec::set_pixel,
    );
    assert!(rgba.is_err());
}

/// The size of a block follows from the data window in the header. A damaged
/// data window can claim blocks of any size, which must not be allocated
/// before the compressed data is found to be far too short to fill them.
#[test]
fn huge_block_is_an_error_before_allocating() {
    use std::{convert::TryInto, io::Cursor};

    use exr::block::{chunk::Chunk, UncompressedBlock};

    let compressions = [
        Compression::RLE,
        Compression::ZIP1,
        Compression::ZIP16,
        Compression::PIZ,
        Compression::PXR24,
        Compression::B44,
        Compression::DWAA(None),
    ];

    for compression in compressions {
        let size = Vec2(64, 40);
        let image = Image::from_layer(Layer::new(
            size,
            LayerAttributes::named("wide"),
            Encoding {
                compression,
                blocks: Blocks::ScanLines,
                line_order: LineOrder::Increasing,
            },
            SpecificChannels::rgb(|position: Vec2<usize>| {
                (position.x() as f32, position.y() as f32, 0.5_f32)
            }),
        ));

        let mut bytes = Vec::new();
        image.write().non_parallel().to_buffered(Cursor::new(&mut bytes)).unwrap();

        // the attribute is its name, its type, the value size, and the
        // value: the minimum x and y, and then the maximum x and y
        let attribute_start = b"dataWindow\0box2i\0\x10\0\0\0";
        let value_position = bytes
            .windows(attribute_start.len())
            .position(|window| window == attribute_start)
            .expect("the writer should store the data window")
            + attribute_start.len();

        let max_x = &mut bytes[value_position + 8..value_position + 12];
        assert_eq!(i32::from_le_bytes((&*max_x).try_into().unwrap()), 63);
        max_x.copy_from_slice(&400_000_000_i32.to_le_bytes());

        // a block now claims 400 million pixels in each scan line,
        // far beyond what the allocator of this test hands out
        let reader = exr::block::read(Cursor::new(&bytes), false).unwrap();
        let meta_data = reader.meta_data().clone();
        assert!(meta_data.headers[0].max_block_byte_size() > MAX_ALLOCATION);

        let chunks: Vec<Chunk> = reader.all_chunks(false).unwrap().collect::<Result<_>>().unwrap();

        for chunk in chunks {
            let result = UncompressedBlock::decompress_chunk(chunk, &meta_data, false);
            assert!(result.is_err(), "{} should refuse a huge block", compression);
        }
    }
}
