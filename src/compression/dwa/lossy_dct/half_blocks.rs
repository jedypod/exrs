// Whole-block conversions between half and f32. The `half` crate's slice
// conversions use F16C too, but make one out-of-line call per eight values,
// which is most of their cost here. Inside a multiversioned function the
// instructions inline, so a block costs one call. The results are the same
// as `half`'s, bit for bit.

use half::{
    f16,
    slice::{HalfBitsSliceExt, HalfFloatSliceExt},
};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
fn widen_with_f16c(halves: &[u16; 64], floats: &mut [f32; 64]) {
    use std::arch::x86_64::*;

    for (halves, floats) in halves.chunks_exact(8).zip(floats.chunks_exact_mut(8)) {
        let word = |halves: &[u16]| {
            let [a, b, c, d] = [0, 1, 2, 3].map(|index| u64::from(halves[index]));
            (a | b << 16 | c << 32 | d << 48) as i64
        };
        let input = _mm_set_epi64x(word(&halves[4..]), word(&halves[..4]));
        let wide = _mm256_castps_si256(_mm256_cvtph_ps(input));

        let words = [
            _mm256_extract_epi64::<0>(wide),
            _mm256_extract_epi64::<1>(wide),
            _mm256_extract_epi64::<2>(wide),
            _mm256_extract_epi64::<3>(wide),
        ];
        for (pair, word) in floats.chunks_exact_mut(2).zip(words) {
            pair[0] = f32::from_bits(word as u32);
            pair[1] = f32::from_bits((word >> 32) as u32);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
fn narrow_with_f16c(floats: &[f32; 64], halves: &mut [f16; 64]) {
    use std::arch::x86_64::*;

    for (floats, halves) in floats.chunks_exact(8).zip(halves.chunks_exact_mut(8)) {
        let input = _mm256_setr_ps(
            floats[0], floats[1], floats[2], floats[3], floats[4], floats[5], floats[6], floats[7],
        );
        let narrow = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(input);

        let words = [_mm_cvtsi128_si64(narrow), _mm_extract_epi64::<1>(narrow)];
        for (four, word) in halves.chunks_exact_mut(4).zip(words) {
            for (index, half) in four.iter_mut().enumerate() {
                *half = f16::from_bits((word >> (index * 16)) as u16);
            }
        }
    }
}

/// Widen a block of half bits.
#[cfg_attr(target_arch = "x86_64", multiversion::multiversion(targets("x86_64+avx2+f16c")))]
pub(super) fn halves_to_f32(halves: &[u16; 64], floats: &mut [f32; 64]) {
    #[cfg(target_arch = "x86_64")]
    multiversion::target::match_target! {
        "x86_64+avx2+f16c" => widen_with_f16c(halves, floats),
        _ => halves.reinterpret_cast::<f16>().convert_to_f32_slice(floats),
    }

    #[cfg(not(target_arch = "x86_64"))]
    halves.reinterpret_cast::<f16>().convert_to_f32_slice(floats);
}

/// Narrow a block to half.
#[cfg_attr(target_arch = "x86_64", multiversion::multiversion(targets("x86_64+avx2+f16c")))]
pub(super) fn f32_to_halves(floats: &[f32; 64], halves: &mut [f16; 64]) {
    #[cfg(target_arch = "x86_64")]
    multiversion::target::match_target! {
        "x86_64+avx2+f16c" => narrow_with_f16c(floats, halves),
        _ => halves.convert_from_f32_slice(floats),
    }

    #[cfg(not(target_arch = "x86_64"))]
    halves.convert_from_f32_slice(floats);
}

#[cfg(test)]
mod test {
    use super::*;

    /// `half` converts each value separately here, through its own code path.
    fn assert_narrows_like_half(float_bits: impl IntoIterator<Item = u32>) {
        let mut floats = [0.0; 64];
        let mut count = 0;
        let mut check = |floats: &[f32; 64]| {
            let mut halves = [f16::ZERO; 64];
            f32_to_halves(floats, &mut halves);
            for (float, half) in floats.iter().zip(halves) {
                let expected = f16::from_f32(*float).to_bits();
                assert_eq!(half.to_bits(), expected, "{:#010x}", float.to_bits());
            }
        };

        for bits in float_bits {
            floats[count] = f32::from_bits(bits);
            count += 1;
            if count == 64 {
                check(&floats);
                count = 0;
            }
        }
        check(&floats);
    }

    #[test]
    fn every_half_widens_like_half() {
        for start in (0..=u16::MAX as usize).step_by(64) {
            let halves: [u16; 64] = std::array::from_fn(|index| (start + index) as u16);
            let mut floats = [0.0; 64];
            halves_to_f32(&halves, &mut floats);

            for (half, float) in halves.iter().zip(floats) {
                let expected = f16::from_bits(*half).to_f32().to_bits();
                assert_eq!(float.to_bits(), expected, "{half:#06x}");
            }
        }
    }

    #[test]
    fn floats_around_each_half_narrow_like_half() {
        // the floats nearest each half, and each rounding midpoint
        assert_narrows_like_half((0..=u16::MAX).flat_map(|half| {
            let float = f16::from_bits(half).to_f32().to_bits();
            const OFFSETS: [u32; 7] = [0, 1, 2, 0xfff, 0x1000, 0x1001, 0x1fff];
            OFFSETS
                .iter()
                .flat_map(move |&offset| [float.wrapping_add(offset), float.wrapping_sub(offset)])
        }));
    }

    /// Takes a while without optimization, so it only runs on request:
    /// `cargo test --release -- --ignored every_float`
    #[test]
    #[ignore]
    fn every_float_narrows_like_half() {
        assert_narrows_like_half(0..=u32::MAX);
    }
}
