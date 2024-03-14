#![allow(dead_code)]

use core::{
    arch::{aarch64::uint16x8_t, asm},
    mem::MaybeUninit,
    ptr,
};

use half::f16;

#[target_feature(enable = "fp16")]
#[inline]
unsafe fn fadd_f16x8(a: &uint16x8_t, b: &uint16x8_t) -> uint16x8_t {
    let result: uint16x8_t;
    asm!(
        "fadd {result:v}.8h, {a:v}.8h, {b:v}.8h",
        a = in(vreg) *a,
        b = in(vreg) *b,
        result = out(vreg) result,
        options(pure, nomem, nostack));
    result
}

#[target_feature(enable = "fp16")]
#[inline]
unsafe fn fadd_assign_f16x8(a: &mut uint16x8_t, b: &uint16x8_t) {
    asm!(
        "fadd {a:v}.8h, {a:v}.8h, {b:v}.8h",
        a = inlateout(vreg) *a,
        b = in(vreg) *b,
        options(pure, nomem, nostack));
}

#[target_feature(enable = "fp16")]
// SAFETY: a and b must be aligned to 128 bits
unsafe fn add_f16x8_aligned(a: &[f16; 8], b: &[f16; 8]) -> [f16; 8] {
    let a = a.as_ptr() as *const uint16x8_t;
    let b = b.as_ptr() as *const uint16x8_t;
    let result = unsafe { fadd_f16x8(&*a, &*b) };
    *(&result as *const uint16x8_t).cast()
}

// SAFETY: a and b must be aligned to 128 bits
pub(super) unsafe fn add_f16_aligned_slice(a: &[f16], b: &[f16], result: &mut [f16]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());
    assert_eq!(a.len() % 8, 0);
    for i in (0..a.len()).step_by(8) {
        let a = unsafe { &*(a.as_ptr().add(i) as *const [f16; 8]) };
        let b = unsafe { &*(b.as_ptr().add(i) as *const [f16; 8]) };
        let result = unsafe { &mut *(result.as_mut_ptr().add(i) as *mut [f16; 8]) };
        *result = unsafe { add_f16x8_aligned(a, b) };
    }
}

#[target_feature(enable = "fp16")]
// SAFETY: a and b must be aligned to 128 bits
unsafe fn add_assign_f16x8_aligned(a: &mut [f16; 8], b: &[f16; 8]) {
    let a = a.as_mut_ptr() as *mut uint16x8_t;
    let b = b.as_ptr() as *const uint16x8_t;
    unsafe { fadd_assign_f16x8(&mut *a, &*b) };
}

// SAFETY: a and b must be aligned to 128 bits
pub(super) unsafe fn add_assign_f16_aligned_slice(a: &mut [f16], b: &[f16]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 8, 0);
    for i in (0..a.len()).step_by(8) {
        let a = unsafe { &mut *(a.as_mut_ptr().add(i) as *mut [f16; 8]) };
        let b = unsafe { &*(b.as_ptr().add(i) as *const [f16; 8]) };
        unsafe { add_assign_f16x8_aligned(a, b) };
    }
}

#[target_feature(enable = "fp16")]
unsafe fn add_f16x8(a: &[f16; 8], b: &[f16; 8]) -> [f16; 8] {
    let mut aa = MaybeUninit::<uint16x8_t>::uninit();
    ptr::copy_nonoverlapping(a.as_ptr(), aa.as_mut_ptr().cast(), 8);
    let mut bb = MaybeUninit::<uint16x8_t>::uninit();
    ptr::copy_nonoverlapping(b.as_ptr(), bb.as_mut_ptr().cast(), 8);
    let result = unsafe { fadd_f16x8(&aa.assume_init(), &bb.assume_init()) };
    *(&result as *const uint16x8_t).cast()
}

#[target_feature(enable = "fp16")]
unsafe fn add_assign_f16x8(a: &mut [f16; 8], b: &[f16; 8]) {
    let mut aa = MaybeUninit::<uint16x8_t>::uninit();
    ptr::copy_nonoverlapping(a.as_ptr(), aa.as_mut_ptr().cast(), 8);
    let mut aa = aa.assume_init();
    let mut bb = MaybeUninit::<uint16x8_t>::uninit();
    ptr::copy_nonoverlapping(b.as_ptr(), bb.as_mut_ptr().cast(), 8);
    let bb = bb.assume_init();
    asm!(
        "fadd {aa:v}.8h, {aa:v}.8h, {bb:v}.8h",
        aa = inlateout(vreg) aa,
        bb = in(vreg) bb,
        options(pure, nomem, nostack));
    ptr::copy_nonoverlapping(&aa as *const uint16x8_t as *const f16, a.as_mut_ptr(), 8);
}

// test
#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::alignment;
    use aligned_box::AlignedBox;

    #[test]
    fn test_add_f16x8() {
        // create two arrays of f16
        let a: [f16; 8] = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let b: [f16; 8] = [
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
            f16::from_f32(8.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
            f16::from_f32(8.0),
        ];
        // call the function
        let result = unsafe { add_f16x8(&a, &b) };

        assert_eq!(
            result,
            [
                f16::from_f32(6.0),
                f16::from_f32(8.0),
                f16::from_f32(10.0),
                f16::from_f32(12.0),
                f16::from_f32(6.0),
                f16::from_f32(8.0),
                f16::from_f32(10.0),
                f16::from_f32(12.0)
            ]
        );
    }

    #[test]
    fn test_add_f16x8_aligned() {
        let a =
            AlignedBox::<[f16]>::slice_from_value(alignment(128), 128, f16::from_f32(1.0)).unwrap();
        let b =
            AlignedBox::<[f16]>::slice_from_value(alignment(128), 128, f16::from_f32(2.0)).unwrap();

        // forcibly convert &a as &[f16; 8]
        let a = unsafe { &*(a.as_ptr() as *const [f16; 8]) };
        let b = unsafe { &*(b.as_ptr() as *const [f16; 8]) };

        let result = unsafe { add_f16x8_aligned(&a, &b) };

        assert_eq!(
            result,
            [
                f16::from_f32(3.0),
                f16::from_f32(3.0),
                f16::from_f32(3.0),
                f16::from_f32(3.0),
                f16::from_f32(3.0),
                f16::from_f32(3.0),
                f16::from_f32(3.0),
                f16::from_f32(3.0),
            ]
        );
    }

    #[test]
    fn test_add_f16_aligned_slice() {
        let a =
            AlignedBox::<[f16]>::slice_from_value(alignment(128), 128, f16::from_f32(1.0)).unwrap();
        let b =
            AlignedBox::<[f16]>::slice_from_value(alignment(128), 128, f16::from_f32(2.0)).unwrap();
        let mut result = AlignedBox::<[f16]>::slice_from_default(alignment(128), 128).unwrap();

        unsafe { add_f16_aligned_slice(&a, &b, &mut result) };

        // check all elements are 3.0
        for i in 0..result.len() {
            assert_eq!(result[i], f16::from_f32(3.0));
        }
    }

    #[test]
    fn test_add_assign_f16x8() {
        // create two arrays of f16
        let mut a: [f16; 8] = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let b: [f16; 8] = [
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
            f16::from_f32(8.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
            f16::from_f32(8.0),
        ];
        // call the function
        unsafe { add_assign_f16x8(&mut a, &b) };

        assert_eq!(
            a,
            [
                f16::from_f32(6.0),
                f16::from_f32(8.0),
                f16::from_f32(10.0),
                f16::from_f32(12.0),
                f16::from_f32(6.0),
                f16::from_f32(8.0),
                f16::from_f32(10.0),
                f16::from_f32(12.0)
            ]
        );
    }
}
