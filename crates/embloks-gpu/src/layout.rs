//! Uniform structs shared by Rust and WGSL, generated from one field list so the
//! two sides can't disagree about the layout. All fields are 4-byte scalars.

/// Declares a `#[repr(C)]` struct of u32/f32 fields plus `WGSL`, its shader-side
/// declaration. The struct is padded to a multiple of 16 bytes, as uniforms need.
macro_rules! gpu_struct {
    ($name:ident { $($field:ident : $ty:ident),* $(,)? }) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name {
            $(pub $field: $ty,)*
            pub _pad: [u32; (4 - ([$(stringify!($field)),*].len() % 4)) % 4],
        }

        // Sound: every field is a 4-byte scalar and the padding is explicit, so the
        // struct has no implicit padding and any bit pattern is valid.
        unsafe impl bytemuck::Zeroable for $name {}
        unsafe impl bytemuck::Pod for $name {}

        impl $name {
            pub fn wgsl() -> String {
                let mut out = format!("struct {} {{\n", stringify!($name));
                $(out += &format!("    {}: {},\n", stringify!($field), stringify!($ty));)*
                let pad = (4 - ([$(stringify!($field)),*].len() % 4)) % 4;
                for i in 0..pad {
                    out += &format!("    _pad{i}: u32,\n");
                }
                out + "}\n"
            }
        }
    };
}

pub(crate) use gpu_struct;
