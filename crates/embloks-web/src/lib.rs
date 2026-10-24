//! Embloks in the browser. Builds with Trunk (see index.html); needs WebGPU.
//!
//! Only `dom` (the panels' HTML and key mapping) builds natively, for its tests;
//! the rest is browser-only.

// Used by the browser-only modules; natively only its tests run.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
mod dom;

#[cfg(target_arch = "wasm32")]
mod app;
#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(target_arch = "wasm32")]
mod saves;
