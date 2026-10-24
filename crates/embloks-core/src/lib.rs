//! Embloks: a grid-based electromagnetism sandbox.
//!
//! This crate is the platform-independent core: the world being painted, the
//! physics as a double-precision reference implementation, example scenes, saves,
//! and the app's state and input handling. The GPU backend (embloks-gpu) runs the
//! same physics in compute shaders; the web front end (embloks-web) draws it.

pub mod battery;
pub mod brush;
pub mod close_range;
pub mod connectivity;
pub mod controller;
pub mod fields;
pub mod format;
pub mod grid;
pub mod layers;
pub mod materials;
pub mod maxwell;
pub mod particles;
pub mod physics;
pub mod scenes;
pub mod settings;
pub mod simulation;
pub mod storage;
pub mod world;

#[cfg(test)]
mod test_util;
