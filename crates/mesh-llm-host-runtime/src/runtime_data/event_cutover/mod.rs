//! Field-level legacy/reducer cutover fencing for `RuntimeDataCollector`.
//!
//! See `matrix.rs` for the compile-time field → generation → producer-task
//! authority table, `merge.rs` for the generation-tagged merge gate wired
//! into `collector::update_snapshots`, and `producer_write_methods.rs` for
//! the plan's eleven-write-method documentation matrix.

mod matrix;
mod merge;
#[cfg(test)]
mod producer_write_methods;

#[cfg(test)]
mod tests;

#[cfg(not(test))]
pub(crate) use matrix::FieldId;
pub(crate) use matrix::field_of_dirty;
#[cfg(test)]
pub(crate) use matrix::{CUTOVER_MATRIX, FieldId, Generation, dirty_scope_name};
#[cfg(test)]
pub(crate) use merge::{MergeOutcome, apply_legacy_write_for_generation};
pub(crate) use merge::{ShadowHealth, merge_legacy_publish, should_apply_legacy_write};
#[cfg(test)]
pub(crate) use producer_write_methods::{
    PRODUCER_WRITE_METHODS, extract_non_test_write_method_names,
};
