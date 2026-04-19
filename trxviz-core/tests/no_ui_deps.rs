/// Linkage test: `trxviz-core` must not pull egui into its dependency tree.
///
/// This test does nothing at runtime. Its purpose is to give CI a hook:
/// running `cargo tree -p trxviz-core | grep -i egui` after a successful
/// `cargo test -p trxviz-core` (which compiles this file) ensures no egui
/// symbol is reachable from the core crate.
///
/// The authoritative check lives in CI (`cargo-machete` + `cargo tree`
/// assertions). This file is a belt-and-suspenders signal that the crate
/// at least *compiles* without GUI context.
use trxviz_core as _;

#[test]
fn core_crate_compiles_without_gui_context() {
    // Intentionally empty — compilation is the test.
}
