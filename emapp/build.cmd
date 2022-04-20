set RUSTFLAGS=--cfg=web_sys_unstable_apis
cargo build --no-default-features --target wasm32-unknown-unknown --lib
set RUSTFLAGS=
wasm-bindgen --out-dir target/generated --web target/wasm32-unknown-unknown/debug/nanoemweb_emapp.wasm