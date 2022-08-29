# MDanceIO

`mdanceio` is a cross-platform MMD(MikuMikuDance) compatible implementation. It targets at browser though WebGPU on wasm. Rewrite [nanoem](https://github.com/hkrn/nanoem) in Rust. 

This project is still in initial development phase. 

## Motivation

I built this project mainly to learn Rust and WebGPU. I hope to provide MikuMikuDance implementations in the browser, as well as via cloud rendering and on AR/VR in the future. 

`mdanceio` works as a crate that provides MMD rendering service on a specific `Surface`, `TextureView` or `Canvas`. Or it can directly return a BytesArray. 

There is another project build on `mdanceio` which provides basic MMD rendering service via WebRTC. 

## Getting Started

### How can I get supported Models and Motions

### Example

The example will play a model with specific motion in a native window. 

```bash
cargo run --package mdanceio --example winit_app -- --model <Model Path> --motion <Motion Path>
```

### WebGPU Demo

#### Prerequisite

Install [wasm-pack](https://rustwasm.github.io/wasm-pack/), a rust -> wasm workflow tool. 

We need `nodejs` to serve the demo. 

We also need [Google Chrome Canary](https://www.google.com/chrome/canary/) that supports `WebGPU`, and enable the feature flag `#enable-unsafe-webgpu`. 

#### Build wasm package

```bash
wasm-pack build mdanceio --out-dir ../target/pkg
```

> Build requires environment variable: RUSTFLAGS=--cfg=web_sys_unstable_apis

#### Run Web Demo

```bash
cd mdance-demo
npm install
npm run start
```

## Target Platform Support

## Future Plan

## Thanks
