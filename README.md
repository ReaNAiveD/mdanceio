# MDanceIO

[![Build Status](https://github.com/ReaNAiveD/mdanceio/workflows/CI/badge.svg)](https://github.com/ReaNAiveD/mdanceio/actions)
[![NPM](https://img.shields.io/npm/v/mdanceio.svg)](https://www.npmjs.com/package/mdanceio)
[![NPM](https://img.shields.io/npm/v/@webgl-supports/mdanceio)](https://www.npmjs.com/package/@webgl-supports/mdanceio)
[![Crate](https://img.shields.io/crates/v/mdanceio.svg)](https://crates.io/crates/mdanceio)
[![Maven](https://img.shields.io/maven-central/v/cn.svecri/mdanceio-ar)](https://repo1.maven.org/maven2/cn/svecri/mdanceio-ar/)

`mdanceio` is a cross-platform MMD(MikuMikuDance) compatible implementation. It targets at browser though WebGPU on wasm. Rewrite [nanoem](https://github.com/hkrn/nanoem) in Rust. 

This project is still in initial development phase. 

## Motivation

I built this project mainly to learn Rust and WebGPU. I hope to provide a MikuMikuDance implementation in the browser, as well as via cloud rendering and on AR/VR in the future. 

`mdanceio` works as a crate that provides MMD rendering service on a specific `Surface`, `TextureView` or `Canvas`. Or it can directly return a BytesArray. 

There is another [project](https://github.com/ReaNAiveD/mdrs) build on `mdanceio` which provides basic MMD remote rendering service via WebRTC. 

## Getting Started

### How can I get supported Models and Motions

You can fetch models and motions from [模之屋(PlayBox)](https://www.aplaybox.com/), a community sharing character models. 

The project build is likely buggy and unfinished. You can try the following model and motion which is tested and welling working to get started. 

- Model: [【原神】砂糖Sucrose](https://www.aplaybox.com/details/model/LXbOVepFhfRw)
    - ◆模型提供：miHoYo
    - ◆模型改造：观海
- Motion: [神里凌华传说任务舞蹈](https://www.aplaybox.com/details/motion/EkgMGiVYgOuZ)
    - 动作：Lct火红枣

The demo GIFs in this README use the above model and motion. 

### Example

The example will play a model with specific motion in a native window. 

```bash
cargo run --package mdanceio --example winit_app -- --model <Model Path> --motion <Motion Path>
```

You can build as an executable as well. 

```bash
cargo build --package mdanceio --example winit_app --release
```

You can also fetch the executable in Actions. 

### WebGPU Demo

You can visit the [demo here](https://reanaived.github.io/mdanceio?webgpu=true)(requires Chrome Canary with `#enable-unsafe-webgpu` enabled). 

### WebGL Demo

You can visit the [demo using WebGL](https://reanaived.github.io/mdanceio). 

#### Prerequisite

Install [wasm-pack](https://rustwasm.github.io/wasm-pack/), a rust -> wasm workflow tool. 

You need `nodejs` to serve the demo. 

You also need [Google Chrome Canary](https://www.google.com/chrome/canary/) that supports `WebGPU`, and enable the feature flag `#enable-unsafe-webgpu`. 

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

You can also fetch prebuilt web bundle in Actions. 

### Remote Rendering

I have a demo project about how to use mdanceio as a rendering service [here](https://github.com/ReaNAiveD/mdrs). 

The service uses WebRTC to communicate with the browser. 

You can follow its guidance to play with it. 

## Target Platform Support

| Platform | Support |
| ------ | :----: |
| Windows | ✅ |
| Linux | 🆗 |
| MacOS | 🆗 |
| Browser(WebGPU) | ✅ |
| Browser(WebGL) | 🆗 |
| Android | 🛠️ |
| OpenXR | 🛠️ |

> ✅ = First Class Support
> 🆗 = Best Effort Support
> 🛠️ = Unsupported, but planned

## File Format Support

### Model Format

| Format | Support |
| --- | :---: |
| PMX | ✅ |
| PMD | ❌ |

### Motion Format

| Format | Support |
| --- | :---: |
| VMD | ✅ |
| NMD | ❌ |

## Future Plan

- The core functionality has not yet completed. We will cover all MikuMikuDance features in the future. 
- I'm interested in supporting `mdanceio` in an AR/VR environment. We will extract SDK for AR usage and provide a demo. 
- Provide an architecture that supports cloud rendering will. 
- Provide support for MME or similar technologies. 
