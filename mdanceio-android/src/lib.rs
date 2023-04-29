#![allow(unknown_lints)]

pub mod android_proxy;

use android_proxy::AndroidProxy;
use android_proxy::MdanceioAndroidError;

uniffi_macros::include_scaffolding!("mdanceio");
