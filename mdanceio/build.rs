#[cfg(target_os = "android")]
fn main() {
    uniffi_build::generate_scaffolding("./src/math.udl").unwrap();
}

#[cfg(not(target_os = "android"))]
fn main() {

}