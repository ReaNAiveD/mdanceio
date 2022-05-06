[CmdletBinding()]
param (
    [Alias("B")]
    [Parameter(Mandatory=$false)]
    [switch]
    $Browser
)

$Env:RUSTFLAGS = '--cfg=web_sys_unstable_apis'
cargo build --no-default-features --target wasm32-unknown-unknown --lib
$Env:RUSTFLAGS = ''
wasm-bindgen --out-dir target/generated --target web target/wasm32-unknown-unknown/debug/nanoemweb_emapp.wasm
Copy-Item target/generated/* -Destination target\pkg
if ($Browser -eq $true)
{
    $Env:WGPU_TRACE = 'target/wgpu-trace'
    Start-Process "C:\Program Files\Firefox Nightly\firefox.exe" "http://localhost:3000"
    $Env:WGPU_TRACE = ''
}