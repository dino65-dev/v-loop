#![forbid(unsafe_code)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

fn value_error(error: vloop_core::CoreError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction]
fn api_version() -> u32 {
    vloop_core::API_VERSION
}

#[pyfunction]
fn sha256_hex(data: &[u8]) -> String {
    vloop_core::sha256_hex(data)
}

#[pyfunction]
fn is_sha256_hex(value: &str) -> bool {
    vloop_core::is_sha256_hex(value)
}

#[pyfunction]
fn ed25519_public_key(private_key: &[u8]) -> PyResult<Vec<u8>> {
    vloop_core::ed25519_public_key(private_key)
        .map(|value| value.to_vec())
        .map_err(value_error)
}

#[pyfunction]
fn ed25519_sign(private_key: &[u8], payload: &[u8]) -> PyResult<String> {
    vloop_core::ed25519_sign(private_key, payload).map_err(value_error)
}

#[pyfunction]
fn ed25519_verify(public_key: &[u8], payload: &[u8], signature: &str) -> PyResult<bool> {
    vloop_core::ed25519_verify(public_key, payload, signature).map_err(value_error)
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(api_version, module)?)?;
    module.add_function(wrap_pyfunction!(sha256_hex, module)?)?;
    module.add_function(wrap_pyfunction!(is_sha256_hex, module)?)?;
    module.add_function(wrap_pyfunction!(ed25519_public_key, module)?)?;
    module.add_function(wrap_pyfunction!(ed25519_sign, module)?)?;
    module.add_function(wrap_pyfunction!(ed25519_verify, module)?)?;
    Ok(())
}
