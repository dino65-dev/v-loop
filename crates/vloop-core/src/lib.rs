#![forbid(unsafe_code)]
//! Small deterministic primitives used at V-Loop security boundaries.
//!
//! Python owns object normalization and public dataclasses.  This crate owns
//! byte-level hashing and Ed25519 operations over those already canonical
//! bytes, so a native implementation cannot silently change a signed wire
//! format.

use base64::{engine::general_purpose::URL_SAFE, Engine as _};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};
use thiserror::Error;
use zeroize::Zeroizing;

pub const API_VERSION: u32 = 1;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CoreError {
    #[error("Ed25519 private keys must contain exactly 32 bytes")]
    InvalidPrivateKey,
    #[error("Ed25519 public keys must contain exactly 32 bytes")]
    InvalidPublicKey,
    #[error("Ed25519 signature is malformed")]
    InvalidSignatureEncoding,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

pub fn ed25519_public_key(private_key: &[u8]) -> Result<[u8; 32], CoreError> {
    let key: [u8; 32] = private_key
        .try_into()
        .map_err(|_| CoreError::InvalidPrivateKey)?;
    let key = Zeroizing::new(key);
    Ok(SigningKey::from_bytes(&key).verifying_key().to_bytes())
}

pub fn ed25519_sign(private_key: &[u8], payload: &[u8]) -> Result<String, CoreError> {
    let key: [u8; 32] = private_key
        .try_into()
        .map_err(|_| CoreError::InvalidPrivateKey)?;
    let key = Zeroizing::new(key);
    let signature = SigningKey::from_bytes(&key).sign(payload);
    Ok(URL_SAFE.encode(signature.to_bytes()))
}

pub fn ed25519_verify(
    public_key: &[u8],
    payload: &[u8],
    signature: &str,
) -> Result<bool, CoreError> {
    let key: [u8; 32] = public_key
        .try_into()
        .map_err(|_| CoreError::InvalidPublicKey)?;
    let verifying_key = VerifyingKey::from_bytes(&key).map_err(|_| CoreError::InvalidPublicKey)?;
    let signature = URL_SAFE
        .decode(signature)
        .map_err(|_| CoreError::InvalidSignatureEncoding)?;
    let signature =
        Signature::from_slice(&signature).map_err(|_| CoreError::InvalidSignatureEncoding)?;
    Ok(verifying_key.verify(payload, &signature).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_exact_bytes() {
        assert_eq!(
            sha256_hex(br#"{"a":1,"b":2}"#),
            "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
        );
    }

    #[test]
    fn signs_and_verifies_urlsafe_base64() {
        let private_key = [7_u8; 32];
        let public_key = ed25519_public_key(&private_key).unwrap();
        let signature = ed25519_sign(&private_key, b"vloop capability payload").unwrap();
        assert!(ed25519_verify(&public_key, b"vloop capability payload", &signature).unwrap());
        assert!(!ed25519_verify(&public_key, b"another payload", &signature).unwrap());
    }
}
