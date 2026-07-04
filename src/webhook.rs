//! huggian webhook receiver primitives.
//!
//! Every huggian-core webhook is signed with HMAC-SHA256 over
//! `"{timestamp}.{body}"` and carries the unix timestamp in `X-BBX-Timestamp`.
//! Receivers verify the signature against the raw body (before JSON parsing)
//! and reject deliveries outside a replay window. This module is the single
//! source of that logic; it was previously copy-pasted, identically, into
//! ez-catalog and ez-stock.

use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Default replay window (seconds) huggian deliveries are validated against.
pub const REPLAY_WINDOW_SECS: i64 = 300;

/// Verify an `X-BBX-Signature` header value.
///
/// `signature_hex` is the hex-encoded HMAC-SHA256 of
/// `timestamp.as_bytes()` + `b"."` + `body`. The compare is constant-time.
/// Returns `false` (never panics/errors) on malformed hex or an invalid key
/// length, so callers can treat every failure mode as "reject".
pub fn verify_signature(secret: &[u8], timestamp: &str, body: &[u8], signature_hex: &str) -> bool {
    let Ok(signature) = hex::decode(signature_hex) else {
        return false;
    };
    let Ok(mut mac) = HmacSha256::new_from_slice(secret) else {
        return false;
    };
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(body);
    mac.verify_slice(&signature).is_ok()
}

/// True when the delivery's `event_secs` is within `window_secs` of `now_secs`
/// (absolute difference). Guards against replayed or clock-skewed deliveries.
pub fn within_replay_window(now_secs: i64, event_secs: i64, window_secs: i64) -> bool {
    (now_secs - event_secs).abs() <= window_secs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sign(secret: &[u8], ts: &str, body: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(secret).unwrap();
        mac.update(ts.as_bytes());
        mac.update(b".");
        mac.update(body);
        hex::encode(mac.finalize().into_bytes())
    }

    #[test]
    fn signature_round_trip() {
        let secret = b"shh";
        let ts = "1714600000";
        let body = b"{\"hello\":\"world\"}";
        let sig = sign(secret, ts, body);
        assert!(verify_signature(secret, ts, body, &sig));
        assert!(!verify_signature(secret, ts, body, &"00".repeat(32)));
        assert!(!verify_signature(b"wrong", ts, body, &sig));
    }

    #[test]
    fn empty_signature_returns_false() {
        assert!(!verify_signature(b"shh", "1714600000", b"body", ""));
    }

    #[test]
    fn odd_length_hex_signature_returns_false() {
        assert!(!verify_signature(b"shh", "1714600000", b"body", "abc"));
    }

    #[test]
    fn empty_secret_returns_false() {
        let ts = "1714600000";
        let body = b"body";
        let sig = sign(b"shh", ts, body);
        assert!(!verify_signature(b"", ts, body, &sig));
    }

    #[test]
    fn replay_window_bounds() {
        assert!(within_replay_window(1000, 1000, 300));
        assert!(within_replay_window(1000, 1300, 300));
        assert!(within_replay_window(1000, 700, 300));
        assert!(!within_replay_window(1000, 1301, 300));
        assert!(!within_replay_window(1000, 699, 300));
    }
}
