//! Shared primitives for integrating with huggian-core.
//!
//! v1 covered the webhook receiver layer only — the one piece that was
//! byte-for-byte duplicated across ez-catalog and ez-stock. v2 adds a
//! read-only product-registry lookup, behind its own Cargo feature so
//! `webhook`-only consumers (ez-commerce, today) don't inherit an HTTP
//! client + async runtime they never asked for. Product-specific
//! entitlement/provisioning logic deliberately stays in each consuming
//! service — this crate resolves "which product is this slug", nothing more.

#[cfg(feature = "webhook")]
pub mod webhook;

#[cfg(feature = "products")]
pub mod products;

#[cfg(feature = "ai-gateway")]
pub mod ai_gateway;
