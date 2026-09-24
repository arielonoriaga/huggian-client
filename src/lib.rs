//! Shared primitives for integrating with huggian-core.
//!
//! v1 covered the webhook receiver layer only — the one piece that was
//! byte-for-byte duplicated across ez-catalog and ez-stock. v2 adds a
//! read-only product-registry lookup, behind its own Cargo feature so
//! `webhook`-only consumers (ez-commerce, today) don't inherit an HTTP
//! client + async runtime they never asked for. Product-specific
//! entitlement/provisioning logic deliberately stays in each consuming
//! service — this crate resolves "which product is this slug", nothing more.
//!
//! v0.4 adds `billing`: the per-product *subscription flow* (trial →
//! checkout → cancel, plus the pure subscription-state rules and a per-company
//! cache). That flow is identical for every product that sells a plan through
//! Huggian, so it lives here once; what stays product-specific is the plan
//! slug, the trial length and how a request is tied to a company.

#[cfg(feature = "webhook")]
pub mod webhook;

#[cfg(feature = "products")]
pub mod products;

#[cfg(feature = "ai-gateway")]
pub mod ai_gateway;

#[cfg(feature = "billing")]
pub mod billing;

#[cfg(feature = "billing-axum")]
pub mod billing_axum;
