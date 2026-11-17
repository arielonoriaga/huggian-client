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
//!
//! v0.5 adds `HuggianAiClient::extract_document` (the gateway's
//! `/api/ai/extract-document`) and the extraction route's error codes.
//! Breaking: `AiErrorCode` and `AiGatewayError` gained variants, so
//! exhaustive matches in consumers must add arms.
//!
//! Build a dedicated `HuggianAiClient` for extraction (own token, breaker and
//! bulkhead): sharing one with chat lets long extractions hold chat's slots.
//! `Extraction::cost_usd` is the gateway's estimate for display/alerting
//! only; never feed it into a money ledger.
//!
//! v0.5.1 hardens extraction: the request timeout derives from the gateway's
//! (`timeout_ms` + 10 s), client-side timeouts are content outcomes
//! (`ExtractionTimeout`, never counted by the breaker), requests are validated
//! locally (new `AiGatewayError::InvalidRequest` — v0.5.0 had no adopters, so
//! the added variant ships in a patch release), and `Extraction`'s `Debug`
//! redacts `data`.
//!
//! v0.5.2 adds `billing::Plan::tier` (core's currency-independent tier, e.g.
//! `profesional` for both `profesional` and `profesional-usd`). Optional and
//! defaulted, so an older core still parses; a consumer building `Plan` with a
//! struct literal must add the field (none did at release).

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
