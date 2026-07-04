//! Shared primitives for integrating with huggian-core.
//!
//! v1 covers the webhook receiver layer only — the one piece that was
//! byte-for-byte duplicated across ez-catalog and ez-stock. The billing REST
//! client is a planned follow-up module; product-specific entitlement /
//! provisioning logic deliberately stays in each consuming service.

pub mod webhook;
