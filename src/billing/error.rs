use thiserror::Error;

/// Everything that can go wrong talking to Huggian's billing API.
///
/// Deliberately small and product-agnostic: each consumer maps these to its
/// own error/HTTP envelope (ez-commerce's `AppError`, cochera's `CocheraError`).
#[derive(Debug, Error)]
pub enum BillingError {
    /// 401/403 — the product API key is missing, wrong, or the product is inactive.
    #[error("huggian rejected the api key")]
    Unauthorized,
    /// 404 on an operation where a missing row is an error (the name says which).
    #[error("{0} not found")]
    NotFound(&'static str),
    /// Any other non-2xx answer.
    #[error("huggian {op} returned {status}: {body}")]
    Upstream {
        op: &'static str,
        status: u16,
        body: String,
    },
    /// The request never got an answer (connect/timeout).
    #[error("huggian {op} unreachable: {source}")]
    Transport {
        op: &'static str,
        #[source]
        source: reqwest::Error,
    },
    /// A 2xx whose body is not the shape we expect.
    #[error("huggian {op} sent an unparseable body: {detail}")]
    Decode { op: &'static str, detail: String },
    /// The product's plan slug is not in Huggian's *active* catalogue
    /// (`GET /v1/plans` lists active plans only, so an inactive plan lands here).
    #[error("plan `{0}` is not available in the catalogue")]
    PlanUnavailable(String),
}
