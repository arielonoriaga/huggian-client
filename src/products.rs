//! Read-only lookup against huggian-core's product registry
//! (`GET /v1/products`), cached with a refresh interval.
//!
//! Uses the *read-scoped* platform key — never the admin key that can
//! register a product or rotate another product's API key. Distributing
//! the admin key into every consumer of this crate would hand each of them
//! platform-registration power over the whole system for what should be a
//! read-only slug lookup — the same class of gap that first motivated
//! centralizing this at all.

use serde::Deserialize;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Product {
    pub id: Uuid,
    pub name: String,
    pub slug: String,
    pub webhook_url: Option<String>,
    pub is_active: bool,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
struct ProductsResponse {
    products: Vec<Product>,
}

#[derive(Debug)]
pub enum ProductsError {
    Request(String),
    UpstreamStatus(u16),
    UnknownSlug(String),
}

impl std::fmt::Display for ProductsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(msg) => write!(f, "request to huggian-core failed: {msg}"),
            Self::UpstreamStatus(code) => write!(f, "huggian-core returned status {code}"),
            Self::UnknownSlug(slug) => write!(f, "unknown product slug: {slug}"),
        }
    }
}

impl std::error::Error for ProductsError {}

struct Cache {
    fetched_at: Option<Instant>,
    by_slug: HashMap<String, Product>,
}

/// Default cache lifetime before `resolve` refetches the registry.
pub const DEFAULT_TTL: Duration = Duration::from_secs(300);

pub struct ProductsClient {
    base_url: String,
    read_key: String,
    http: reqwest::Client,
    cache: RwLock<Cache>,
    ttl: Duration,
}

impl ProductsClient {
    pub fn new(base_url: impl Into<String>, read_key: impl Into<String>) -> Self {
        Self::with_ttl(base_url, read_key, DEFAULT_TTL)
    }

    pub fn with_ttl(base_url: impl Into<String>, read_key: impl Into<String>, ttl: Duration) -> Self {
        Self {
            base_url: base_url.into(),
            read_key: read_key.into(),
            http: reqwest::Client::new(),
            cache: RwLock::new(Cache { fetched_at: None, by_slug: HashMap::new() }),
            ttl,
        }
    }

    /// Resolves a product slug to its registry row.
    ///
    /// Refreshes the cache when stale, or when the slug isn't in the current
    /// cache at all — so a product registered after this process's last
    /// refresh resolves without waiting out a full TTL, at the cost of one
    /// extra round trip for a genuinely unknown slug.
    pub async fn resolve(&self, slug: &str) -> Result<Product, ProductsError> {
        if let Some(product) = self.cached(slug).await {
            return Ok(product);
        }
        self.refresh().await?;
        self.cached(slug).await.ok_or_else(|| ProductsError::UnknownSlug(slug.to_string()))
    }

    async fn cached(&self, slug: &str) -> Option<Product> {
        let cache = self.cache.read().await;
        let fresh = cache.fetched_at.is_some_and(|at| at.elapsed() < self.ttl);
        if fresh { cache.by_slug.get(slug).cloned() } else { None }
    }

    /// Forces a re-fetch of the registry, regardless of TTL.
    pub async fn refresh(&self) -> Result<(), ProductsError> {
        let response = self
            .http
            .get(format!("{}/v1/products", self.base_url))
            .header("X-Platform-Key", &self.read_key)
            .send()
            .await
            .map_err(|e| ProductsError::Request(e.to_string()))?;
        if !response.status().is_success() {
            return Err(ProductsError::UpstreamStatus(response.status().as_u16()));
        }
        let body: ProductsResponse = response
            .json()
            .await
            .map_err(|e| ProductsError::Request(e.to_string()))?;
        let mut cache = self.cache.write().await;
        cache.by_slug = body.products.into_iter().map(|p| (p.slug.clone(), p)).collect();
        cache.fetched_at = Some(Instant::now());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sample_body() -> serde_json::Value {
        serde_json::json!({
            "products": [
                {
                    "id": "e91e4a01-49d9-4a56-a68a-4998bf61c2f0",
                    "name": "EZ Catalog",
                    "slug": "ez-catalog",
                    "webhookUrl": null,
                    "isActive": true,
                    "createdAt": "2026-05-05T00:04:43.361Z",
                },
            ],
        })
    }

    #[tokio::test]
    async fn resolve_fetches_and_caches_using_the_read_key() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/products"))
            .and(header("X-Platform-Key", "read-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(sample_body()))
            .expect(1) // second resolve() must hit the cache, not fetch again
            .mount(&server)
            .await;

        let client = ProductsClient::new(server.uri(), "read-key");
        let product = client.resolve("ez-catalog").await.unwrap();
        assert_eq!(product.slug, "ez-catalog");
        assert!(product.webhook_url.is_none());

        // Cached — no second request expected (the mock's `.expect(1)` fails
        // the test on drop if a second one is observed).
        client.resolve("ez-catalog").await.unwrap();
    }

    #[tokio::test]
    async fn resolve_returns_unknown_slug_for_a_slug_not_in_the_registry() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/products"))
            .respond_with(ResponseTemplate::new(200).set_body_json(sample_body()))
            .mount(&server)
            .await;

        let client = ProductsClient::new(server.uri(), "read-key");
        let err = client.resolve("does-not-exist").await.unwrap_err();
        assert!(matches!(err, ProductsError::UnknownSlug(slug) if slug == "does-not-exist"));
    }

    #[tokio::test]
    async fn resolve_surfaces_a_non_success_upstream_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/products"))
            .respond_with(ResponseTemplate::new(403))
            .mount(&server)
            .await;

        let client = ProductsClient::new(server.uri(), "wrong-key");
        let err = client.resolve("ez-catalog").await.unwrap_err();
        assert!(matches!(err, ProductsError::UpstreamStatus(403)));
    }

    #[tokio::test]
    async fn resolve_refetches_once_the_ttl_expires() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/products"))
            .respond_with(ResponseTemplate::new(200).set_body_json(sample_body()))
            .expect(2)
            .mount(&server)
            .await;

        let client = ProductsClient::with_ttl(server.uri(), "read-key", Duration::from_millis(1));
        client.resolve("ez-catalog").await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        client.resolve("ez-catalog").await.unwrap();
    }
}
