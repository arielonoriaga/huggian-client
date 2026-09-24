//! Thin HTTP client for the subscription subset of huggian-core's API.
//!
//! One attempt per call, no retry, no idempotency header: core reads none on
//! these routes. It is idempotent by natural key instead — `POST /customers`
//! and `POST /subscriptions` return the existing row — so a retried call is safe
//! by construction, and the caller decides whether to retry.

use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::{RequestBuilder, Response, StatusCode, Url};
use serde::{de::DeserializeOwned, Serialize};

use super::error::BillingError;
use super::types::{CheckoutOutcome, CheckoutResponse, Customer, Plan, Subscription};

/// Reads that sit on the request path of a gated call get a tight ceiling so a
/// cold cache cannot hold a request for the shared client's full timeout.
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct BillingConfig {
    pub base_url: String,
    /// The product's own `X-Api-Key` (e.g. `COCHERA_HUGGIAN_API_KEY`). Core
    /// derives the product — and so the customer namespace — from it.
    pub api_key: String,
    pub read_timeout: Duration,
}

impl BillingConfig {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            read_timeout: DEFAULT_READ_TIMEOUT,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BillingClient {
    http: reqwest::Client,
    cfg: BillingConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CreateCustomer<'a> {
    external_id: &'a str,
    name: &'a str,
    email: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<&'a serde_json::Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CreateSubscription<'a> {
    customer_id: &'a str,
    plan_id: &'a str,
    payment_provider: &'a str,
    current_period_start: String,
    current_period_end: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    trial_ends_at: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Checkout<'a> {
    subscription_id: &'a str,
    provider: &'a str,
    customer_email: &'a str,
    back_url: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    success_url: Option<&'a str>,
}

#[derive(Serialize)]
struct Cancel<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'a str>,
}

#[derive(serde::Deserialize)]
struct Page<T> {
    data: Vec<T>,
}

impl BillingClient {
    /// `http` is injected (like `ai_gateway`) so the product owns its pool and
    /// its overall timeout.
    #[must_use]
    pub fn new(http: reqwest::Client, cfg: BillingConfig) -> Self {
        Self { http, cfg }
    }

    /// `GET /v1/customers/external/{id}`; a 404 is `Ok(None)`.
    pub async fn find_customer_by_external_id(
        &self,
        external_id: &str,
    ) -> Result<Option<Customer>, BillingError> {
        let url = self.url(&["v1", "customers", "external", external_id])?;
        let req = self.http.get(url).timeout(self.cfg.read_timeout);
        match self.call("find customer", req).await {
            Ok(c) => Ok(Some(c)),
            Err(BillingError::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// `POST /v1/customers`. Returns the existing customer when one already
    /// exists for this product + `external_id`.
    pub async fn create_customer(
        &self,
        external_id: &str,
        name: &str,
        email: &str,
        metadata: Option<&serde_json::Value>,
    ) -> Result<Customer, BillingError> {
        let url = self.url(&["v1", "customers"])?;
        let body = CreateCustomer {
            external_id,
            name,
            email,
            metadata,
        };
        self.call("create customer", self.http.post(url).json(&body))
            .await
    }

    /// `GET /v1/plans?kind=plan`: this product's **active** plans only.
    pub async fn list_plans(&self) -> Result<Vec<Plan>, BillingError> {
        let url = self.url(&["v1", "plans"])?;
        let req = self.http.get(url).query(&[("kind", "plan")]);
        Ok(self.call::<Page<Plan>>("list plans", req).await?.data)
    }

    /// `POST /v1/subscriptions`. With `trial_ends_at` the subscription starts
    /// `trialing` and nothing is charged; without it, `incomplete`. Core requires
    /// `period_end == plan.period_end_from(period_start)`. Returns the customer's
    /// open subscription for the plan if one already exists.
    pub async fn create_subscription(
        &self,
        customer_id: &str,
        plan_id: &str,
        payment_provider: &str,
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
        trial_ends_at: Option<DateTime<Utc>>,
    ) -> Result<Subscription, BillingError> {
        let url = self.url(&["v1", "subscriptions"])?;
        let body = CreateSubscription {
            customer_id,
            plan_id,
            payment_provider,
            current_period_start: period_start.to_rfc3339(),
            current_period_end: period_end.to_rfc3339(),
            trial_ends_at: trial_ends_at.map(|t| t.to_rfc3339()),
        };
        self.call("create subscription", self.http.post(url).json(&body))
            .await
    }

    /// `GET /v1/subscriptions/customer/{id}`, keeping plan rows only. An unknown
    /// customer is an empty list, not a 404.
    pub async fn customer_subscriptions(
        &self,
        customer_id: &str,
    ) -> Result<Vec<Subscription>, BillingError> {
        let url = self.url(&["v1", "subscriptions", "customer", customer_id])?;
        let req = self.http.get(url).timeout(self.cfg.read_timeout);
        let mut subs = self
            .call::<Page<Subscription>>("get customer subscriptions", req)
            .await?
            .data;
        subs.retain(Subscription::is_plan);
        Ok(subs)
    }

    /// `POST /v1/payments/checkout`. `customer_email` may be `""`: core then
    /// uses the email on the customer record.
    pub async fn initiate_checkout(
        &self,
        subscription_id: &str,
        provider: &str,
        customer_email: &str,
        back_url: &str,
        success_url: Option<&str>,
    ) -> Result<CheckoutOutcome, BillingError> {
        let url = self.url(&["v1", "payments", "checkout"])?;
        let body = Checkout {
            subscription_id,
            provider,
            customer_email,
            back_url,
            success_url,
        };
        let resp: CheckoutResponse = self
            .call("initiate checkout", self.http.post(url).json(&body))
            .await?;
        Ok(resp.into())
    }

    /// `DELETE /v1/subscriptions/{id}`. Both 204 and 202 (`cascade_pending`,
    /// an add-on cascade that will finish later) are success.
    pub async fn cancel_subscription(
        &self,
        subscription_id: &str,
        reason: Option<&str>,
    ) -> Result<(), BillingError> {
        let url = self.url(&["v1", "subscriptions", subscription_id])?;
        self.send(
            "cancel subscription",
            self.http.delete(url).json(&Cancel { reason }),
        )
        .await
        .map(drop)
    }

    // -- plumbing ---------------------------------------------------------------

    /// Base URL + path segments, percent-encoded so an id can never add a path.
    fn url(&self, segments: &[&str]) -> Result<Url, BillingError> {
        let decode = |detail: String| BillingError::Decode {
            op: "build url",
            detail,
        };
        let mut url = Url::parse(&self.cfg.base_url).map_err(|e| decode(e.to_string()))?;
        url.path_segments_mut()
            .map_err(|()| decode("base url cannot carry a path".into()))?
            .pop_if_empty()
            .extend(segments);
        Ok(url)
    }

    async fn call<T: DeserializeOwned>(
        &self,
        op: &'static str,
        req: RequestBuilder,
    ) -> Result<T, BillingError> {
        let resp = self.send(op, req).await?;
        let body = resp.text().await.map_err(|e| BillingError::Decode {
            op,
            detail: format!("failed to read body: {e}"),
        })?;
        serde_json::from_str(&body).map_err(|e| BillingError::Decode {
            op,
            detail: format!("{e} | body: {body}"),
        })
    }

    async fn send(&self, op: &'static str, req: RequestBuilder) -> Result<Response, BillingError> {
        let resp = req
            .header("X-Api-Key", &self.cfg.api_key)
            .send()
            .await
            .map_err(|source| BillingError::Transport { op, source })?;
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        match status {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Err(BillingError::Unauthorized),
            StatusCode::NOT_FOUND => Err(BillingError::NotFound(op)),
            _ => Err(BillingError::Upstream {
                op,
                status: status.as_u16(),
                body: resp.text().await.unwrap_or_default(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn client(server: &MockServer) -> BillingClient {
        BillingClient::new(reqwest::Client::new(), BillingConfig::new(server.uri(), "key-1"))
    }

    fn at(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    #[tokio::test]
    async fn find_customer_sends_the_api_key_and_a_404_is_none() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/customers/external/co-1"))
            .and(header("X-Api-Key", "key-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({"id":"c1","externalId":"co-1","email":"a@b.c","name":"X","createdAt":"z"}),
            ))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/customers/external/nope"))
            .respond_with(ResponseTemplate::new(404).set_body_json(json!({"error":"x"})))
            .mount(&server)
            .await;

        let c = client(&server);
        assert_eq!(c.find_customer_by_external_id("co-1").await.unwrap().unwrap().id, "c1");
        assert!(c.find_customer_by_external_id("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn an_external_id_can_never_add_a_path_segment() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/customers/external/a%2Fb"))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;
        assert!(client(&server).find_customer_by_external_id("a/b").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn create_customer_body_is_camel_case_and_omits_absent_metadata() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/customers"))
            .and(header("X-Api-Key", "key-1"))
            .and(body_json(json!({"externalId":"co-1","name":"Garage","email":"o@x.io"})))
            .respond_with(ResponseTemplate::new(201).set_body_json(json!({"id":"c1"})))
            .expect(1)
            .mount(&server)
            .await;
        let c = client(&server).create_customer("co-1", "Garage", "o@x.io", None).await.unwrap();
        assert_eq!(c.id, "c1");
    }

    #[tokio::test]
    async fn list_plans_asks_for_plan_kind_and_unwraps_data() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/plans"))
            .and(query_param("kind", "plan"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data":[{"id":"p1","name":"Cochera","slug":"cochera","priceCents":3000000,"currency":"ARS","interval":"month","limits":[]}],
                "total":1,"page":1,"limit":20
            })))
            .mount(&server)
            .await;
        let plans = client(&server).list_plans().await.unwrap();
        assert_eq!((plans[0].slug.as_str(), plans[0].price_cents), ("cochera", 3_000_000));
    }

    #[tokio::test]
    async fn create_subscription_body_carries_the_trial_only_when_given() {
        let server = MockServer::start().await;
        let start = at("2026-09-23T12:00:00Z");
        let end = at("2026-10-23T12:00:00Z");
        let with_trial = json!({
            "customerId":"c1","planId":"p1","paymentProvider":"mercadopago",
            "currentPeriodStart":start.to_rfc3339(),"currentPeriodEnd":end.to_rfc3339(),
            "trialEndsAt":end.to_rfc3339()
        });
        let without = json!({
            "customerId":"c1","planId":"p1","paymentProvider":"mercadopago",
            "currentPeriodStart":start.to_rfc3339(),"currentPeriodEnd":end.to_rfc3339()
        });
        let ok = json!({"id":"s1","status":"trialing","planId":"p1"});
        Mock::given(method("POST")).and(path("/v1/subscriptions")).and(body_json(with_trial))
            .respond_with(ResponseTemplate::new(201).set_body_json(ok.clone())).expect(1).mount(&server).await;
        Mock::given(method("POST")).and(path("/v1/subscriptions")).and(body_json(without))
            .respond_with(ResponseTemplate::new(201).set_body_json(json!({"id":"s2","status":"incomplete","planId":"p1"}))).expect(1).mount(&server).await;

        let c = client(&server);
        let s = c.create_subscription("c1", "p1", "mercadopago", start, end, Some(end)).await.unwrap();
        assert_eq!(s.status, "trialing");
        let s = c.create_subscription("c1", "p1", "mercadopago", start, end, None).await.unwrap();
        assert_eq!(s.status, "incomplete");
    }

    #[tokio::test]
    async fn customer_subscriptions_drops_addon_rows_wherever_they_sit() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/subscriptions/customer/c1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data":[
                {"id":"a","status":"active","planId":"x","kind":"addon"},
                {"id":"p","status":"active","planId":"y","kind":"plan"},
                {"id":"o","status":"active","planId":"z"}
            ],"total":3})))
            .mount(&server)
            .await;
        let ids: Vec<_> = client(&server).customer_subscriptions("c1").await.unwrap().into_iter().map(|s| s.id).collect();
        assert_eq!(ids, ["p", "o"]);
    }

    #[tokio::test]
    async fn checkout_body_is_camel_case_and_a_declined_link_is_not_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/payments/checkout"))
            .and(body_json(json!({"subscriptionId":"s1","provider":"mercadopago","customerEmail":"","backUrl":"/plan"})))
            .respond_with(ResponseTemplate::new(201).set_body_json(json!({"id":null,"checkoutUrl":null,"reason":"checkout_already_pending"})))
            .expect(1)
            .mount(&server)
            .await;
        let out = client(&server).initiate_checkout("s1", "mercadopago", "", "/plan", None).await.unwrap();
        assert_eq!(out, CheckoutOutcome::Declined(super::super::types::CheckoutDeclined::AlreadyPending));
    }

    #[tokio::test]
    async fn checkout_success_returns_the_url_and_forwards_success_url() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/payments/checkout"))
            .and(body_json(json!({"subscriptionId":"s1","provider":"mercadopago","customerEmail":"a@b.c","backUrl":"/plan","successUrl":"https://app/x"})))
            .respond_with(ResponseTemplate::new(201).set_body_json(json!({"checkoutUrl":"https://mp/init"})))
            .expect(1)
            .mount(&server)
            .await;
        let out = client(&server).initiate_checkout("s1", "mercadopago", "a@b.c", "/plan", Some("https://app/x")).await.unwrap();
        assert_eq!(out, CheckoutOutcome::Url("https://mp/init".into()));
    }

    #[tokio::test]
    async fn cancel_is_a_delete_with_a_json_body_and_204_and_202_are_success() {
        let server = MockServer::start().await;
        Mock::given(method("DELETE")).and(path("/v1/subscriptions/s1")).and(header("X-Api-Key", "key-1")).and(body_json(json!({"reason":"moving"})))
            .respond_with(ResponseTemplate::new(204)).expect(1).mount(&server).await;
        Mock::given(method("DELETE")).and(path("/v1/subscriptions/s2")).and(body_json(json!({})))
            .respond_with(ResponseTemplate::new(202).set_body_json(json!({"status":"cascade_pending"}))).expect(1).mount(&server).await;
        Mock::given(method("DELETE")).and(path("/v1/subscriptions/gone"))
            .respond_with(ResponseTemplate::new(404)).mount(&server).await;

        let c = client(&server);
        c.cancel_subscription("s1", Some("moving")).await.unwrap();
        c.cancel_subscription("s2", None).await.unwrap();
        assert!(matches!(c.cancel_subscription("gone", None).await, Err(BillingError::NotFound(_))));
    }

    #[tokio::test]
    async fn statuses_map_to_typed_errors() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).and(path("/v1/plans")).respond_with(ResponseTemplate::new(403)).up_to_n_times(1).mount(&server).await;
        let c = client(&server);
        assert!(matches!(c.list_plans().await, Err(BillingError::Unauthorized)));

        Mock::given(method("GET")).and(path("/v1/plans")).respond_with(ResponseTemplate::new(500).set_body_string("boom")).up_to_n_times(1).mount(&server).await;
        match c.list_plans().await {
            Err(BillingError::Upstream { status, body, .. }) => assert_eq!((status, body.as_str()), (500, "boom")),
            other => panic!("expected Upstream, got {other:?}"),
        }

        Mock::given(method("GET")).and(path("/v1/plans")).respond_with(ResponseTemplate::new(200).set_body_string("not json")).mount(&server).await;
        assert!(matches!(c.list_plans().await, Err(BillingError::Decode { .. })));
    }

    #[tokio::test]
    async fn an_unreachable_host_is_a_transport_error() {
        let c = BillingClient::new(reqwest::Client::new(), BillingConfig::new("http://127.0.0.1:1", "k"));
        assert!(matches!(c.list_plans().await, Err(BillingError::Transport { .. })));
    }
}
