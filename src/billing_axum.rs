//! Optional axum 0.7 router over [`BillingService`]: the four routes every
//! product's billing page needs, written once.
//!
//! ```text
//! GET  /subscription   state (provisions the trial on first use)
//! GET  /plans          the plans on sale
//! POST /checkout       {back_url?, payer_email?} -> {checkout_url}
//! POST /cancel                                   -> {status:"cancelled"}
//! ```
//!
//! Authentication is the product's: implement [`IdentitySource`] to turn a
//! request's headers into a [`CompanyIdentity`] (JWT claims plus wherever the
//! company name and email come from). Nest the result under the product's own
//! prefix. What this router does *not* carry is anything ez-stock-specific
//! (usage bars, limits, entitlements): those stay in ez-commerce.
//!
//! Errors are `{"error": CODE}` with stable codes, so a frontend can translate.

use std::future::Future;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::billing::state::{SubscriptionKind, SubscriptionState};
use crate::billing::types::{CheckoutDeclined, CheckoutOutcome, Plan};
use crate::billing::{BillingError, BillingService, CompanyIdentity};

/// An error response: a status and a stable, screaming-snake code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HttpError {
    pub status: StatusCode,
    pub code: &'static str,
}

impl HttpError {
    #[must_use]
    pub const fn new(status: StatusCode, code: &'static str) -> Self {
        Self { status, code }
    }

    /// What an [`IdentitySource`] returns for a missing or invalid credential.
    #[must_use]
    pub const fn unauthorized() -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "UNAUTHORIZED")
    }
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        (self.status, Json(serde_json::json!({ "error": self.code }))).into_response()
    }
}

impl From<BillingError> for HttpError {
    fn from(e: BillingError) -> Self {
        match e {
            BillingError::NotFound(_) => Self::new(StatusCode::NOT_FOUND, "SUBSCRIPTION_NOT_FOUND"),
            BillingError::PlanUnavailable(_) => {
                Self::new(StatusCode::SERVICE_UNAVAILABLE, "SUBSCRIPTION_PLAN_UNAVAILABLE")
            }
            _ => Self::new(StatusCode::BAD_GATEWAY, "BILLING_UPSTREAM_ERROR"),
        }
    }
}

/// How a product ties a request to a company.
pub trait IdentitySource: Send + Sync + 'static {
    fn identify(
        &self,
        headers: &HeaderMap,
    ) -> impl Future<Output = Result<CompanyIdentity, HttpError>> + Send;
}

struct Shared<I> {
    service: BillingService,
    identity: I,
}

/// Mount under the product's prefix, e.g. `.nest("/cochera", router(..))`.
pub fn router<I: IdentitySource>(service: BillingService, identity: I) -> Router {
    Router::new()
        .route("/subscription", get(subscription::<I>))
        .route("/plans", get(plans::<I>))
        .route("/checkout", post(checkout::<I>))
        .route("/cancel", post(cancel::<I>))
        .with_state(Arc::new(Shared { service, identity }))
}

#[derive(Serialize)]
struct SubscriptionView {
    status: String,
    state: SubscriptionKind,
    plan: Option<String>,
    trial_ends_at: Option<DateTime<Utc>>,
    trial_days_remaining: Option<i64>,
    grace_ends_at: Option<DateTime<Utc>>,
}

impl SubscriptionView {
    fn new(s: SubscriptionState, now: DateTime<Utc>) -> Self {
        Self {
            trial_days_remaining: s.trial_days_remaining(now),
            status: s.raw_status,
            state: s.kind,
            plan: s.plan_slug,
            trial_ends_at: s.trial_ends_at,
            grace_ends_at: s.grace_ends_at,
        }
    }
}

#[derive(Serialize)]
struct PlanView {
    /// The slug: what the product and its frontend refer to a plan by.
    id: String,
    name: String,
    price_cents: i64,
    /// Display only; `price_cents` is the exact amount.
    price_ars: f64,
    interval: &'static str,
}

impl From<Plan> for PlanView {
    fn from(p: Plan) -> Self {
        Self {
            interval: p.normalized_interval(),
            price_ars: p.price_cents as f64 / 100.0,
            id: p.slug,
            name: p.name,
            price_cents: p.price_cents,
        }
    }
}

#[derive(Deserialize)]
struct CheckoutBody {
    back_url: Option<String>,
    payer_email: Option<String>,
}

async fn subscription<I: IdentitySource>(
    State(s): State<Arc<Shared<I>>>,
    headers: HeaderMap,
) -> Result<Json<SubscriptionView>, HttpError> {
    let who = s.identity.identify(&headers).await?;
    let state = s.service.state_provisioning(&who).await?;
    Ok(Json(SubscriptionView::new(state, Utc::now())))
}

async fn plans<I: IdentitySource>(
    State(s): State<Arc<Shared<I>>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, HttpError> {
    s.identity.identify(&headers).await?;
    let plans: Vec<PlanView> = s.service.plans().await?.into_iter().map(Into::into).collect();
    Ok(Json(serde_json::json!({ "plans": plans })))
}

async fn checkout<I: IdentitySource>(
    State(s): State<Arc<Shared<I>>>,
    headers: HeaderMap,
    Json(body): Json<CheckoutBody>,
) -> Result<Json<serde_json::Value>, HttpError> {
    let who = s.identity.identify(&headers).await?;
    let back_url = body.back_url.as_deref().unwrap_or("/");
    match s.service.checkout(&who, back_url, None, body.payer_email.as_deref()).await? {
        CheckoutOutcome::Url(url) => Ok(Json(serde_json::json!({ "checkout_url": url }))),
        CheckoutOutcome::Declined(why) => Err(match why {
            CheckoutDeclined::AlreadyPending => {
                HttpError::new(StatusCode::CONFLICT, "SUBSCRIPTION_PAYMENT_ALREADY_SCHEDULED")
            }
            CheckoutDeclined::ProviderUnreachable => {
                HttpError::new(StatusCode::CONFLICT, "MERCADOPAGO_UNREACHABLE")
            }
            CheckoutDeclined::StalePreapprovalCancelFailed => HttpError::new(
                StatusCode::CONFLICT,
                "MERCADOPAGO_PREVIOUS_SUBSCRIPTION_NOT_CANCELLED",
            ),
            CheckoutDeclined::Other(_) => {
                HttpError::new(StatusCode::BAD_GATEWAY, "BILLING_UPSTREAM_ERROR")
            }
        }),
    }
}

async fn cancel<I: IdentitySource>(
    State(s): State<Arc<Shared<I>>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, HttpError> {
    let who = s.identity.identify(&headers).await?;
    s.service.cancel(&who.company_id, None).await?;
    Ok(Json(serde_json::json!({ "status": "cancelled" })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::billing::{BillingClient, BillingConfig, ServiceConfig};
    use axum::body::Body;
    use axum::http::Request;
    use serde_json::{json, Value};
    use tower::ServiceExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Authenticates by the presence of an `x-company` header.
    struct FakeIdentity;

    impl IdentitySource for FakeIdentity {
        async fn identify(&self, headers: &HeaderMap) -> Result<CompanyIdentity, HttpError> {
            let id = headers.get("x-company").and_then(|v| v.to_str().ok()).ok_or(HttpError::unauthorized())?;
            Ok(CompanyIdentity { company_id: id.into(), name: "Garage".into(), email: "o@g.io".into() })
        }
    }

    fn app(server: &MockServer) -> Router {
        let client = BillingClient::new(reqwest::Client::new(), BillingConfig::new(server.uri(), "k"));
        router(BillingService::new(client, ServiceConfig::new("cochera", 30)), FakeIdentity)
    }

    async fn call(app: Router, verb: &str, uri: &str, company: Option<&str>, body: Option<Value>) -> (StatusCode, Value) {
        let mut req = Request::builder().method(verb).uri(uri);
        if let Some(c) = company {
            req = req.header("x-company", c);
        }
        let req = match body {
            Some(b) => req.header("content-type", "application/json").body(Body::from(b.to_string())),
            None => req.body(Body::empty()),
        }
        .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    fn plan_json() -> Value {
        json!({"id":"p1","name":"Cochera","slug":"cochera","priceCents":3000000,"currency":"ARS","interval":"month","limits":[]})
    }

    fn sub_json(id: &str, status: &str, trial: Option<String>) -> Value {
        json!({"id":id,"status":status,"planId":"p1","plan":plan_json(),"trialEndsAt":trial,"currentPeriodEnd":(Utc::now()+chrono::Duration::days(20)).to_rfc3339()})
    }

    async fn mount(server: &MockServer, verb: &str, p: &str, status: u16, body: Value) {
        Mock::given(method(verb)).and(path(p)).respond_with(ResponseTemplate::new(status).set_body_json(body)).mount(server).await;
    }

    async fn known_customer_with(server: &MockServer, rows: Vec<Value>) {
        mount(server, "GET", "/v1/customers/external/co-1", 200, json!({"id":"c1","externalId":"co-1"})).await;
        mount(server, "GET", "/v1/subscriptions/customer/c1", 200, json!({"data":rows,"total":rows.len()})).await;
        mount(server, "GET", "/v1/plans", 200, json!({"data":[plan_json()],"total":1})).await;
    }

    #[tokio::test]
    async fn every_route_rejects_an_unidentified_request_with_401() {
        let server = MockServer::start().await;
        for (verb, uri) in [("GET", "/subscription"), ("GET", "/plans"), ("POST", "/cancel")] {
            let (status, body) = call(app(&server), verb, uri, None, None).await;
            assert_eq!((status, body["error"].as_str()), (StatusCode::UNAUTHORIZED, Some("UNAUTHORIZED")), "{verb} {uri}");
        }
        let (status, _) = call(app(&server), "POST", "/checkout", None, Some(json!({}))).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn subscription_returns_the_snake_case_view() {
        let server = MockServer::start().await;
        let trial_end = (Utc::now() + chrono::Duration::days(10)).to_rfc3339();
        known_customer_with(&server, vec![sub_json("s1", "trialing", Some(trial_end))]).await;
        let (status, body) = call(app(&server), "GET", "/subscription", Some("co-1"), None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["state"], "trial_active");
        assert_eq!(body["plan"], "cochera");
        assert_eq!(body["status"], "trialing");
        assert!(body["trial_days_remaining"].as_i64().unwrap() >= 9);
        assert!(body["grace_ends_at"].is_null());
    }

    #[tokio::test]
    async fn plans_lists_slug_id_and_both_price_forms() {
        let server = MockServer::start().await;
        known_customer_with(&server, vec![]).await;
        let (status, body) = call(app(&server), "GET", "/plans", Some("co-1"), None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["plans"][0],
            json!({"id":"cochera","name":"Cochera","price_cents":3000000,"price_ars":30000.0,"interval":"month"})
        );
    }

    #[tokio::test]
    async fn checkout_returns_the_payment_url() {
        let server = MockServer::start().await;
        known_customer_with(&server, vec![sub_json("s1", "incomplete", None)]).await;
        mount(&server, "POST", "/v1/payments/checkout", 201, json!({"checkoutUrl":"https://mp/init"})).await;
        let (status, body) = call(app(&server), "POST", "/checkout", Some("co-1"), Some(json!({"back_url":"/plan"}))).await;
        assert_eq!((status, body["checkout_url"].as_str()), (StatusCode::OK, Some("https://mp/init")));
    }

    #[tokio::test]
    async fn a_declined_checkout_maps_to_a_409_with_a_stable_code() {
        for (reason, code) in [
            ("checkout_already_pending", "SUBSCRIPTION_PAYMENT_ALREADY_SCHEDULED"),
            ("provider_unreachable", "MERCADOPAGO_UNREACHABLE"),
            ("stale_preapproval_cancel_failed", "MERCADOPAGO_PREVIOUS_SUBSCRIPTION_NOT_CANCELLED"),
        ] {
            let server = MockServer::start().await;
            known_customer_with(&server, vec![sub_json("s1", "incomplete", None)]).await;
            mount(&server, "POST", "/v1/payments/checkout", 201, json!({"checkoutUrl":null,"reason":reason})).await;
            let (status, body) = call(app(&server), "POST", "/checkout", Some("co-1"), Some(json!({}))).await;
            assert_eq!((status, body["error"].as_str()), (StatusCode::CONFLICT, Some(code)), "{reason}");
        }
    }

    #[tokio::test]
    async fn cancel_succeeds_and_reports_what_is_missing() {
        let server = MockServer::start().await;
        known_customer_with(&server, vec![sub_json("s1", "active", None)]).await;
        mount(&server, "DELETE", "/v1/subscriptions/s1", 204, Value::Null).await;
        let (status, body) = call(app(&server), "POST", "/cancel", Some("co-1"), None).await;
        assert_eq!((status, body["status"].as_str()), (StatusCode::OK, Some("cancelled")));

        let server = MockServer::start().await;
        mount(&server, "GET", "/v1/customers/external/co-1", 404, Value::Null).await;
        let (status, body) = call(app(&server), "POST", "/cancel", Some("co-1"), None).await;
        assert_eq!((status, body["error"].as_str()), (StatusCode::NOT_FOUND, Some("SUBSCRIPTION_NOT_FOUND")));
    }

    #[tokio::test]
    async fn an_inactive_plan_is_a_503_and_a_dead_upstream_a_502() {
        let server = MockServer::start().await;
        mount(&server, "GET", "/v1/customers/external/co-1", 200, json!({"id":"c1"})).await;
        mount(&server, "GET", "/v1/plans", 200, json!({"data":[],"total":0})).await;
        mount(&server, "GET", "/v1/subscriptions/customer/c1", 200, json!({"data":[],"total":0})).await;
        let (status, body) = call(app(&server), "GET", "/subscription", Some("co-1"), None).await;
        assert_eq!((status, body["error"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("SUBSCRIPTION_PLAN_UNAVAILABLE")));

        let dead = BillingClient::new(reqwest::Client::new(), BillingConfig::new("http://127.0.0.1:1", "k"));
        let app = router(BillingService::new(dead, ServiceConfig::new("cochera", 30)), FakeIdentity);
        let (status, body) = call(app, "GET", "/plans", Some("co-1"), None).await;
        assert_eq!((status, body["error"].as_str()), (StatusCode::BAD_GATEWAY, Some("BILLING_UPSTREAM_ERROR")));
    }
}
