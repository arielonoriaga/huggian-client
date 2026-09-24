//! The subscription flow itself — the part every product re-implemented.
//!
//! `BillingService` = [`BillingClient`] + [`StateCache`] + a product's
//! [`ServiceConfig`] (plan slug, trial length). It is single-plan by design:
//! products that sell one base plan (cochera) need nothing more, and plan
//! changes / add-ons stay in the product that has them (ez-commerce).
//!
//! Trials are created by the *caller*, not by Huggian (core has no per-product
//! trial setting): [`BillingService::ensure_trial`] passes `trialEndsAt`, and the
//! period stays the plan's own (core rejects any other period).

use std::sync::Arc;
use std::time::Duration;

use chrono::{Duration as Days, Utc};

use super::cache::{StateCache, DEFAULT_TTL};
use super::client::BillingClient;
use super::error::BillingError;
use super::state::{resolve_state, SubscriptionKind, SubscriptionState};
use super::types::{CheckoutOutcome, Customer, Plan, Subscription};

/// Who is being billed. The product supplies it: Huggian needs a name and email
/// to create the customer, and `company_id` becomes the customer's `external_id`.
#[derive(Debug, Clone)]
pub struct CompanyIdentity {
    pub company_id: String,
    pub name: String,
    pub email: String,
}

#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// The base plan this product sells (`GET /v1/plans` slug).
    pub plan_slug: String,
    /// Trial length granted on first use. `0` disables trial provisioning.
    pub trial_days: i64,
    pub provider: String,
    pub cache_ttl: Duration,
}

impl ServiceConfig {
    #[must_use]
    pub fn new(plan_slug: impl Into<String>, trial_days: i64) -> Self {
        Self {
            plan_slug: plan_slug.into(),
            trial_days,
            provider: "mercadopago".to_string(),
            cache_ttl: DEFAULT_TTL,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TrialOutcome {
    /// A new trialing subscription was created.
    Created(Subscription),
    /// The customer already had a subscription on this plan (any status). A
    /// cancelled or expired one is returned as-is: no second free trial.
    Existing(Subscription),
}

#[derive(Debug, Clone)]
pub struct BillingService {
    client: Arc<BillingClient>,
    cache: StateCache,
    cfg: Arc<ServiceConfig>,
}

impl BillingService {
    #[must_use]
    pub fn new(client: BillingClient, cfg: ServiceConfig) -> Self {
        Self {
            client: Arc::new(client),
            cache: StateCache::new(cfg.cache_ttl),
            cfg: Arc::new(cfg),
        }
    }

    #[must_use]
    pub fn config(&self) -> &ServiceConfig {
        &self.cfg
    }

    /// The product's active plans (`cochera` stays out until it is activated).
    pub async fn plans(&self) -> Result<Vec<Plan>, BillingError> {
        self.client.list_plans().await
    }

    /// The company's state, cached per company. Read-only: never creates anything.
    pub async fn state(&self, company_id: &str) -> Result<SubscriptionState, BillingError> {
        if let Some(cached) = self.cache.get(company_id) {
            return Ok(cached);
        }
        let state = match self.client.find_customer_by_external_id(company_id).await? {
            None => SubscriptionState::never_subscribed(),
            Some(customer) => {
                let subs = self.client.customer_subscriptions(&customer.id).await?;
                resolve_state(&subs, Utc::now())
            }
        };
        self.cache.put(company_id, state.clone());
        Ok(state)
    }

    /// Like [`state`](Self::state), but a company that has never subscribed gets
    /// its trial first (lazily, on first use), then its real state is returned.
    pub async fn state_provisioning(
        &self,
        who: &CompanyIdentity,
    ) -> Result<SubscriptionState, BillingError> {
        let state = self.state(&who.company_id).await?;
        if state.kind != SubscriptionKind::NeverSubscribed || self.cfg.trial_days <= 0 {
            return Ok(state);
        }
        self.ensure_trial(who).await?;
        self.state(&who.company_id).await
    }

    /// Find-or-create the customer, then start the trial unless they already have
    /// a subscription on this plan. Safe to call twice: core also dedupes.
    pub async fn ensure_trial(&self, who: &CompanyIdentity) -> Result<TrialOutcome, BillingError> {
        let customer = self.customer_for(who).await?;
        let plan = self.plan().await?;
        let subs = self.client.customer_subscriptions(&customer.id).await?;
        if let Some(existing) = subs.into_iter().find(|s| s.plan_id == plan.id) {
            return Ok(TrialOutcome::Existing(existing));
        }
        let now = Utc::now();
        let created = self
            .client
            .create_subscription(
                &customer.id,
                &plan.id,
                &self.cfg.provider,
                now,
                plan.period_end_from(now),
                Some(now + Days::days(self.cfg.trial_days)),
            )
            .await?;
        self.cache.invalidate(&who.company_id);
        Ok(TrialOutcome::Created(created))
    }

    /// A payment link for the plan. Reuses the customer's open subscription (an
    /// expired trial included); a cancelled one gets a fresh subscription, with no
    /// trial. `payer_email` defaults to the identity's email.
    pub async fn checkout(
        &self,
        who: &CompanyIdentity,
        back_url: &str,
        success_url: Option<&str>,
        payer_email: Option<&str>,
    ) -> Result<CheckoutOutcome, BillingError> {
        let customer = self.customer_for(who).await?;
        let plan = self.plan().await?;
        let subs = self.client.customer_subscriptions(&customer.id).await?;
        let sub = match subs.into_iter().find(|s| s.plan_id == plan.id && s.is_open()) {
            Some(open) => open,
            None => {
                let now = Utc::now();
                self.client
                    .create_subscription(
                        &customer.id,
                        &plan.id,
                        &self.cfg.provider,
                        now,
                        plan.period_end_from(now),
                        None,
                    )
                    .await?
            }
        };
        let outcome = self
            .client
            .initiate_checkout(
                &sub.id,
                &self.cfg.provider,
                payer_email.unwrap_or(&who.email),
                back_url,
                success_url,
            )
            .await?;
        self.cache.invalidate(&who.company_id);
        Ok(outcome)
    }

    /// Cancels the company's open subscription. `NotFound("customer")` /
    /// `NotFound("subscription")` say what was missing.
    pub async fn cancel(&self, company_id: &str, reason: Option<&str>) -> Result<(), BillingError> {
        let customer = self
            .client
            .find_customer_by_external_id(company_id)
            .await?
            .ok_or(BillingError::NotFound("customer"))?;
        let sub = self
            .client
            .customer_subscriptions(&customer.id)
            .await?
            .into_iter()
            .find(Subscription::is_open)
            .ok_or(BillingError::NotFound("subscription"))?;
        self.client.cancel_subscription(&sub.id, reason).await?;
        self.cache.invalidate(company_id);
        Ok(())
    }

    async fn customer_for(&self, who: &CompanyIdentity) -> Result<Customer, BillingError> {
        match self.client.find_customer_by_external_id(&who.company_id).await? {
            Some(existing) => Ok(existing),
            None => {
                self.client
                    .create_customer(&who.company_id, &who.name, &who.email, None)
                    .await
            }
        }
    }

    async fn plan(&self) -> Result<Plan, BillingError> {
        self.client
            .list_plans()
            .await?
            .into_iter()
            .find(|p| p.slug == self.cfg.plan_slug)
            .ok_or_else(|| BillingError::PlanUnavailable(self.cfg.plan_slug.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::billing::client::BillingConfig;
    use crate::billing::types::CheckoutDeclined;
    use serde_json::{json, Value};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    fn who() -> CompanyIdentity {
        CompanyIdentity {
            company_id: "co-1".into(),
            name: "Garage Norte".into(),
            email: "owner@garage.io".into(),
        }
    }

    fn service(server: &MockServer, trial_days: i64) -> BillingService {
        BillingService::new(
            BillingClient::new(reqwest::Client::new(), BillingConfig::new(server.uri(), "k")),
            ServiceConfig::new("cochera", trial_days),
        )
    }

    fn in_days(d: i64) -> String {
        (Utc::now() + Days::days(d)).to_rfc3339()
    }

    fn plan_json() -> Value {
        json!({"id":"p1","name":"Cochera","slug":"cochera","priceCents":3000000,"currency":"ARS","interval":"month","limits":[]})
    }

    fn sub_json(id: &str, status: &str, trial: Option<String>) -> Value {
        json!({"id":id,"status":status,"planId":"p1","plan":plan_json(),"trialEndsAt":trial,"currentPeriodEnd":in_days(20)})
    }

    async fn customer(server: &MockServer, expect: Option<u64>) {
        let m = Mock::given(method("GET"))
            .and(path("/v1/customers/external/co-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"id":"c1","externalId":"co-1"})));
        match expect { Some(n) => m.expect(n), None => m }.mount(server).await;
    }

    async fn no_customer(server: &MockServer) {
        Mock::given(method("GET"))
            .and(path("/v1/customers/external/co-1"))
            .respond_with(ResponseTemplate::new(404))
            .mount(server)
            .await;
    }

    async fn plans(server: &MockServer, active: bool) {
        let data = if active { vec![plan_json()] } else { vec![] };
        Mock::given(method("GET"))
            .and(path("/v1/plans"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data":data,"total":data.len()})))
            .mount(server)
            .await;
    }

    async fn subs(server: &MockServer, rows: Vec<Value>, expect: Option<u64>) {
        let m = Mock::given(method("GET"))
            .and(path("/v1/subscriptions/customer/c1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data":rows,"total":rows.len()})));
        match expect { Some(n) => m.expect(n), None => m }.mount(server).await;
    }

    async fn creates_subscription(server: &MockServer, times: u64, check: impl Fn(&Value) -> bool + Send + Sync + 'static) {
        Mock::given(method("POST"))
            .and(path("/v1/subscriptions"))
            .and(move |req: &Request| serde_json::from_slice::<Value>(&req.body).map(|b| check(&b)).unwrap_or(false))
            .respond_with(ResponseTemplate::new(201).set_body_json(sub_json("new", "trialing", Some(in_days(30)))))
            .expect(times)
            .mount(server)
            .await;
    }

    fn secs(b: &Value, key: &str) -> chrono::DateTime<Utc> {
        chrono::DateTime::parse_from_rfc3339(b[key].as_str().unwrap()).unwrap().with_timezone(&Utc)
    }

    // -- state -----------------------------------------------------------------

    #[tokio::test]
    async fn state_is_never_subscribed_without_a_customer_and_is_cached() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).and(path("/v1/customers/external/co-1"))
            .respond_with(ResponseTemplate::new(404)).expect(1).mount(&server).await;
        let svc = service(&server, 30);
        assert_eq!(svc.state("co-1").await.unwrap().kind, SubscriptionKind::NeverSubscribed);
        assert_eq!(svc.state("co-1").await.unwrap().kind, SubscriptionKind::NeverSubscribed);
    }

    #[tokio::test]
    async fn state_resolves_a_live_trial() {
        let server = MockServer::start().await;
        customer(&server, None).await;
        subs(&server, vec![sub_json("s1", "trialing", Some(in_days(10)))], None).await;
        let s = service(&server, 30).state("co-1").await.unwrap();
        assert_eq!(s.kind, SubscriptionKind::TrialActive);
        assert_eq!(s.plan_slug.as_deref(), Some("cochera"));
    }

    #[tokio::test]
    async fn a_second_read_inside_the_ttl_makes_no_new_round_trip() {
        let server = MockServer::start().await;
        customer(&server, Some(1)).await;
        subs(&server, vec![sub_json("s1", "active", None)], Some(1)).await;
        let svc = service(&server, 30);
        svc.state("co-1").await.unwrap();
        assert_eq!(svc.state("co-1").await.unwrap().kind, SubscriptionKind::Active);
    }

    // -- trial -----------------------------------------------------------------

    #[tokio::test]
    async fn ensure_trial_creates_the_customer_and_a_trial_that_leaves_the_plan_period_alone() {
        let server = MockServer::start().await;
        no_customer(&server).await;
        Mock::given(method("POST")).and(path("/v1/customers"))
            .respond_with(ResponseTemplate::new(201).set_body_json(json!({"id":"c1"}))).expect(1).mount(&server).await;
        plans(&server, true).await;
        subs(&server, vec![], None).await;
        creates_subscription(&server, 1, |b| {
            let (start, end, trial) = (secs(b, "currentPeriodStart"), secs(b, "currentPeriodEnd"), secs(b, "trialEndsAt"));
            b["customerId"] == "c1" && b["planId"] == "p1" && b["paymentProvider"] == "mercadopago"
                && end - start == Days::days(30)      // the plan's period, exactly
                && trial - start == Days::days(45)    // the trial is independent of it
        }).await;
        let out = service(&server, 45).ensure_trial(&who()).await.unwrap();
        assert!(matches!(out, TrialOutcome::Created(_)));
    }

    #[tokio::test]
    async fn ensure_trial_twice_never_creates_a_second_subscription() {
        let server = MockServer::start().await;
        customer(&server, None).await;
        plans(&server, true).await;
        subs(&server, vec![sub_json("s1", "trialing", Some(in_days(5)))], None).await;
        creates_subscription(&server, 0, |_| true).await;
        assert!(matches!(service(&server, 30).ensure_trial(&who()).await.unwrap(), TrialOutcome::Existing(_)));
    }

    #[tokio::test]
    async fn a_cancelled_customer_does_not_get_a_second_free_trial() {
        let server = MockServer::start().await;
        customer(&server, None).await;
        plans(&server, true).await;
        subs(&server, vec![sub_json("s1", "cancelled", Some(in_days(-40)))], None).await;
        creates_subscription(&server, 0, |_| true).await;
        assert!(matches!(service(&server, 30).ensure_trial(&who()).await.unwrap(), TrialOutcome::Existing(_)));
    }

    #[tokio::test]
    async fn an_inactive_plan_is_reported_as_unavailable() {
        let server = MockServer::start().await;
        customer(&server, None).await;
        plans(&server, false).await;
        match service(&server, 30).ensure_trial(&who()).await {
            Err(BillingError::PlanUnavailable(slug)) => assert_eq!(slug, "cochera"),
            other => panic!("expected PlanUnavailable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn state_provisioning_starts_the_trial_once_and_returns_the_real_state() {
        let server = MockServer::start().await;
        // customer: absent for the first read, present afterwards
        Mock::given(method("GET")).and(path("/v1/customers/external/co-1"))
            .respond_with(ResponseTemplate::new(404)).up_to_n_times(1).mount(&server).await;
        customer(&server, None).await;
        plans(&server, true).await;
        // subscriptions: empty while provisioning, then the new trial
        Mock::given(method("GET")).and(path("/v1/subscriptions/customer/c1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data":[],"total":0})))
            .up_to_n_times(1).mount(&server).await;
        subs(&server, vec![sub_json("new", "trialing", Some(in_days(30)))], None).await;
        creates_subscription(&server, 1, |_| true).await;
        let s = service(&server, 30).state_provisioning(&who()).await.unwrap();
        assert_eq!(s.kind, SubscriptionKind::TrialActive);
    }

    #[tokio::test]
    async fn zero_trial_days_disables_provisioning() {
        let server = MockServer::start().await;
        no_customer(&server).await;
        creates_subscription(&server, 0, |_| true).await;
        let s = service(&server, 0).state_provisioning(&who()).await.unwrap();
        assert_eq!(s.kind, SubscriptionKind::NeverSubscribed);
    }

    // -- checkout --------------------------------------------------------------

    async fn checkout_mock(server: &MockServer, sub_id: &'static str, body: Value) {
        Mock::given(method("POST")).and(path("/v1/payments/checkout"))
            .and(move |req: &Request| {
                serde_json::from_slice::<Value>(&req.body).map(|b| b["subscriptionId"] == sub_id && b["customerEmail"] == "owner@garage.io" && b["provider"] == "mercadopago").unwrap_or(false)
            })
            .respond_with(ResponseTemplate::new(201).set_body_json(body)).expect(1).mount(server).await;
    }

    #[tokio::test]
    async fn checkout_reuses_an_expired_trial_and_returns_the_payment_url() {
        let server = MockServer::start().await;
        customer(&server, None).await;
        plans(&server, true).await;
        subs(&server, vec![sub_json("s1", "trialing", Some(in_days(-3)))], None).await;
        creates_subscription(&server, 0, |_| true).await;
        checkout_mock(&server, "s1", json!({"checkoutUrl":"https://mp/init"})).await;
        let out = service(&server, 30).checkout(&who(), "/plan", None, None).await.unwrap();
        assert_eq!(out, CheckoutOutcome::Url("https://mp/init".into()));
    }

    #[tokio::test]
    async fn checkout_after_a_cancellation_creates_a_new_subscription_without_a_trial() {
        let server = MockServer::start().await;
        customer(&server, None).await;
        plans(&server, true).await;
        subs(&server, vec![sub_json("old", "cancelled", None)], None).await;
        creates_subscription(&server, 1, |b| b.get("trialEndsAt").is_none()).await;
        checkout_mock(&server, "new", json!({"checkoutUrl":"https://mp/init"})).await;
        service(&server, 30).checkout(&who(), "/plan", None, None).await.unwrap();
    }

    #[tokio::test]
    async fn a_declined_checkout_is_returned_not_raised() {
        let server = MockServer::start().await;
        customer(&server, None).await;
        plans(&server, true).await;
        subs(&server, vec![sub_json("s1", "incomplete", None)], None).await;
        checkout_mock(&server, "s1", json!({"checkoutUrl":null,"reason":"provider_unreachable"})).await;
        let out = service(&server, 30).checkout(&who(), "/plan", None, None).await.unwrap();
        assert_eq!(out, CheckoutOutcome::Declined(CheckoutDeclined::ProviderUnreachable));
    }

    #[tokio::test]
    async fn checkout_and_cancel_invalidate_the_cached_state() {
        let server = MockServer::start().await;
        customer(&server, None).await;
        plans(&server, true).await;
        subs(&server, vec![sub_json("s1", "incomplete", None)], None).await;
        checkout_mock(&server, "s1", json!({"checkoutUrl":"https://mp/init"})).await;
        Mock::given(method("DELETE")).and(path("/v1/subscriptions/s1"))
            .respond_with(ResponseTemplate::new(204)).expect(1).mount(&server).await;

        let svc = service(&server, 30);
        svc.state("co-1").await.unwrap();
        assert!(svc.cache.get("co-1").is_some());
        svc.checkout(&who(), "/plan", None, None).await.unwrap();
        assert!(svc.cache.get("co-1").is_none(), "checkout must drop the cached state");
        svc.state("co-1").await.unwrap();
        svc.cancel("co-1", None).await.unwrap();
        assert!(svc.cache.get("co-1").is_none(), "cancel must drop the cached state");
    }

    // -- cancel ----------------------------------------------------------------

    #[tokio::test]
    async fn cancel_targets_the_open_subscription_and_forwards_the_reason() {
        let server = MockServer::start().await;
        customer(&server, None).await;
        subs(&server, vec![sub_json("gone", "cancelled", None), sub_json("live", "active", None)], None).await;
        Mock::given(method("DELETE")).and(path("/v1/subscriptions/live"))
            .and(move |req: &Request| serde_json::from_slice::<Value>(&req.body).map(|b| b["reason"] == "moving").unwrap_or(false))
            .respond_with(ResponseTemplate::new(204)).expect(1).mount(&server).await;
        service(&server, 30).cancel("co-1", Some("moving")).await.unwrap();
    }

    #[tokio::test]
    async fn cancel_says_what_is_missing() {
        let server = MockServer::start().await;
        no_customer(&server).await;
        assert!(matches!(service(&server, 30).cancel("co-1", None).await, Err(BillingError::NotFound("customer"))));

        let server = MockServer::start().await;
        customer(&server, None).await;
        subs(&server, vec![sub_json("gone", "cancelled", None)], None).await;
        assert!(matches!(service(&server, 30).cancel("co-1", None).await, Err(BillingError::NotFound("subscription"))));
    }
}
