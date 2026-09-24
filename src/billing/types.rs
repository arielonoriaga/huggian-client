//! Wire types for Huggian's customer / plan / subscription / checkout API.
//!
//! Field names and defaults mirror what ez-commerce already deserializes
//! (`huggian_client.rs`), verified against huggian-core's DTOs. All are
//! `Deserialize`-only and tolerant of unknown fields on purpose: a new field on
//! core must never break every product's entitlement resolution.

use chrono::{DateTime, Duration, Months, Utc};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Customer {
    pub id: String,
    pub email: Option<String>,
    #[serde(rename = "externalId")]
    pub external_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Plan {
    pub id: String,
    pub name: String,
    pub slug: String,
    #[serde(rename = "priceCents")]
    pub price_cents: i64,
    pub currency: String,
    #[serde(default = "default_interval")]
    pub interval: String,
    #[serde(default)]
    pub limits: Vec<PlanLimit>,
}

fn default_interval() -> String {
    "month".to_string()
}

impl Plan {
    /// Monthly and annual are the only contracts; anything else is treated as
    /// monthly, matching the missing-field default.
    #[must_use]
    pub fn normalized_interval(&self) -> &'static str {
        if self.interval == "year" {
            "year"
        } else {
            "month"
        }
    }

    /// End of the first billing period. Core rejects a subscription whose
    /// `currentPeriodEnd` is not exactly this (month = +30 days, year = +12
    /// months), so the trial length goes in `trialEndsAt`, never here.
    #[must_use]
    pub fn period_end_from(&self, start: DateTime<Utc>) -> DateTime<Utc> {
        if self.normalized_interval() == "year" {
            start + Months::new(12)
        } else {
            start + Duration::days(30)
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlanLimit {
    pub key: String,
    pub value: i32,
}

/// Plan capacity plus add-on grants for one resource. `plan_value`/`addon_value`
/// default so a schema tweak on those fields cannot fail the whole subscription.
#[derive(Debug, Clone, Deserialize)]
pub struct EffectiveLimit {
    pub key: String,
    pub value: i32,
    #[serde(rename = "planValue", default)]
    pub plan_value: i32,
    #[serde(rename = "addonValue", default)]
    pub addon_value: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Subscription {
    pub id: String,
    pub status: String,
    #[serde(rename = "planId")]
    pub plan_id: String,
    pub plan: Option<Plan>,
    /// RFC 3339.
    #[serde(rename = "trialEndsAt")]
    pub trial_ends_at: Option<String>,
    /// RFC 3339.
    #[serde(rename = "currentPeriodEnd")]
    pub current_period_end: Option<String>,
    /// `"plan" | "addon"`. Absent means a plan: a core that predates the split
    /// only ever had plans. See [`Subscription::is_plan`].
    pub kind: Option<String>,
    /// Null when `plan` is null (customer with no plan) or on an old core.
    #[serde(rename = "effectiveLimits")]
    pub effective_limits: Option<Vec<EffectiveLimit>>,
}

impl Subscription {
    /// True for a plan row or a row from a core that predates `kind`. Fail
    /// restrictive: any other value (a future third kind, wrong case) is not a plan.
    #[must_use]
    pub fn is_plan(&self) -> bool {
        matches!(self.kind.as_deref(), Some("plan") | None)
    }

    /// Statuses core treats as terminal (they free the "one open subscription
    /// per customer+plan" slot). Everything else counts as open.
    #[must_use]
    pub fn is_open(&self) -> bool {
        !matches!(
            self.status.as_str(),
            "cancelled" | "incomplete_expired" | "expired"
        )
    }
}

/// What `POST /v1/payments/checkout` answered.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CheckoutResponse {
    #[serde(rename = "checkoutUrl")]
    pub checkout_url: Option<String>,
    pub reason: Option<String>,
}

/// The result of asking for a payment link. A declined checkout is a normal
/// answer (HTTP 201 with `checkoutUrl: null` + a `reason`), not an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckoutOutcome {
    Url(String),
    Declined(CheckoutDeclined),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckoutDeclined {
    /// A payment authorisation is already waiting for the payer.
    AlreadyPending,
    /// The payment provider could not be reached.
    ProviderUnreachable,
    /// An older authorisation could not be cancelled before minting a new one.
    StalePreapprovalCancelFailed,
    /// A reason this version does not know, carried through verbatim.
    Other(String),
}

impl From<CheckoutResponse> for CheckoutOutcome {
    fn from(r: CheckoutResponse) -> Self {
        if let Some(url) = r.checkout_url {
            return Self::Url(url);
        }
        Self::Declined(match r.reason.as_deref() {
            Some("checkout_already_pending") => CheckoutDeclined::AlreadyPending,
            Some("provider_unreachable") => CheckoutDeclined::ProviderUnreachable,
            Some("stale_preapproval_cancel_failed") => {
                CheckoutDeclined::StalePreapprovalCancelFailed
            }
            Some(other) => CheckoutDeclined::Other(other.to_string()),
            None => CheckoutDeclined::Other("no_checkout_url".to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn plan(interval: Option<&str>) -> Plan {
        let mut v = json!({"id":"p","name":"N","slug":"s","priceCents":1,"currency":"ARS","limits":[]});
        if let Some(i) = interval {
            v["interval"] = json!(i);
        }
        serde_json::from_value(v).unwrap()
    }

    fn at(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    #[test]
    fn missing_interval_defaults_to_monthly_and_unknown_is_normalised() {
        assert_eq!(plan(None).normalized_interval(), "month");
        assert_eq!(plan(Some("weekly")).normalized_interval(), "month");
        assert_eq!(plan(Some("year")).normalized_interval(), "year");
    }

    #[test]
    fn period_end_is_thirty_days_for_month_and_twelve_months_for_year() {
        let start = at("2026-01-31T00:00:00Z");
        assert_eq!(plan(None).period_end_from(start), at("2026-03-02T00:00:00Z"));
        assert_eq!(
            plan(Some("year")).period_end_from(start),
            at("2027-01-31T00:00:00Z")
        );
    }

    #[test]
    fn a_plan_without_limits_still_deserialises() {
        let p: Plan = serde_json::from_value(
            json!({"id":"p","name":"N","slug":"s","priceCents":3000000,"currency":"ARS"}),
        )
        .unwrap();
        assert!(p.limits.is_empty());
    }

    fn sub(extra: serde_json::Value) -> Subscription {
        let mut v = json!({"id":"s1","status":"active","planId":"p1","plan":null,"trialEndsAt":null,"currentPeriodEnd":null});
        v.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn kind_absent_or_plan_is_a_plan_and_anything_else_is_not() {
        assert!(sub(json!({})).is_plan());
        assert!(sub(json!({"kind":"plan"})).is_plan());
        assert!(!sub(json!({"kind":"addon"})).is_plan());
        assert!(!sub(json!({"kind":"Plan"})).is_plan());
    }

    #[test]
    fn effective_limits_null_and_unknown_fields_are_tolerated() {
        let s = sub(json!({"effectiveLimits":null,"somethingNew":1}));
        assert!(s.effective_limits.is_none());
        let s = sub(json!({"effectiveLimits":[{"key":"users","value":5}]}));
        let l = &s.effective_limits.unwrap()[0];
        assert_eq!((l.value, l.plan_value, l.addon_value), (5, 0, 0));
    }

    #[test]
    fn terminal_statuses_are_not_open() {
        for closed in ["cancelled", "incomplete_expired", "expired"] {
            assert!(!sub(json!({"status":closed})).is_open(), "{closed}");
        }
        for open in ["trialing", "active", "incomplete", "past_due"] {
            assert!(sub(json!({"status":open})).is_open(), "{open}");
        }
    }

    #[test]
    fn checkout_outcome_maps_url_and_every_known_reason() {
        let r = |u: Option<&str>, reason: Option<&str>| {
            CheckoutOutcome::from(CheckoutResponse {
                checkout_url: u.map(str::to_string),
                reason: reason.map(str::to_string),
            })
        };
        assert_eq!(r(Some("https://mp/x"), None), CheckoutOutcome::Url("https://mp/x".into()));
        assert_eq!(
            r(None, Some("checkout_already_pending")),
            CheckoutOutcome::Declined(CheckoutDeclined::AlreadyPending)
        );
        assert_eq!(
            r(None, Some("provider_unreachable")),
            CheckoutOutcome::Declined(CheckoutDeclined::ProviderUnreachable)
        );
        assert_eq!(
            r(None, Some("stale_preapproval_cancel_failed")),
            CheckoutOutcome::Declined(CheckoutDeclined::StalePreapprovalCancelFailed)
        );
        assert_eq!(
            r(None, Some("brand_new")),
            CheckoutOutcome::Declined(CheckoutDeclined::Other("brand_new".into()))
        );
        assert_eq!(
            r(None, None),
            CheckoutOutcome::Declined(CheckoutDeclined::Other("no_checkout_url".into()))
        );
    }
}
