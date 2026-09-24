//! Pure subscription-state rules: no I/O, `now` is injected so every branch is
//! testable without a clock. Ported from ez-commerce's `resolve_state.rs`; the
//! ez-stock-specific parts (plan tiers, limit floors, feature grants) are not
//! here — each product interprets `plan_slug` itself.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use super::types::Subscription;

/// How long a `past_due` subscription keeps granting access after its period ends.
pub const GRACE_DAYS: i64 = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubscriptionKind {
    NeverSubscribed,
    TrialActive,
    Active,
    PastDue,
    TrialExpired,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubscriptionState {
    pub kind: SubscriptionKind,
    pub plan_slug: Option<String>,
    /// Huggian's own status string, for display/debugging.
    pub raw_status: String,
    pub trial_ends_at: Option<DateTime<Utc>>,
    /// Only set while `PastDue`: when access stops.
    pub grace_ends_at: Option<DateTime<Utc>>,
}

impl SubscriptionState {
    #[must_use]
    pub fn never_subscribed() -> Self {
        Self {
            kind: SubscriptionKind::NeverSubscribed,
            plan_slug: None,
            raw_status: "never_subscribed".to_string(),
            trial_ends_at: None,
            grace_ends_at: None,
        }
    }

    /// Paying, in trial, or in the past-due grace window.
    #[must_use]
    pub fn grants_access(&self) -> bool {
        matches!(
            self.kind,
            SubscriptionKind::Active | SubscriptionKind::TrialActive | SubscriptionKind::PastDue
        )
    }

    /// Whole days left in the trial (never negative). `None` outside a trial.
    #[must_use]
    pub fn trial_days_remaining(&self, now: DateTime<Utc>) -> Option<i64> {
        match (self.kind, self.trial_ends_at) {
            (SubscriptionKind::TrialActive | SubscriptionKind::TrialExpired, Some(end)) => {
                Some((end - now).num_days().max(0))
            }
            _ => None,
        }
    }
}

fn parse_dt(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// The customer's state from their subscriptions. The first plan-kind row wins
/// (core returns the relevant one first); no plan row means `NeverSubscribed`.
#[must_use]
pub fn resolve_state(subs: &[Subscription], now: DateTime<Utc>) -> SubscriptionState {
    let Some(sub) = subs.iter().find(|s| s.is_plan()) else {
        return SubscriptionState::never_subscribed();
    };

    let trial_end = sub.trial_ends_at.as_deref().and_then(parse_dt);
    let period_end = sub.current_period_end.as_deref().and_then(parse_dt);

    let kind = match sub.status.as_str() {
        "active" => SubscriptionKind::Active,
        "trialing" => match trial_end {
            Some(t) if t > now => SubscriptionKind::TrialActive,
            _ => SubscriptionKind::TrialExpired,
        },
        "past_due" | "unpaid" => match period_end {
            Some(p) if now <= p + Duration::days(GRACE_DAYS) => SubscriptionKind::PastDue,
            _ => SubscriptionKind::Cancelled,
        },
        _ => SubscriptionKind::Cancelled,
    };

    let grace_ends_at = match kind {
        SubscriptionKind::PastDue => period_end.map(|p| p + Duration::days(GRACE_DAYS)),
        _ => None,
    };

    SubscriptionState {
        kind,
        plan_slug: sub.plan.as_ref().map(|p| p.slug.clone()),
        raw_status: sub.status.clone(),
        trial_ends_at: trial_end,
        grace_ends_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn now() -> DateTime<Utc> {
        parse_dt("2026-09-23T12:00:00Z").unwrap()
    }

    fn sub(status: &str, trial: Option<&str>, period_end: Option<&str>) -> Subscription {
        serde_json::from_value(json!({
            "id":"s1","status":status,"planId":"p1",
            "plan":{"id":"p1","name":"Cochera","slug":"cochera","priceCents":3000000,"currency":"ARS","interval":"month","limits":[]},
            "trialEndsAt":trial,"currentPeriodEnd":period_end
        }))
        .unwrap()
    }

    fn kind(subs: &[Subscription]) -> SubscriptionKind {
        resolve_state(subs, now()).kind
    }

    #[test]
    fn no_subscriptions_is_never_subscribed() {
        let s = resolve_state(&[], now());
        assert_eq!(s, SubscriptionState::never_subscribed());
        assert!(!s.grants_access());
    }

    #[test]
    fn an_addon_row_sorted_first_is_skipped_for_the_plan_row() {
        let addon: Subscription = serde_json::from_value(
            json!({"id":"a","status":"active","planId":"x","kind":"addon"}),
        )
        .unwrap();
        let subs = [addon, sub("trialing", Some("2026-10-01T00:00:00Z"), None)];
        assert_eq!(kind(&subs), SubscriptionKind::TrialActive);
    }

    #[test]
    fn only_addon_rows_is_never_subscribed() {
        let addon: Subscription = serde_json::from_value(
            json!({"id":"a","status":"active","planId":"x","kind":"addon"}),
        )
        .unwrap();
        assert_eq!(kind(&[addon]), SubscriptionKind::NeverSubscribed);
    }

    #[test]
    fn active_grants_access_and_carries_the_plan_slug() {
        let s = resolve_state(&[sub("active", None, None)], now());
        assert_eq!((s.kind, s.plan_slug.as_deref()), (SubscriptionKind::Active, Some("cochera")));
        assert!(s.grants_access());
    }

    #[test]
    fn trialing_with_a_future_end_is_trial_active_and_counts_days() {
        let s = resolve_state(&[sub("trialing", Some("2026-09-28T12:00:00Z"), None)], now());
        assert_eq!(s.kind, SubscriptionKind::TrialActive);
        assert!(s.grants_access());
        assert_eq!(s.trial_days_remaining(now()), Some(5));
    }

    #[test]
    fn trialing_with_a_past_or_missing_end_is_trial_expired() {
        assert_eq!(kind(&[sub("trialing", Some("2026-09-01T00:00:00Z"), None)]), SubscriptionKind::TrialExpired);
        assert_eq!(kind(&[sub("trialing", None, None)]), SubscriptionKind::TrialExpired);
        let s = resolve_state(&[sub("trialing", Some("2026-09-01T00:00:00Z"), None)], now());
        assert!(!s.grants_access());
        assert_eq!(s.trial_days_remaining(now()), Some(0), "never negative");
    }

    #[test]
    fn trial_days_are_none_outside_a_trial_even_if_a_trial_date_lingers() {
        let s = resolve_state(&[sub("active", Some("2026-09-01T00:00:00Z"), None)], now());
        assert_eq!(s.trial_days_remaining(now()), None);
    }

    #[test]
    fn past_due_inside_the_grace_window_still_grants_access() {
        let s = resolve_state(&[sub("past_due", None, Some("2026-09-20T12:00:00Z"))], now());
        assert_eq!(s.kind, SubscriptionKind::PastDue);
        assert!(s.grants_access());
        assert_eq!(s.grace_ends_at, parse_dt("2026-09-27T12:00:00Z"));
    }

    #[test]
    fn unpaid_is_treated_like_past_due() {
        assert_eq!(kind(&[sub("unpaid", None, Some("2026-09-20T12:00:00Z"))]), SubscriptionKind::PastDue);
    }

    #[test]
    fn past_due_beyond_grace_or_without_a_period_end_is_cancelled() {
        assert_eq!(kind(&[sub("past_due", None, Some("2026-09-01T00:00:00Z"))]), SubscriptionKind::Cancelled);
        assert_eq!(kind(&[sub("past_due", None, None)]), SubscriptionKind::Cancelled);
    }

    #[test]
    fn grace_boundary_is_inclusive() {
        // period_end + 7d == now exactly
        assert_eq!(kind(&[sub("past_due", None, Some("2026-09-16T12:00:00Z"))]), SubscriptionKind::PastDue);
        assert_eq!(kind(&[sub("past_due", None, Some("2026-09-16T11:59:59Z"))]), SubscriptionKind::Cancelled);
    }

    #[test]
    fn only_past_due_exposes_a_grace_deadline() {
        for status in ["active", "trialing", "cancelled"] {
            let s = resolve_state(&[sub(status, Some("2026-10-01T00:00:00Z"), Some("2026-09-20T12:00:00Z"))], now());
            assert_eq!(s.grace_ends_at, None, "{status}");
        }
    }

    #[test]
    fn unknown_or_terminal_statuses_are_cancelled() {
        for status in ["cancelled", "incomplete", "expired", "mystery"] {
            assert_eq!(kind(&[sub(status, None, None)]), SubscriptionKind::Cancelled, "{status}");
        }
    }

    #[test]
    fn kind_serialises_as_snake_case() {
        assert_eq!(serde_json::to_value(SubscriptionKind::TrialActive).unwrap(), json!("trial_active"));
        assert_eq!(serde_json::to_value(SubscriptionKind::NeverSubscribed).unwrap(), json!("never_subscribed"));
    }
}
