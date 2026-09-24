//! Per-company cache of a resolved [`SubscriptionState`].
//!
//! The answer changes a few times a month per company, so it is trusted for a
//! minute rather than costing two Huggian round trips (customer + subscriptions)
//! on every gated request. Uses tokio's `Instant` so tests can pause the clock.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::time::Instant;

use super::state::SubscriptionState;

pub const DEFAULT_TTL: Duration = Duration::from_secs(60);

/// Cheap to clone: clones share the same map.
#[derive(Debug, Clone)]
pub struct StateCache {
    ttl: Duration,
    inner: Arc<Mutex<HashMap<String, (Instant, SubscriptionState)>>>,
}

impl StateCache {
    #[must_use]
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            inner: Arc::default(),
        }
    }

    /// The cached state if it is still fresh.
    #[must_use]
    pub fn get(&self, company_id: &str) -> Option<SubscriptionState> {
        let map = self.lock();
        let (stored_at, state) = map.get(company_id)?;
        (stored_at.elapsed() < self.ttl).then(|| state.clone())
    }

    /// Stores a state, sweeping expired entries first so the map cannot grow
    /// with every company ever seen.
    pub fn put(&self, company_id: &str, state: SubscriptionState) {
        let mut map = self.lock();
        map.retain(|_, (stored_at, _)| stored_at.elapsed() < self.ttl);
        map.insert(company_id.to_string(), (Instant::now(), state));
    }

    /// Forget one company (after a checkout or cancel changed its state).
    pub fn invalidate(&self, company_id: &str) {
        self.lock().remove(company_id);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, (Instant, SubscriptionState)>> {
        // A poisoned lock only means another thread panicked mid-insert; the map
        // is still a valid cache, so keep serving it.
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Default for StateCache {
    fn default() -> Self {
        Self::new(DEFAULT_TTL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::billing::state::SubscriptionKind;

    fn state(kind: SubscriptionKind) -> SubscriptionState {
        SubscriptionState { kind, ..SubscriptionState::never_subscribed() }
    }

    #[tokio::test(start_paused = true)]
    async fn a_fresh_entry_is_served_and_a_stale_one_is_not() {
        let cache = StateCache::new(Duration::from_secs(60));
        cache.put("co-1", state(SubscriptionKind::Active));
        assert_eq!(cache.get("co-1").unwrap().kind, SubscriptionKind::Active);

        tokio::time::advance(Duration::from_secs(59)).await;
        assert!(cache.get("co-1").is_some());
        tokio::time::advance(Duration::from_secs(2)).await;
        assert!(cache.get("co-1").is_none());
    }

    #[test]
    fn entries_never_leak_between_companies() {
        let cache = StateCache::default();
        cache.put("co-1", state(SubscriptionKind::Active));
        assert!(cache.get("co-2").is_none());
    }

    #[test]
    fn invalidate_removes_only_the_named_company_and_tolerates_absence() {
        let cache = StateCache::default();
        cache.put("co-1", state(SubscriptionKind::Active));
        cache.put("co-2", state(SubscriptionKind::Cancelled));
        cache.invalidate("co-1");
        cache.invalidate("never-seen");
        assert!(cache.get("co-1").is_none());
        assert!(cache.get("co-2").is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn writing_sweeps_expired_entries_but_keeps_fresh_ones() {
        let cache = StateCache::new(Duration::from_secs(60));
        cache.put("old", state(SubscriptionKind::Active));
        tokio::time::advance(Duration::from_secs(61)).await;
        cache.put("fresh", state(SubscriptionKind::Active));
        assert_eq!(cache.inner.lock().unwrap().len(), 1);
        assert!(cache.get("fresh").is_some());
    }

    #[test]
    fn clones_share_the_same_entries() {
        let a = StateCache::default();
        let b = a.clone();
        a.put("co-1", state(SubscriptionKind::Active));
        assert!(b.get("co-1").is_some());
    }
}
