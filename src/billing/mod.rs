//! Per-product subscription flow over huggian-core (see the crate docs).

pub mod cache;
pub mod client;
pub mod error;
pub mod service;
pub mod state;
pub mod types;

pub use client::{BillingClient, BillingConfig};
pub use error::BillingError;
pub use types::{CheckoutDeclined, CheckoutOutcome, Customer, Plan, Subscription};
pub use cache::StateCache;
pub use state::{resolve_state, SubscriptionKind, SubscriptionState, GRACE_DAYS};
pub use service::{BillingService, CompanyIdentity, ServiceConfig, TrialOutcome};
