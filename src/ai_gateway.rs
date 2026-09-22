//! Retrying, circuit-breaking HTTP client for huggian-core's AI gateway
//! (`POST /api/ai/complete`, `POST /api/ai/complete-structured`).
//!
//! Ported verbatim from ez-addon's `infrastructure::ai_gateway::huggian`
//! module (see that repo's docs/phase4-ai.md) so every consumer of
//! huggian's AI gateway shares one tuned retry/breaker implementation
//! instead of each hand-rolling its own. Deliberately does NOT define a
//! pluggable-provider trait or read env vars itself — those are a
//! consumer's own architecture (e.g. ez-addon's `AiGateway` trait plus its
//! dual `AI_HUGGIAN_SERVICE_TOKEN`/`CATALOG_HUGGIAN_SERVICE_TOKEN`
//! resolution) and stay in that consumer's code. This module is the wire
//! client + the error taxonomy that mirrors huggian's own contract.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::time::{sleep, timeout, Instant};

/// Best-effort parse of `error_code` from huggian's standard error JSON
/// shape `{ "error": "...", "error_code": "...", "request_id": "..." }`.
/// Returns `Unknown` if parse fails or field absent.
fn parse_error_code(body: &str) -> AiErrorCode {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .as_ref()
        .and_then(|v| v.get("error_code"))
        .and_then(|v| v.as_str())
        .map(AiErrorCode::parse)
        .unwrap_or(AiErrorCode::Unknown)
}

fn parse_error_message(body: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .as_ref()
        .and_then(|v| v.get("error"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Mirrors the closed `AIErrorCode` enum emitted by huggian's AI gateway
/// in the `error_code` JSON field. See
/// huggian-management/services/core/src/contexts/ai/domain/types.ts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiErrorCode {
    NoProvider,
    ProviderMisconfigured,
    UpstreamError,
    UpstreamTimeout,
    RateLimit,
    InvalidJson,
    InvalidToken,
    Unknown,
}

impl std::fmt::Display for AiErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AiErrorCode {
    pub fn parse(s: &str) -> Self {
        match s {
            "no_provider" => Self::NoProvider,
            "provider_misconfigured" => Self::ProviderMisconfigured,
            "upstream_error" => Self::UpstreamError,
            "upstream_timeout" => Self::UpstreamTimeout,
            "rate_limit" => Self::RateLimit,
            "invalid_json" => Self::InvalidJson,
            "invalid_token" => Self::InvalidToken,
            _ => Self::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoProvider => "no_provider",
            Self::ProviderMisconfigured => "provider_misconfigured",
            Self::UpstreamError => "upstream_error",
            Self::UpstreamTimeout => "upstream_timeout",
            Self::RateLimit => "rate_limit",
            Self::InvalidJson => "invalid_json",
            Self::InvalidToken => "invalid_token",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Error)]
pub enum AiGatewayError {
    #[error("no provider available (503)")]
    NoProvider,
    #[error("provider misconfigured (503): {0}")]
    ProviderMisconfigured(String),
    #[error("rate limited (429)")]
    RateLimited,
    #[error("upstream timeout (504)")]
    UpstreamTimeout,
    #[error("invalid json from provider (422)")]
    InvalidJson,
    #[error("upstream error {status} [{code}]: {body}")]
    Upstream {
        status: u16,
        code: AiErrorCode,
        body: String,
    },
    #[error("transport: {0}")]
    Transport(String),
    #[error("deserialization: {0}")]
    Deserialization(String),
    /// [`HuggianAiClient`]'s per-instance circuit breaker is open — too
    /// many consecutive availability failures, so this call was
    /// short-circuited with ZERO HTTP requests. Distinct from `NoProvider`
    /// (huggian's gateway itself saying "no provider configured", a 503
    /// wire response) — this one never left the process. See the incident
    /// this breaker exists for: a permanently-broken provider row amplified
    /// into ~3.1M error rows / 30 days via blind per-call retries
    /// (`post_with_retry`'s doc comment below).
    #[error("ai gateway circuit breaker open (last upstream failure: {last}) — call not attempted")]
    CircuitOpen { last: AiErrorCode },
}

impl AiGatewayError {
    /// The closed error-code taxonomy this client ever produces — a
    /// caller's degradation path should switch on this rather than
    /// re-deriving it from the variant, so a new caller can't accidentally
    /// branch on prose.
    pub fn code(&self) -> AiErrorCode {
        match self {
            AiGatewayError::NoProvider => AiErrorCode::NoProvider,
            AiGatewayError::ProviderMisconfigured(_) => AiErrorCode::ProviderMisconfigured,
            AiGatewayError::RateLimited => AiErrorCode::RateLimit,
            AiGatewayError::UpstreamTimeout => AiErrorCode::UpstreamTimeout,
            AiGatewayError::InvalidJson => AiErrorCode::InvalidJson,
            AiGatewayError::Upstream { code, .. } => *code,
            // Report the failure that OPENED the breaker, so usage rows and
            // the degradation path stay semantically identical to the call
            // we would have made had we attempted it. The distinct VARIANT
            // (and its Display) is what tells an operator this call never
            // left the process — without it, a suppressed call is
            // indistinguishable from a live upstream failure in every log
            // and metric.
            AiGatewayError::CircuitOpen { last } => *last,
            AiGatewayError::Transport(_) | AiGatewayError::Deserialization(_) => AiErrorCode::Unknown,
        }
    }
}

/// Token usage huggian's gateway echoes back in the `usage
/// {input_tokens, output_tokens}` field once a caller opts in wire-side.
/// Absent on any response from a gateway build that doesn't send it — a
/// caller must treat `None` as "unknown", never as "zero cost".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub text: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone)]
pub struct StructuredCompletion {
    pub data: Value,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone)]
pub struct HuggianAiConfig {
    pub base_url: String,
    pub service_token: String,
    pub timeout: Duration,
    pub max_retries: u32,
    /// Consecutive availability failures (transport/5xx/503/504/429 — see
    /// `is_availability_failure`) before the circuit breaker opens.
    pub breaker_threshold: u32,
    /// How long the breaker stays open before allowing one half-open probe.
    pub breaker_cooldown: Duration,
}

/// Only failures that mean "the provider/gateway is unreachable or down"
/// count toward tripping the breaker. `InvalidJson` (422) and
/// `Deserialization` are the model/gateway answering successfully with a
/// body we couldn't use — a formatting problem, not a downtime signal —
/// tripping the breaker on those would take a reachable gateway offline
/// over bad output. `CircuitOpen` never reaches this classifier: the
/// breaker short-circuits before a call is made, so there's no outcome to
/// record.
///
/// `Upstream` is deliberately NOT blanket-true. It is the catch-all built
/// for every response that isn't 429/503/504/422 (see
/// `post_with_retry_inner`), which includes the gateway's **client** errors:
/// a 400 from its zod validation (an oversized prompt, a bad `timeout_ms`)
/// and a 401 for a missing or rotated service token. Neither means the
/// provider is down. Counting them would let one malformed batch item — or
/// a token rotation — trip the breaker and deny service to every other
/// healthy caller on this instance: the same amplification shape this
/// breaker exists to prevent, triggered by a client bug instead of a
/// provider outage.
fn is_availability_failure(e: &AiGatewayError) -> bool {
    match e {
        AiGatewayError::RateLimited
        | AiGatewayError::UpstreamTimeout
        | AiGatewayError::NoProvider
        | AiGatewayError::ProviderMisconfigured(_)
        | AiGatewayError::Transport(_) => true,
        AiGatewayError::Upstream { status, .. } => (500..600).contains(status),
        AiGatewayError::InvalidJson
        | AiGatewayError::Deserialization(_)
        | AiGatewayError::CircuitOpen { .. } => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BreakerPhase {
    Closed,
    /// Rejecting every call with zero HTTP requests until `open_until`.
    Open,
    /// Cooldown elapsed; exactly one call is let through as a probe.
    HalfOpen,
}

struct BreakerState {
    phase: BreakerPhase,
    consecutive_failures: u32,
    open_until: Instant,
    /// The code of the failure that opened the breaker, replayed to callers
    /// while it stays open. Stored as the code rather than the whole error so
    /// no large body string is cloned on every short-circuited call.
    last_code: Option<AiErrorCode>,
}

/// Breaker scoped to ONE [`HuggianAiClient`] instance — callers that want
/// independent trip state per module/provider construct their own client.
///
/// Lock is std `Mutex`, taken briefly and never held across an `.await` —
/// `before_call`/`after_call` are the only two critical sections, both
/// synchronous. Poisoning is recovered rather than propagated
/// (`unwrap_or_else(PoisonError::into_inner)`): a panic anywhere inside a
/// critical section would otherwise make `CallGuard::drop` panic again while
/// already unwinding, and a panic during unwind aborts the whole process.
/// Breaker state is a failure counter, not an invariant worth killing the
/// process over.
///
/// Uses `tokio::time::Instant` (not `std::time::Instant`) so
/// `#[tokio::test(start_paused = true)]` + `tokio::time::advance` can move
/// the breaker's clock in tests without a real sleep.
struct CircuitBreaker {
    state: Mutex<BreakerState>,
    threshold: u32,
    cooldown: Duration,
}

impl CircuitBreaker {
    fn new(threshold: u32, cooldown: Duration) -> Self {
        Self {
            state: Mutex::new(BreakerState {
                phase: BreakerPhase::Closed,
                consecutive_failures: 0,
                open_until: Instant::now(),
                last_code: None,
            }),
            threshold,
            cooldown,
        }
    }

    /// Recovers a poisoned lock instead of propagating the panic — see the
    /// type's doc comment: panicking here would abort the process via
    /// `CallGuard::drop` during unwind.
    fn lock(&self) -> std::sync::MutexGuard<'_, BreakerState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn blocked(code: Option<AiErrorCode>) -> AiGatewayError {
        AiGatewayError::CircuitOpen { last: code.unwrap_or(AiErrorCode::Unknown) }
    }

    /// Called before every network attempt. `Err` means: breaker open (or
    /// a half-open probe is already in flight) — return immediately,
    /// making zero HTTP requests. `Ok` covers normal Closed traffic AND
    /// the single probe let through on the Open -> HalfOpen transition,
    /// and yields a [`CallGuard`] the caller MUST settle.
    fn before_call(&self) -> Result<CallGuard<'_>, AiGatewayError> {
        let mut s = self.lock();
        match s.phase {
            BreakerPhase::Closed => Ok(self.admit()),
            BreakerPhase::HalfOpen => Err(Self::blocked(s.last_code)),
            BreakerPhase::Open => {
                if Instant::now() < s.open_until {
                    Err(Self::blocked(s.last_code))
                } else {
                    s.phase = BreakerPhase::HalfOpen;
                    Ok(self.admit())
                }
            }
        }
    }

    fn admit(&self) -> CallGuard<'_> {
        CallGuard { breaker: self, settled: false }
    }

    /// Called once with the outcome of a call `before_call` admitted.
    fn after_call(&self, result: &Result<Value, AiGatewayError>) {
        let mut s = self.lock();
        match result {
            Ok(_) => {
                s.phase = BreakerPhase::Closed;
                s.consecutive_failures = 0;
                s.last_code = None;
            }
            Err(e) if is_availability_failure(e) => {
                s.last_code = Some(e.code());
                s.consecutive_failures = s.consecutive_failures.saturating_add(1);
                if s.phase == BreakerPhase::HalfOpen || s.consecutive_failures >= self.threshold {
                    // Arm the cooldown only on the TRANSITION into Open. Once
                    // `consecutive_failures >= threshold` it stays true until a
                    // success, so re-arming on every later failure let calls
                    // admitted before the trip — still completing, each up to
                    // `cfg.timeout` later — keep sliding `open_until` forward
                    // off the last straggler rather than the triggering
                    // failure, silently multiplying the intended cooldown.
                    if s.phase != BreakerPhase::Open {
                        s.open_until = Instant::now() + self.cooldown;
                    }
                    s.phase = BreakerPhase::Open;
                }
            }
            Err(_) => {
                // Non-availability failure: the gateway answered, so it's
                // reachable. Closes a half-open probe; otherwise doesn't
                // touch the Closed failure streak either way.
                if s.phase == BreakerPhase::HalfOpen {
                    s.phase = BreakerPhase::Closed;
                    s.consecutive_failures = 0;
                    s.last_code = None;
                }
            }
        }
    }
}

/// Handed out by [`CircuitBreaker::before_call`] for every admitted call and
/// settled with the call's outcome. If it is dropped WITHOUT being settled —
/// the future was cancelled (an axum handler whose client disconnected, a
/// `tokio::select!` that lost, a panic) — the attempt is recorded as an
/// availability failure rather than silently vanishing.
///
/// Without this, an admitted HALF-OPEN probe that never completes strands the
/// breaker in `HalfOpen` forever: `before_call` then rejects every subsequent
/// call, so the gateway stays dead for the life of the process, with zero HTTP
/// requests and no path back. Recording the abandoned probe as a failure sends
/// it to `Open` with a fresh cooldown instead, so it self-heals.
#[must_use = "an admitted call must be settled or the breaker records it as abandoned"]
struct CallGuard<'a> {
    breaker: &'a CircuitBreaker,
    settled: bool,
}

impl CallGuard<'_> {
    fn settle(mut self, result: &Result<Value, AiGatewayError>) {
        self.breaker.after_call(result);
        self.settled = true;
    }
}

impl Drop for CallGuard<'_> {
    fn drop(&mut self) {
        if !self.settled {
            self.breaker.after_call(&Err(AiGatewayError::Transport(
                "call abandoned before completion (cancelled or panicked)".to_string(),
            )));
        }
    }
}

pub struct HuggianAiClient {
    client: reqwest::Client,
    cfg: HuggianAiConfig,
    breaker: CircuitBreaker,
    /// Shared per-client bulkhead — every call through this instance passes
    /// through this boundary, so no single caller can overwhelm the
    /// upstream independently of the others sharing it.
    concurrency: Arc<Semaphore>,
}

impl HuggianAiClient {
    /// `concurrency`: max in-flight requests through this client. Callers
    /// that read it from env (as ez-addon's `AI_GATEWAY_CONCURRENCY` did)
    /// keep doing that themselves and pass the resolved value here — this
    /// crate does not read env vars.
    pub fn new(client: reqwest::Client, cfg: HuggianAiConfig, concurrency: usize) -> Self {
        let breaker = CircuitBreaker::new(cfg.breaker_threshold, cfg.breaker_cooldown);
        Self {
            client,
            cfg,
            breaker,
            concurrency: Arc::new(Semaphore::new(concurrency.clamp(1, 64))),
        }
    }

    async fn post_with_retry(&self, path: &str, body: &Value) -> Result<Value, AiGatewayError> {
        // Waiting for a saturated upstream budget is part of the logical
        // request deadline. Without this timeout, background callers could
        // accumulate behind the semaphore while HTTP callers hold sockets
        // indefinitely.
        let _permit = timeout(self.cfg.timeout, self.concurrency.acquire())
            .await
            .map_err(|_| AiGatewayError::UpstreamTimeout)?
            .map_err(|_| AiGatewayError::UpstreamTimeout)?;
        // Queue saturation is a local capacity signal, not evidence that the
        // provider is unhealthy; acquire the circuit-breaker guard only after
        // a slot exists so a local overload cannot trip the shared breaker.
        let guard = self.breaker.before_call()?;
        let result = self.post_with_retry_inner(path, body).await;
        guard.settle(&result);
        result
    }

    async fn post_with_retry_inner(&self, path: &str, body: &Value) -> Result<Value, AiGatewayError> {
        // Tolerate a trailing slash on the base URL — common config typo.
        let base = self.cfg.base_url.trim_end_matches('/');
        let url = format!("{base}{path}");
        let mut attempt: u32 = 0;
        loop {
            let res = self
                .client
                .post(&url)
                .timeout(self.cfg.timeout)
                .bearer_auth(&self.cfg.service_token)
                .json(body)
                .send()
                .await;
            match res {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return resp
                            .json::<Value>()
                            .await
                            .map_err(|e| AiGatewayError::Transport(e.to_string()));
                    }
                    let status_code = status.as_u16();
                    let body = resp.text().await.unwrap_or_default();
                    let code = parse_error_code(&body);

                    if status_code == 429 {
                        if attempt < self.cfg.max_retries {
                            backoff(attempt).await;
                            attempt += 1;
                            continue;
                        }
                        return Err(AiGatewayError::RateLimited);
                    }
                    if status_code == 503 {
                        if matches!(code, AiErrorCode::ProviderMisconfigured) {
                            return Err(AiGatewayError::ProviderMisconfigured(
                                parse_error_message(&body).unwrap_or_default(),
                            ));
                        }
                        return Err(AiGatewayError::NoProvider);
                    }
                    if status_code == 504 {
                        return Err(AiGatewayError::UpstreamTimeout);
                    }
                    if status_code == 422 {
                        // The body carries huggian's `raw` (first 1000 chars of
                        // what the model actually wrote) — the only place it
                        // survives, since neither side persists it.
                        return Err(AiGatewayError::InvalidJson);
                    }
                    // Incident (2026-08, ez-addon): a permanently-broken
                    // gemini provider row produced ~3.1M error rows in 30
                    // days (~117k/day). Mechanics: huggian's gateway maps
                    // ANY unrecognised provider exception to HTTP 502
                    // `upstream_error` — a FINAL classification, not a
                    // transient blip. This client used to retry every 5xx
                    // blindly, so one logical call became `max_retries + 1`
                    // HTTP requests and that many error rows.
                    //
                    // The gateway is the component that owns
                    // provider-level retry/fallback. A 5xx that carries a
                    // gateway-classified `error_code` means the gateway
                    // already made its decision — retrying it is
                    // amplification, not resilience. Only retry 5xx bodies
                    // we can't attribute to the gateway's own classifier
                    // (`Unknown` — an nginx 502, a crashed process, a proxy
                    // hiccup: not the gateway speaking).
                    if status.is_server_error()
                        && matches!(code, AiErrorCode::Unknown)
                        && attempt < self.cfg.max_retries
                    {
                        backoff(attempt).await;
                        attempt += 1;
                        continue;
                    }
                    return Err(AiGatewayError::Upstream { status: status_code, code, body });
                }
                Err(e) => {
                    if attempt < self.cfg.max_retries && (e.is_timeout() || e.is_connect()) {
                        backoff(attempt).await;
                        attempt += 1;
                        continue;
                    }
                    return Err(AiGatewayError::Transport(e.to_string()));
                }
            }
        }
    }

    pub async fn complete(
        &self,
        prompt: &str,
        kind: Option<&str>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<String, AiGatewayError> {
        Ok(self.complete_with_usage(prompt, kind, max_tokens, temperature).await?.text)
    }

    pub async fn complete_structured_value(
        &self,
        prompt: &str,
        json_schema: &Value,
        kind: Option<&str>,
    ) -> Result<Value, AiGatewayError> {
        Ok(self.complete_structured_with_usage(prompt, json_schema, kind).await?.data)
    }

    pub async fn complete_with_usage(
        &self,
        prompt: &str,
        kind: Option<&str>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<Completion, AiGatewayError> {
        let body = serde_json::to_value(CompleteRequest {
            prompt,
            kind,
            max_tokens,
            temperature,
        })
        .map_err(|e| AiGatewayError::Transport(e.to_string()))?;
        let v = self.post_with_retry("/api/ai/complete", &body).await?;
        let resp: CompleteResponse = serde_json::from_value(v)
            .map_err(|e| AiGatewayError::Deserialization(e.to_string()))?;
        Ok(Completion { text: resp.text, usage: resp.usage.map(Usage::from) })
    }

    pub async fn complete_structured_with_usage(
        &self,
        prompt: &str,
        json_schema: &Value,
        kind: Option<&str>,
    ) -> Result<StructuredCompletion, AiGatewayError> {
        let body = serde_json::to_value(StructuredRequest {
            prompt,
            json_schema,
            kind,
        })
        .map_err(|e| AiGatewayError::Transport(e.to_string()))?;
        let v = self.post_with_retry("/api/ai/complete-structured", &body).await?;
        let resp: StructuredResponse = serde_json::from_value(v)
            .map_err(|e| AiGatewayError::Deserialization(e.to_string()))?;
        Ok(StructuredCompletion { data: resp.data, usage: resp.usage.map(Usage::from) })
    }
}

async fn backoff(attempt: u32) {
    // Exponential cap with full jitter. Deterministic backoff synchronizes
    // callers that observed the same outage and creates a retry herd; random
    // delay spreads the next attempts over the whole allowed window.
    let cap_ms = std::cmp::min(200u64.saturating_mul(1 << attempt), 5_000);
    let delay_ms = rand::thread_rng().gen_range(0..=cap_ms);
    sleep(Duration::from_millis(delay_ms)).await;
}

#[derive(Debug, Serialize)]
struct CompleteRequest<'a> {
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

/// Wire shape of huggian's `usage {input_tokens, output_tokens}` field.
/// `#[serde(default)]` on the field that holds this in `CompleteResponse`/
/// `StructuredResponse` means a gateway build that hasn't shipped it yet
/// simply omits it — never a deserialization failure.
#[derive(Debug, Deserialize)]
struct UsageWire {
    input_tokens: u64,
    output_tokens: u64,
}

impl From<UsageWire> for Usage {
    fn from(u: UsageWire) -> Self {
        Usage { input_tokens: u.input_tokens, output_tokens: u.output_tokens }
    }
}

#[derive(Debug, Deserialize)]
struct CompleteResponse {
    text: String,
    #[serde(default)]
    usage: Option<UsageWire>,
}

#[derive(Debug, Serialize)]
struct StructuredRequest<'a> {
    prompt: &'a str,
    json_schema: &'a Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct StructuredResponse {
    data: Value,
    #[serde(default)]
    usage: Option<UsageWire>,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use serde_json::json;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn http_client() -> reqwest::Client {
        reqwest::Client::builder().build().unwrap()
    }

    fn base_cfg(base_url: String) -> HuggianAiConfig {
        HuggianAiConfig {
            base_url,
            service_token: "test-token".to_string(),
            timeout: Duration::from_secs(5),
            max_retries: 3,
            breaker_threshold: 5,
            breaker_cooldown: Duration::from_secs(30),
        }
    }

    fn client(cfg: HuggianAiConfig) -> HuggianAiClient {
        HuggianAiClient::new(http_client(), cfg, 8)
    }

    // --- Retry matrix ---------------------------------------------------

    #[tokio::test]
    async fn rate_limit_429_retries_up_to_max_then_fails() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(429)).mount(&server).await;
        let cfg = HuggianAiConfig { max_retries: 2, ..base_cfg(server.uri()) };
        let ai = client(cfg);

        let err = ai.complete("hi", None, None, None).await.unwrap_err();
        assert!(matches!(err, AiGatewayError::RateLimited));
        assert_eq!(server.received_requests().await.unwrap().len(), 3, "1 initial + 2 retries");
    }

    #[tokio::test]
    async fn transport_timeout_retries_then_fails() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(3)))
            .mount(&server)
            .await;
        // Wide margins on BOTH sides. A 30ms client timeout against a 150ms
        // delay raced under a loaded parallel suite: the timeout could expire
        // during connection setup, so the request never reached wiremock and
        // the received-count assertion saw 1 instead of 2. The timeout must be
        // long enough to always connect and short enough to always fire before
        // the response — 300ms against 3s leaves an order of magnitude either
        // way, at a cost of ~0.8s for the whole test.
        let cfg =
            HuggianAiConfig { timeout: Duration::from_millis(300), max_retries: 1, ..base_cfg(server.uri()) };
        let ai = client(cfg);

        let err = ai.complete("hi", None, None, None).await.unwrap_err();
        assert!(matches!(err, AiGatewayError::Transport(_)));
        assert_eq!(server.received_requests().await.unwrap().len(), 2, "1 initial + 1 retry");
    }

    #[tokio::test]
    async fn gateway_classified_502_upstream_error_returns_immediately_no_retry() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(502).set_body_json(json!({"error_code": "upstream_error", "error": "boom"})))
            .mount(&server)
            .await;
        let cfg = HuggianAiConfig { max_retries: 3, ..base_cfg(server.uri()) };
        let ai = client(cfg);

        let err = ai.complete("hi", None, None, None).await.unwrap_err();
        assert!(matches!(err, AiGatewayError::Upstream { status: 502, code: AiErrorCode::UpstreamError, .. }));
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            1,
            "the gateway already classified this — retrying it IS the incident"
        );
    }

    #[tokio::test]
    async fn unparseable_502_body_still_retries() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(502).set_body_string("<html>bad gateway</html>"))
            .mount(&server)
            .await;
        let cfg = HuggianAiConfig { max_retries: 2, ..base_cfg(server.uri()) };
        let ai = client(cfg);

        let err = ai.complete("hi", None, None, None).await.unwrap_err();
        assert!(matches!(err, AiGatewayError::Upstream { code: AiErrorCode::Unknown, .. }));
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            3,
            "no gateway classification found in the body -> old retry behaviour preserved"
        );
    }

    #[tokio::test]
    async fn no_provider_503_returns_immediately() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(503)).mount(&server).await;
        let cfg = HuggianAiConfig { max_retries: 3, ..base_cfg(server.uri()) };
        let ai = client(cfg);

        let err = ai.complete("hi", None, None, None).await.unwrap_err();
        assert!(matches!(err, AiGatewayError::NoProvider));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn upstream_timeout_504_returns_immediately() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(504)).mount(&server).await;
        let cfg = HuggianAiConfig { max_retries: 3, ..base_cfg(server.uri()) };
        let ai = client(cfg);

        let err = ai.complete("hi", None, None, None).await.unwrap_err();
        assert!(matches!(err, AiGatewayError::UpstreamTimeout));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn invalid_json_422_returns_immediately() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).respond_with(ResponseTemplate::new(422)).mount(&server).await;
        let cfg = HuggianAiConfig { max_retries: 3, ..base_cfg(server.uri()) };
        let ai = client(cfg);

        let err = ai.complete("hi", None, None, None).await.unwrap_err();
        assert!(matches!(err, AiGatewayError::InvalidJson));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    // --- Circuit breaker --------------------------------------------------

    #[tokio::test]
    async fn breaker_opens_after_threshold_and_blocks_next_call() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(502).set_body_json(json!({"error_code": "upstream_error"})))
            .mount(&server)
            .await;
        let cfg = HuggianAiConfig { breaker_threshold: 3, max_retries: 0, ..base_cfg(server.uri()) };
        let ai = client(cfg);

        for _ in 0..3 {
            let _ = ai.complete("hi", None, None, None).await;
        }
        assert_eq!(server.received_requests().await.unwrap().len(), 3, "threshold reached, one request each");

        // A suppressed call reports the DISTINCT `CircuitOpen` variant, so a
        // log or metric can tell "we never called" from "we called and it
        // failed" — while `code()` still replays the failure that opened the
        // breaker, keeping usage rows and the degradation path unchanged.
        let err = ai.complete("hi", None, None, None).await.unwrap_err();
        assert!(
            matches!(err, AiGatewayError::CircuitOpen { last: AiErrorCode::UpstreamError }),
            "breaker-open call must be distinguishable from a live failure, got {err:?}"
        );
        assert_eq!(err.code(), AiErrorCode::UpstreamError, "code() replays the opening failure");
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            3,
            "breaker-open call must make ZERO HTTP requests"
        );
    }

    // Real clock, short cooldown: `start_paused` cannot be combined with the
    // real HTTP these tests make against wiremock. Tokio auto-advances a paused
    // clock whenever it thinks the runtime is idle, which it does while a
    // request is in flight — firing reqwest's own timeout before the response
    // lands and turning a 429/502 assertion into a spurious Transport error.
    // Only the pure-breaker unit test below may stay paused.
    #[tokio::test]
    async fn half_open_probe_success_closes_breaker_and_resumes_traffic() {
        let server = MockServer::start().await;
        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();
        Mock::given(method("POST"))
            .respond_with(move |_req: &wiremock::Request| {
                if cc.fetch_add(1, Ordering::SeqCst) < 3 {
                    ResponseTemplate::new(502).set_body_json(json!({"error_code": "upstream_error"}))
                } else {
                    ResponseTemplate::new(200).set_body_json(json!({"text": "ok"}))
                }
            })
            .mount(&server)
            .await;
        let cfg = HuggianAiConfig {
            breaker_threshold: 3,
            breaker_cooldown: Duration::from_millis(600),
            max_retries: 0,
            ..base_cfg(server.uri())
        };
        let ai = client(cfg);

        for _ in 0..3 {
            let _ = ai.complete("hi", None, None, None).await;
        }
        assert_eq!(server.received_requests().await.unwrap().len(), 3, "breaker now open");

        // Still inside the cooldown window -> zero additional requests.
        let _ = ai.complete("hi", None, None, None).await;
        assert_eq!(server.received_requests().await.unwrap().len(), 3);

        // Cooldown elapses -> exactly one probe.
        tokio::time::sleep(Duration::from_millis(750)).await;
        let probe = ai.complete("hi", None, None, None).await;
        assert!(probe.is_ok(), "probe should hit the now-healthy mock: {probe:?}");
        assert_eq!(server.received_requests().await.unwrap().len(), 4, "exactly one probe request");

        // Breaker closed by the successful probe -> normal traffic resumes.
        let normal = ai.complete("hi", None, None, None).await;
        assert!(normal.is_ok());
        assert_eq!(server.received_requests().await.unwrap().len(), 5);
    }

    #[tokio::test]
    async fn half_open_probe_failure_reopens_breaker() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(502).set_body_json(json!({"error_code": "upstream_error"})))
            .mount(&server)
            .await;
        let cfg = HuggianAiConfig {
            breaker_threshold: 3,
            breaker_cooldown: Duration::from_millis(600),
            max_retries: 0,
            ..base_cfg(server.uri())
        };
        let ai = client(cfg);

        for _ in 0..3 {
            let _ = ai.complete("hi", None, None, None).await;
        }
        assert_eq!(server.received_requests().await.unwrap().len(), 3);

        tokio::time::sleep(Duration::from_millis(750)).await;
        let probe = ai.complete("hi", None, None, None).await;
        assert!(probe.is_err(), "provider is still broken, probe must fail");
        assert_eq!(server.received_requests().await.unwrap().len(), 4, "exactly one probe request");

        // Re-opened -> immediately blocked again, zero more requests.
        let _ = ai.complete("hi", None, None, None).await;
        assert_eq!(server.received_requests().await.unwrap().len(), 4);
    }

    /// A half-open probe that is ADMITTED but never completes — the future
    /// cancelled because a caller disconnected, a `tokio::select!` lost, or
    /// the task panicked — must not strand the breaker.
    ///
    /// Before `CallGuard`, `after_call` simply never ran on those paths, so
    /// the phase stayed `HalfOpen` and `before_call` rejected every later
    /// call with zero HTTP requests: the gateway was dead for the life of
    /// the process, and no cooldown could bring it back. Exercised at the
    /// breaker level so the cancellation is unambiguous rather than raced
    /// against a mock.
    #[tokio::test(start_paused = true)]
    async fn abandoned_half_open_probe_does_not_wedge_the_breaker() {
        let breaker = CircuitBreaker::new(2, Duration::from_secs(10));
        let fail: Result<Value, AiGatewayError> = Err(AiGatewayError::NoProvider);

        for _ in 0..2 {
            breaker.before_call().expect("closed breaker admits").settle(&fail);
        }
        assert!(breaker.before_call().is_err(), "threshold reached — breaker must be open");

        // Cooldown elapses, the probe is admitted, then abandoned: the guard
        // drops without ever being settled.
        tokio::time::advance(Duration::from_secs(11)).await;
        drop(breaker.before_call().expect("cooldown elapsed — probe admitted"));

        // The abandoned probe counts as a failure, so we are Open again with a
        // fresh cooldown — recoverable — rather than wedged in HalfOpen.
        tokio::time::advance(Duration::from_secs(11)).await;
        breaker
            .before_call()
            .expect("breaker must recover after an abandoned probe, not wedge forever")
            .settle(&Ok(Value::Null));

        assert!(breaker.before_call().is_ok(), "a successful probe closes the breaker");
    }

    /// The gateway's CLIENT errors must never trip the breaker. `Upstream` is
    /// the catch-all for everything that isn't 429/503/504/422, so it also
    /// carries huggian's 400 (zod rejected the request — an oversized prompt,
    /// a bad `timeout_ms`) and its 401 (missing or rotated service token).
    /// Neither means a provider is down.
    ///
    /// Counting them would let one malformed batch item, or a token rotation,
    /// trip the breaker and deny service to every other healthy caller sharing
    /// this instance — the original incident's amplification shape, triggered
    /// by a client bug instead of a provider outage.
    #[tokio::test]
    async fn client_4xx_never_trips_the_breaker() {
        for (status, body) in [
            (400u16, json!({"error_code": "unknown", "error": "invalid request"})),
            (401u16, json!({"error_code": "invalid_token", "error": "bad token"})),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(status).set_body_json(body))
                .mount(&server)
                .await;
            let cfg = HuggianAiConfig { breaker_threshold: 3, max_retries: 0, ..base_cfg(server.uri()) };
            let ai = client(cfg);

            // Well past the threshold — every one must still reach the network.
            for _ in 0..6 {
                let err = ai.complete("hi", None, None, None).await.unwrap_err();
                assert!(
                    !matches!(err, AiGatewayError::CircuitOpen { .. }),
                    "{status} tripped the breaker; a client error must not"
                );
            }
            assert_eq!(
                server.received_requests().await.unwrap().len(),
                6,
                "{status} must not open the breaker — all 6 calls reach the gateway"
            );
        }
    }

    #[tokio::test]
    async fn invalid_json_never_trips_the_breaker() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(422).set_body_json(json!({"error_code": "invalid_json"})))
            .mount(&server)
            .await;
        let cfg = HuggianAiConfig { breaker_threshold: 3, max_retries: 0, ..base_cfg(server.uri()) };
        let ai = client(cfg);

        // More consecutive 422s than the breaker threshold — must never trip.
        for _ in 0..5 {
            let err = ai.complete("hi", None, None, None).await.unwrap_err();
            assert!(matches!(err, AiGatewayError::InvalidJson));
        }
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            5,
            "invalid_json must never be short-circuited by the breaker"
        );
    }

    // --- Incident regression ------------------------------------------------

    /// 2026-08 incident (ez-addon): a permanently-broken gemini provider row
    /// produced ~3.1M error rows in 30 days (~117,000/day on consecutive
    /// days). Mechanics: huggian's gateway maps an unrecognised provider
    /// exception to HTTP 502 `upstream_error`; this client retried every 5xx
    /// blindly (max_retries=3 -> 4 requests/call), and a 100-item batch
    /// loop turned that into 100 * 4 = 400 requests per batch, ~115k/day.
    /// Post-fix: the gateway-classified 502 isn't retried at all, AND the
    /// breaker caps the damage across the whole batch once the provider
    /// proves itself down.
    #[tokio::test]
    async fn incident_2026_08_gemini_amplification_stays_bounded() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(502).set_body_json(json!({"error_code": "upstream_error"})))
            .mount(&server)
            .await;
        let cfg = HuggianAiConfig { breaker_threshold: 5, max_retries: 3, ..base_cfg(server.uri()) };
        let ai = client(cfg);

        for _ in 0..100 {
            let _ = ai.complete("hi", None, None, None).await;
        }

        let requests = server.received_requests().await.unwrap().len();
        assert_eq!(requests, 5, "breaker should cap total requests at the threshold, got {requests}");
        assert!(requests < 400, "pre-fix behaviour: 100 items * (max_retries+1) = 400 requests; got {requests}");
    }
}
