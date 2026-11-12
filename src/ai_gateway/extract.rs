//! `POST /api/ai/extract-document`: send a purchase document (images/PDF) to
//! the gateway's Claude extraction route and get schema-shaped JSON back.
//!
//! Single-shot by design (spec §7.1): the only retry in the extraction path is
//! the caller's deliberate fast → accurate escalation, so this never retries,
//! whatever `max_retries` the client was built with. A 504 here is usually a
//! long document that may already have been billed — a content outcome, so it
//! never counts toward this client's breaker. So does a client-side timeout:
//! the request timeout is derived from the gateway's own (`timeout_ms` + 10 s,
//! never below the client's configured timeout), so ours only fires once the
//! gateway has had its full window.
//!
//! Build a dedicated [`HuggianAiClient`] for extraction (own token, breaker
//! and bulkhead): sharing one with chat lets long extractions hold chat's
//! concurrency slots. `cost_usd` is the gateway's estimate for display and
//! alerting only — never feed it into a money ledger.

use std::time::Duration;

use reqwest::header::CONTENT_TYPE;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::timeout;

use super::{parse_error_code, AiErrorCode, AiGatewayError, HuggianAiClient, Usage, UsageWire};

const EXTRACT_PATH: &str = "/api/ai/extract-document";
const GATEWAY_DEFAULT_TIMEOUT_MS: u32 = 120_000;
/// Headroom past the gateway's timeout, so the gateway's 504 (a classified
/// content outcome) normally arrives before our own timer fires.
const GATEWAY_TIMEOUT_MARGIN: Duration = Duration::from_secs(10);

fn extraction_timeout(cfg_timeout: Duration, timeout_ms: Option<u32>) -> Duration {
    let gateway = Duration::from_millis(u64::from(timeout_ms.unwrap_or(GATEWAY_DEFAULT_TIMEOUT_MS)));
    cfg_timeout.max(gateway + GATEWAY_TIMEOUT_MARGIN)
}

/// The gateway's own limits, checked locally so a bad request never spends
/// an upload (or a breaker slot) to learn it from a 400.
fn validate(req: &ExtractRequest<'_>) -> Result<(), AiGatewayError> {
    if !(1..=5).contains(&req.attachments.len()) {
        return Err(AiGatewayError::InvalidRequest("attachments must be 1..=5"));
    }
    if !(1..=32_768).contains(&req.max_tokens) {
        return Err(AiGatewayError::InvalidRequest("max_tokens must be 1..=32768"));
    }
    if req.timeout_ms.is_some_and(|t| !(10_000..=120_000).contains(&t)) {
        return Err(AiGatewayError::InvalidRequest("timeout_ms must be 10000..=120000"));
    }
    Ok(())
}

#[derive(Clone, Copy, Serialize)]
pub struct DocumentAttachment<'a> {
    /// `image/jpeg`, `image/png`, `image/webp` or `application/pdf`.
    pub media_type: &'a str,
    pub data_b64: &'a str,
}

#[derive(Clone, Copy)]
pub struct ExtractRequest<'a> {
    /// `claude-haiku-5-5`, `claude-sonnet-5-5` or `claude-opus-5-5`.
    pub model: &'a str,
    pub system: &'a str,
    /// 1–5 attachments.
    pub attachments: &'a [DocumentAttachment<'a>],
    pub schema: &'a Value,
    /// ≤ 32768.
    pub max_tokens: u32,
    /// Gateway-side timeout, 10_000–120_000 ms; `None` = gateway default.
    pub timeout_ms: Option<u32>,
}

// Hand-written Debug: these hold the document and the prompt, which must
// never reach a log line through `{:?}`.
impl std::fmt::Debug for DocumentAttachment<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DocumentAttachment")
            .field("media_type", &self.media_type)
            .field("data_b64_len", &self.data_b64.len())
            .finish()
    }
}

impl std::fmt::Debug for ExtractRequest<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtractRequest")
            .field("model", &self.model)
            .field("attachments", &self.attachments)
            .field("max_tokens", &self.max_tokens)
            .field("timeout_ms", &self.timeout_ms)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    Refusal,
    MaxTokens,
    InvalidOutput,
}

/// A billed extraction. `data` is `Some` only for [`StopReason::EndTurn`].
#[derive(Clone)]
pub struct Extraction {
    pub data: Option<Value>,
    pub stop_reason: StopReason,
    pub model: String,
    pub usage: Usage,
    /// Gateway's USD price for the call; `None` when the gateway could not
    /// price it (treat as unknown, never as zero).
    pub cost_usd: Option<f64>,
    pub request_id: Option<String>,
}

// Hand-written Debug: `data` is the extracted document (CUITs, amounts) and
// must never reach a log line through `{:?}`; only its serialized size shows.
impl std::fmt::Debug for Extraction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.as_ref().map(|v| format!("Some({} bytes)", v.to_string().len()));
        f.debug_struct("Extraction")
            .field("data", &format_args!("{}", data.as_deref().unwrap_or("None")))
            .field("stop_reason", &self.stop_reason)
            .field("model", &self.model)
            .field("usage", &self.usage)
            .field("cost_usd", &self.cost_usd)
            .field("request_id", &self.request_id)
            .finish()
    }
}

/// Fixed text on purpose: serde's message quotes the offending value, and a
/// response value can be document content.
fn shape_error() -> AiGatewayError {
    AiGatewayError::Deserialization("unexpected extraction response shape".to_string())
}

#[derive(Serialize)]
struct ExtractWire<'a> {
    kind: &'static str,
    model: &'a str,
    system: &'a str,
    attachments: &'a [DocumentAttachment<'a>],
    schema: &'a Value,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout_ms: Option<u32>,
}

#[derive(Deserialize)]
struct ExtractResponseWire {
    #[serde(default)]
    data: Option<Value>,
    stop_reason: StopReason,
    model: String,
    usage: UsageWire,
    #[serde(default)]
    cost_usd: Option<f64>,
    #[serde(default)]
    request_id: Option<String>,
}

impl HuggianAiClient {
    pub async fn extract_document(&self, req: &ExtractRequest<'_>) -> Result<Extraction, AiGatewayError> {
        validate(req)?;
        let value = self.extract_guarded(req, extraction_timeout(self.cfg.timeout, req.timeout_ms)).await?;
        let resp: ExtractResponseWire = serde_json::from_value(value).map_err(|_| shape_error())?;
        Ok(Extraction {
            data: resp.data.filter(|_| resp.stop_reason == StopReason::EndTurn),
            stop_reason: resp.stop_reason,
            model: resp.model,
            usage: Usage::from(resp.usage),
            cost_usd: resp.cost_usd,
            request_id: resp.request_id,
        })
    }

    async fn extract_guarded(&self, req: &ExtractRequest<'_>, request_timeout: Duration) -> Result<Value, AiGatewayError> {
        // Same bulkhead-then-breaker order as `post_with_retry`: waiting for a
        // local slot must not count against the shared breaker.
        let _permit = timeout(request_timeout, self.concurrency.acquire())
            .await
            .map_err(|_| AiGatewayError::UpstreamTimeout)?
            .map_err(|_| AiGatewayError::UpstreamTimeout)?;
        // Lenient guard: if this future is dropped mid-call (the caller gave
        // up on a long document) the attempt is NOT recorded as an outage.
        let guard = self.breaker.before_call()?.abandon_is_not_an_outage();
        let result = self.post_extract_once(req, request_timeout).await;
        guard.settle(&result);
        result
    }

    async fn post_extract_once(&self, req: &ExtractRequest<'_>, request_timeout: Duration) -> Result<Value, AiGatewayError> {
        let url = format!("{}{EXTRACT_PATH}", self.cfg.base_url.trim_end_matches('/'));
        let body = ExtractWire {
            kind: "claude",
            model: req.model,
            system: req.system,
            attachments: req.attachments,
            schema: req.schema,
            max_tokens: req.max_tokens,
            timeout_ms: req.timeout_ms,
        };
        // Pre-sized: the base64 documents dominate, so this avoids the
        // repeated regrow-and-copy of a multi-MB body.
        let estimate = req.attachments.iter().map(|a| a.data_b64.len()).sum::<usize>() + req.system.len() + 4096;
        let mut bytes = Vec::with_capacity(estimate);
        serde_json::to_writer(&mut bytes, &body).map_err(|_| AiGatewayError::InvalidRequest("unserializable request"))?;
        let resp = self
            .client
            .post(&url)
            .timeout(request_timeout)
            .bearer_auth(&self.cfg.service_token)
            .header(CONTENT_TYPE, "application/json")
            .body(bytes)
            .send()
            .await
            .map_err(transport_or_timeout)?;

        let status = resp.status().as_u16();
        if resp.status().is_success() {
            // A 200 whose body is not JSON (a proxy page) is a content problem,
            // not downtime; only a failure to read the body is transport.
            return resp.json::<Value>().await.map_err(|e| {
                if e.is_decode() {
                    shape_error()
                } else {
                    transport_or_timeout(e)
                }
            });
        }
        let text = resp.text().await.unwrap_or_default();
        let code = parse_error_code(&text);
        Err(match (status, code) {
            (413 | 422, _) => AiGatewayError::InputRejected { status, upstream_status: parse_upstream_status(&text) },
            (429, _) => AiGatewayError::RateLimited,
            (503, AiErrorCode::DailyCapReached) => AiGatewayError::DailyCapReached,
            (503, AiErrorCode::Busy) => AiGatewayError::Busy,
            (503, AiErrorCode::ProviderMisconfigured) => {
                AiGatewayError::ProviderMisconfigured(String::new())
            }
            (503, _) => AiGatewayError::NoProvider,
            (504, _) => AiGatewayError::ExtractionTimeout,
            // Body dropped on purpose: the code carries the meaning, and an
            // upstream that echoes request content must not leak it into logs.
            _ => AiGatewayError::Upstream { status, code, body: String::new() },
        })
    }
}

/// Our own timer firing on an established connection (on send or while
/// reading the body) is a long document, like the gateway's 504 — a content
/// outcome, not downtime. A connect timeout is the gateway being unreachable,
/// so it stays `Transport` and counts toward the breaker. Only a
/// `connect_timeout` on the caller's `reqwest::Client` reports as `is_connect`;
/// without one, a hung connect runs into the request timeout instead.
/// reqwest's error text carries the URL, never the request body.
fn transport_or_timeout(e: reqwest::Error) -> AiGatewayError {
    if e.is_timeout() && !e.is_connect() {
        AiGatewayError::ExtractionTimeout
    } else {
        AiGatewayError::Transport(e.to_string())
    }
}

fn parse_upstream_status(body: &str) -> Option<u16> {
    serde_json::from_str::<Value>(body)
        .ok()?
        .get("upstream_status")?
        .as_u64()
        .and_then(|s| u16::try_from(s).ok())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::{json, Value};
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::ai_gateway::{AiErrorCode, AiGatewayError, HuggianAiClient, HuggianAiConfig};

    const DOC: &str = "JVBERi0xLjQK-unique-doc-marker";

    fn ai(base_url: String, breaker_threshold: u32) -> HuggianAiClient {
        HuggianAiClient::new(
            reqwest::Client::new(),
            HuggianAiConfig {
                base_url,
                service_token: "extract-token".to_string(),
                timeout: Duration::from_secs(5),
                max_retries: 3, // must be ignored: extraction never retries
                breaker_threshold,
                breaker_cooldown: Duration::from_secs(30),
            },
            4,
        )
    }

    fn schema() -> Value {
        json!({ "type": "object" })
    }

    async fn call(ai: &HuggianAiClient) -> Result<Extraction, AiGatewayError> {
        let schema = schema();
        let attachments = [DocumentAttachment { media_type: "application/pdf", data_b64: DOC }];
        ai.extract_document(&ExtractRequest {
            model: "claude-haiku-5-5",
            system: "sys",
            attachments: &attachments,
            schema: &schema,
            max_tokens: 24_000,
            timeout_ms: Some(110_000),
        })
        .await
    }

    async fn server_with(status: u16, body: Value) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .mount(&server)
            .await;
        server
    }

    #[tokio::test]
    async fn sends_the_wire_request_and_parses_a_billed_success() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/ai/extract-document"))
            .and(header("authorization", "Bearer extract-token"))
            .and(body_partial_json(json!({
                "kind": "claude",
                "model": "claude-haiku-5-5",
                "system": "sys",
                "attachments": [{ "media_type": "application/pdf", "data_b64": DOC }],
                "schema": { "type": "object" },
                "max_tokens": 24000,
                "timeout_ms": 110000
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "ok": true, "data": { "lines": [] }, "stop_reason": "end_turn",
                "model": "claude-haiku-5-5", "usage": { "input_tokens": 1000, "output_tokens": 100 },
                "cost_usd": 0.00015, "request_id": "r-1"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let out = call(&ai(server.uri(), 3)).await.unwrap();
        assert_eq!(out.data, Some(json!({ "lines": [] })));
        assert_eq!(out.stop_reason, StopReason::EndTurn);
        assert_eq!(out.model, "claude-haiku-5-5");
        assert_eq!(out.usage.input_tokens, 1000);
        assert_eq!(out.usage.output_tokens, 100);
        assert_eq!(out.cost_usd, Some(0.00015));
        assert_eq!(out.request_id.as_deref(), Some("r-1"));
    }

    #[tokio::test]
    async fn content_stops_come_back_as_ok_with_no_data() {
        for (wire, expected) in [
            ("refusal", StopReason::Refusal),
            ("max_tokens", StopReason::MaxTokens),
            ("invalid_output", StopReason::InvalidOutput),
        ] {
            let server = server_with(200, json!({
                "ok": true, "data": { "x": 1 }, "stop_reason": wire, "model": "claude-opus-5-5",
                "usage": { "input_tokens": 10, "output_tokens": 5 }, "cost_usd": null, "request_id": "r"
            }))
            .await;
            let out = call(&ai(server.uri(), 3)).await.unwrap();
            assert_eq!(out.stop_reason, expected);
            assert_eq!(out.data, None);
            assert_eq!(out.cost_usd, None);
        }
    }

    #[tokio::test]
    async fn maps_each_refusal_status_to_its_variant() {
        type Check = fn(&AiGatewayError) -> bool;
        let cases: Vec<(u16, Value, Check)> = vec![
            (413, json!({ "ok": false, "error_code": "input_rejected", "reason": "body_too_large" }),
                (|e| matches!(e, AiGatewayError::InputRejected { status: 413, upstream_status: None })) as Check),
            (422, json!({ "ok": false, "error_code": "input_rejected", "upstream_status": 400 }),
                |e| matches!(e, AiGatewayError::InputRejected { status: 422, upstream_status: Some(400) })),
            (503, json!({ "ok": false, "error_code": "daily_cap_reached" }),
                |e| matches!(e, AiGatewayError::DailyCapReached)),
            (503, json!({ "ok": false, "error_code": "busy" }),
                |e| matches!(e, AiGatewayError::Busy)),
            (503, json!({ "ok": false, "error_code": "no_provider" }),
                |e| matches!(e, AiGatewayError::NoProvider)),
            (503, json!({ "ok": false, "error_code": "breaker_open" }),
                |e| matches!(e, AiGatewayError::NoProvider)),
            (503, json!({ "ok": false, "error_code": "provider_misconfigured", "error": "x" }),
                |e| matches!(e, AiGatewayError::ProviderMisconfigured(_))),
            (504, json!({ "ok": false, "error_code": "upstream_timeout" }),
                |e| matches!(e, AiGatewayError::ExtractionTimeout)),
            (429, json!({ "error_code": "rate_limit" }),
                |e| matches!(e, AiGatewayError::RateLimited)),
            (401, json!({ "error_code": "invalid_token", "error": "invalid service token" }),
                |e| matches!(e, AiGatewayError::Upstream { status: 401, code: AiErrorCode::InvalidToken, .. })),
            (401, json!({ "error": "missing bearer token" }),
                |e| matches!(e, AiGatewayError::Upstream { status: 401, code: AiErrorCode::Unknown, .. })),
            (403, json!({ "error_code": "route_not_allowed" }),
                |e| matches!(e, AiGatewayError::Upstream { status: 403, code: AiErrorCode::RouteNotAllowed, .. })),
            (400, json!({ "ok": false, "error_code": "model_not_allowed" }),
                |e| matches!(e, AiGatewayError::Upstream { status: 400, code: AiErrorCode::ModelNotAllowed, .. })),
            (502, json!({ "ok": false, "error_code": "upstream_error" }),
                |e| matches!(e, AiGatewayError::Upstream { status: 502, code: AiErrorCode::UpstreamError, .. })),
        ];
        for (status, body, is_expected) in cases {
            let server = server_with(status, body.clone()).await;
            let err = call(&ai(server.uri(), 3)).await.unwrap_err();
            assert!(is_expected(&err), "{status} {body} mapped to {err:?}");
            assert_eq!(server.received_requests().await.unwrap().len(), 1, "{status}: exactly one request, never retried");
        }
    }

    fn request<'a>(schema: &'a Value, attachments: &'a [DocumentAttachment<'a>]) -> ExtractRequest<'a> {
        ExtractRequest {
            model: "claude-haiku-5-5",
            system: "sys",
            attachments,
            schema,
            max_tokens: 24_000,
            timeout_ms: Some(110_000),
        }
    }

    async fn slow_server() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(3)))
            .mount(&server)
            .await;
        server
    }

    #[test]
    fn extraction_timeout_outlasts_the_gateway_timeout() {
        assert_eq!(extraction_timeout(Duration::from_secs(30), None), Duration::from_secs(130));
        assert_eq!(extraction_timeout(Duration::from_millis(300), Some(10_000)), Duration::from_secs(20));
        assert_eq!(extraction_timeout(Duration::from_secs(200), Some(10_000)), Duration::from_secs(200));
    }

    #[tokio::test]
    async fn a_client_side_timeout_is_an_extraction_timeout() {
        let server = slow_server().await;
        let (schema, attachments) = (schema(), [DocumentAttachment { media_type: "application/pdf", data_b64: DOC }]);
        let err = ai(server.uri(), 3)
            .post_extract_once(&request(&schema, &attachments), Duration::from_millis(300))
            .await
            .unwrap_err();
        assert!(matches!(err, AiGatewayError::ExtractionTimeout), "{err:?}");
    }

    #[tokio::test]
    async fn client_side_timeouts_never_open_the_breaker_or_retry() {
        let server = slow_server().await;
        let client = ai(server.uri(), 3);
        let (schema, attachments) = (schema(), [DocumentAttachment { media_type: "application/pdf", data_b64: DOC }]);
        for _ in 0..5 {
            let err = client
                .extract_guarded(&request(&schema, &attachments), Duration::from_millis(300))
                .await
                .unwrap_err();
            assert!(matches!(err, AiGatewayError::ExtractionTimeout), "{err:?}");
        }
        assert_eq!(server.received_requests().await.unwrap().len(), 5, "one request per call, never retried");
        server.reset().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "text": "ok" })))
            .mount(&server)
            .await;
        assert_eq!(client.complete("hi", None, None, None).await.unwrap(), "ok");
    }

    #[tokio::test]
    async fn an_unreachable_gateway_is_a_transport_failure() {
        // Bind then release a port: nothing listens there, so the connect is refused.
        let addr = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let uri = format!("http://{addr}");
        let err = call(&ai(uri, 3)).await.unwrap_err();
        assert!(matches!(err, AiGatewayError::Transport(_)), "{err:?}");
    }

    #[tokio::test]
    async fn a_connect_timeout_is_a_transport_failure() {
        // Non-routable: the SYN goes unanswered, so the client's connect_timeout
        // fires. That is the gateway being unreachable, not a long document.
        let http = reqwest::Client::builder().connect_timeout(Duration::from_millis(200)).build().unwrap();
        let client = HuggianAiClient::new(
            http,
            HuggianAiConfig {
                base_url: "http://10.255.255.1:9".to_string(),
                service_token: "t".to_string(),
                timeout: Duration::from_secs(5),
                max_retries: 0,
                breaker_threshold: 3,
                breaker_cooldown: Duration::from_secs(30),
            },
            4,
        );
        let (schema, attachments) = (schema(), [DocumentAttachment { media_type: "application/pdf", data_b64: DOC }]);
        let err = client
            .post_extract_once(&request(&schema, &attachments), Duration::from_secs(5))
            .await
            .unwrap_err();
        assert!(matches!(err, AiGatewayError::Transport(_)), "{err:?}");
    }

    #[tokio::test]
    async fn invalid_requests_are_rejected_without_a_request() {
        let server = server_with(200, json!({})).await;
        let client = ai(server.uri(), 3);
        let schema = schema();
        let one = [DocumentAttachment { media_type: "application/pdf", data_b64: DOC }];
        let six = [one[0]; 6];
        let base = request(&schema, &one);
        for req in [
            ExtractRequest { attachments: &[], ..base },
            ExtractRequest { attachments: &six, ..base },
            ExtractRequest { max_tokens: 0, ..base },
            ExtractRequest { max_tokens: 32_769, ..base },
            ExtractRequest { timeout_ms: Some(9_999), ..base },
            ExtractRequest { timeout_ms: Some(120_001), ..base },
        ] {
            let err = client.extract_document(&req).await.unwrap_err();
            assert!(matches!(err, AiGatewayError::InvalidRequest(_)), "{req:?} -> {err:?}");
            assert_eq!(err.code(), AiErrorCode::Unknown);
        }
        assert_eq!(server.received_requests().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn content_outcomes_and_refusals_never_open_the_breaker() {
        for (status, body) in [
            (504u16, json!({ "error_code": "upstream_timeout" })),
            (422, json!({ "error_code": "input_rejected", "upstream_status": 400 })),
            (413, json!({ "error_code": "input_rejected" })),
            (503, json!({ "error_code": "daily_cap_reached" })),
            (503, json!({ "error_code": "busy" })),
            (400, json!({ "error_code": "model_not_allowed" })),
            (401, json!({ "error_code": "invalid_token" })),
            (403, json!({ "error_code": "route_not_allowed" })),
        ] {
            let server = server_with(status, body).await;
            let client = ai(server.uri(), 3);
            for _ in 0..6 {
                let err = call(&client).await.unwrap_err();
                assert!(!matches!(err, AiGatewayError::CircuitOpen { .. }), "{status} opened the breaker");
            }
            assert_eq!(server.received_requests().await.unwrap().len(), 6, "{status}: all calls reach the gateway");
        }
    }

    #[tokio::test]
    async fn provider_outages_open_the_breaker() {
        for (status, body) in [
            (502u16, json!({ "error_code": "upstream_error" })),
            (503, json!({ "error_code": "no_provider" })),
            (503, json!({ "error_code": "breaker_open" })),
            (429, json!({ "error_code": "rate_limit" })),
        ] {
            let server = server_with(status, body).await;
            let client = ai(server.uri(), 3);
            for _ in 0..3 {
                let _ = call(&client).await;
            }
            let err = call(&client).await.unwrap_err();
            assert!(matches!(err, AiGatewayError::CircuitOpen { .. }), "{status}: expected breaker open, got {err:?}");
            assert_eq!(server.received_requests().await.unwrap().len(), 3);
        }
    }

    #[tokio::test]
    async fn errors_and_request_debug_never_carry_document_content() {
        // A misbehaving upstream that echoes the document back must not get it
        // into our error values.
        let echo = format!("echo {DOC}");
        for (status, body) in [
            (400u16, json!({ "error_code": "upstream_error", "error": echo })),
            (401, json!({ "error_code": "upstream_error", "error": echo })),
            (403, json!({ "error_code": "upstream_error", "error": echo })),
            (502, json!({ "error_code": "upstream_error", "error": echo })),
            (503, json!({ "error_code": "provider_misconfigured", "error": echo })),
        ] {
            let server = server_with(status, body).await;
            let err = call(&ai(server.uri(), 3)).await.unwrap_err();
            assert!(!format!("{err} {err:?}").contains(DOC), "{status} leaked the document");
        }
        let schema = json!({ "secret_schema_marker": true });
        let attachments = [DocumentAttachment { media_type: "application/pdf", data_b64: DOC }];
        let req = ExtractRequest {
            model: "claude-haiku-5-5",
            system: "secret-system-marker",
            attachments: &attachments,
            schema: &schema,
            max_tokens: 1,
            timeout_ms: None,
        };
        let dbg = format!("{req:?}");
        assert!(!dbg.contains(DOC) && !dbg.contains("secret-system-marker") && !dbg.contains("secret_schema_marker"), "{dbg}");
    }

    #[tokio::test]
    async fn a_cancelled_call_does_not_open_the_breaker() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(3)))
            .mount(&server)
            .await;
        let client = ai(server.uri(), 3);
        for _ in 0..5 {
            // Caller gives up long before the 5 s client timeout.
            let _ = tokio::time::timeout(Duration::from_millis(100), call(&client)).await;
        }
        // Now a fast, healthy answer must still get through.
        server.reset().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "ok": true, "data": {}, "stop_reason": "end_turn", "model": "m",
                "usage": { "input_tokens": 1, "output_tokens": 1 }, "cost_usd": null
            })))
            .mount(&server)
            .await;
        assert!(call(&client).await.is_ok(), "cancelled extractions must not open the breaker");
    }

    #[test]
    fn extraction_debug_never_prints_data_values() {
        let out = Extraction {
            data: Some(json!({ "cuit": "20-12345678-9" })),
            stop_reason: StopReason::EndTurn,
            model: "m".to_string(),
            usage: crate::ai_gateway::Usage::default(),
            cost_usd: None,
            request_id: Some("r".to_string()),
        };
        let dbg = format!("{out:?}");
        assert!(!dbg.contains("20-12345678-9"), "{dbg}");
        assert!(dbg.contains("bytes") && dbg.contains("EndTurn"), "{dbg}");
    }

    #[tokio::test]
    async fn a_malformed_200_never_quotes_response_values() {
        let server = server_with(200, json!({
            "ok": true, "data": null, "stop_reason": "end_turn", "model": "m",
            "usage": { "input_tokens": "CUIT-20-12345678-9", "output_tokens": 1 }, "cost_usd": null
        }))
        .await;
        let err = call(&ai(server.uri(), 3)).await.unwrap_err();
        assert!(matches!(err, AiGatewayError::Deserialization(_)), "{err:?}");
        assert!(!format!("{err} {err:?}").contains("12345678"), "{err:?}");
    }

    #[tokio::test]
    async fn an_unknown_stop_reason_is_a_deserialization_error() {
        let server = server_with(200, json!({
            "ok": true, "data": null, "stop_reason": "pause_turn", "model": "m",
            "usage": { "input_tokens": 1, "output_tokens": 1 }, "cost_usd": null
        }))
        .await;
        let err = call(&ai(server.uri(), 3)).await.unwrap_err();
        assert!(matches!(err, AiGatewayError::Deserialization(_)), "{err:?}");
    }

    #[tokio::test]
    async fn an_undecodable_200_body_never_opens_the_breaker() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string("<html>gateway page</html>"))
            .mount(&server)
            .await;
        let client = ai(server.uri(), 3);
        for _ in 0..6 {
            let err = call(&client).await.unwrap_err();
            assert!(matches!(err, AiGatewayError::Deserialization(_)), "{err:?}");
        }
        assert_eq!(server.received_requests().await.unwrap().len(), 6);
    }

    #[tokio::test]
    async fn an_abandoned_half_open_probe_hands_the_next_call_the_probe() {
        let server = server_with(503, json!({ "error_code": "no_provider" })).await;
        let client = HuggianAiClient::new(
            reqwest::Client::new(),
            HuggianAiConfig {
                base_url: server.uri(),
                service_token: "t".to_string(),
                timeout: Duration::from_secs(5),
                max_retries: 0,
                breaker_threshold: 3,
                breaker_cooldown: Duration::from_millis(200),
            },
            4,
        );
        for _ in 0..3 {
            let _ = call(&client).await;
        }
        assert!(matches!(call(&client).await.unwrap_err(), AiGatewayError::CircuitOpen { .. }));
        tokio::time::sleep(Duration::from_millis(300)).await;

        // The half-open probe hangs and the caller gives up.
        server.reset().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(3)))
            .mount(&server)
            .await;
        let _ = tokio::time::timeout(Duration::from_millis(100), call(&client)).await;

        // Provider health is still unknown: the next call must be the probe
        // (reaches the network), and when it fails the breaker re-opens at
        // once. A breaker that had wrongly closed would let the call after
        // that through too.
        server.reset().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({ "error_code": "no_provider" })))
            .mount(&server)
            .await;
        assert!(matches!(call(&client).await.unwrap_err(), AiGatewayError::NoProvider));
        assert_eq!(server.received_requests().await.unwrap().len(), 1, "next call became the probe");
        assert!(matches!(call(&client).await.unwrap_err(), AiGatewayError::CircuitOpen { .. }));
    }
}
