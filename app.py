import asyncio
import gc
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Any, Dict, List, Optional, Tuple

import httpx
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("llm-privacy-proxy")


@dataclass
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8088"))

    inbound_api_keys: List[str] = field(
        default_factory=lambda: [
            x.strip()
            for x in os.getenv("INBOUND_API_KEYS", "").split(",")
            if x.strip()
        ]
    )

    upstream_base_url: str = os.getenv("UPSTREAM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    upstream_api_key: str = os.getenv("UPSTREAM_API_KEY", "")

    privacy_model_id: str = os.getenv("PRIVACY_MODEL_ID", "openai/privacy-filter")
    device: str = os.getenv("DEVICE", "auto")
    torch_dtype: str = os.getenv("TORCH_DTYPE", "auto")

    filter_output: bool = os.getenv("FILTER_OUTPUT", "true").lower() in ("1", "true", "yes", "on")
    min_entity_score: float = float(os.getenv("MIN_ENTITY_SCORE", "0.50"))
    max_string_chars: int = int(os.getenv("MAX_STRING_CHARS", "200000"))
    model_idle_unload_seconds: int = int(os.getenv("MODEL_IDLE_UNLOAD_SECONDS", "300"))
    model_suffix: str = os.getenv("MODEL_SUFFIX", "-anonym")
    inference_max_pending: int = max(1, int(os.getenv("INFERENCE_MAX_PENDING", "8")))
    inference_batch_size: int = max(1, int(os.getenv("INFERENCE_BATCH_SIZE", "4")))
    inference_batch_max_chars: int = max(1, int(os.getenv("INFERENCE_BATCH_MAX_CHARS", "32768")))
    upstream_max_connections: int = max(1, int(os.getenv("UPSTREAM_MAX_CONNECTIONS", "100")))
    upstream_max_keepalive_connections: int = max(0, int(os.getenv("UPSTREAM_MAX_KEEPALIVE_CONNECTIONS", "20")))

    placeholder_style: str = os.getenv("PLACEHOLDER_STYLE", "typed_index")
    skip_json_keys: set = field(
        default_factory=lambda: {
            x.strip()
            for x in os.getenv(
                "SKIP_JSON_KEYS",
                "model,role,type,stream,temperature,max_tokens,top_p,tools,tool_choice,name,thinking,reasoning,reasoning_effort",
            ).split(",")
            if x.strip()
        }
    )

    metrics_require_auth: bool = os.getenv("METRICS_REQUIRE_AUTH", "true").lower() in ("1", "true", "yes", "on")


settings = Settings()


def suffix_model_id(model_id: str) -> str:
    if not model_id or not settings.model_suffix or model_id.endswith(settings.model_suffix):
        return model_id
    return f"{model_id}{settings.model_suffix}"


def unsuffix_model_id(model_id: str) -> str:
    suffix = settings.model_suffix
    if suffix and model_id.endswith(suffix) and len(model_id) > len(suffix):
        return model_id[: -len(suffix)]
    return model_id


def unsuffix_model_path(full_path: str) -> str:
    parts = full_path.split("/")
    if len(parts) >= 2 and parts[0] == "models":
        parts[1] = unsuffix_model_id(parts[1])
    return "/".join(parts)


def rewrite_request_model_ids(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    out = dict(value)
    model_id = out.get("model")
    if isinstance(model_id, str):
        model_id = model_id.strip()
        if not model_id or model_id == settings.model_suffix:
            raise HTTPException(status_code=400, detail="invalid_model_id")
        out["model"] = unsuffix_model_id(model_id)
    return out


def rewrite_response_model_ids(value: Any, *, models_endpoint: bool = False) -> Any:
    if isinstance(value, list):
        return [rewrite_response_model_ids(item, models_endpoint=models_endpoint) for item in value]

    if not isinstance(value, dict):
        return value

    out = dict(value)
    model_id = out.get("model")
    if isinstance(model_id, str):
        out["model"] = suffix_model_id(model_id)

    if models_endpoint:
        object_id = out.get("id")
        if isinstance(object_id, str) and out.get("object") == "model":
            out["id"] = suffix_model_id(object_id)

        data = out.get("data")
        if isinstance(data, list):
            out["data"] = [rewrite_response_model_ids(item, models_endpoint=True) for item in data]

    return out


class GlobalMetrics:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.requests_total = 0
        self.filtered_requests_total = 0
        self.filtered_tokens_total = 0
        self.filtered_spans_total = 0
        self.filtered_by_label: Dict[str, int] = {}

    async def add(
        self,
        tokens: int,
        spans: int,
        labels: Dict[str, int],
        *,
        count_request: bool = True,
    ) -> None:
        async with self.lock:
            if count_request:
                self.requests_total += 1
                if tokens > 0 or spans > 0:
                    self.filtered_requests_total += 1
            self.filtered_tokens_total += tokens
            self.filtered_spans_total += spans
            for k, v in labels.items():
                self.filtered_by_label[k] = self.filtered_by_label.get(k, 0) + v

    async def prometheus(self) -> str:
        async with self.lock:
            lines = [
                "# HELP privacy_proxy_requests_total Total proxied requests.",
                "# TYPE privacy_proxy_requests_total counter",
                f"privacy_proxy_requests_total {self.requests_total}",
                "# HELP privacy_proxy_filtered_requests_total Requests where at least one token/span was filtered.",
                "# TYPE privacy_proxy_filtered_requests_total counter",
                f"privacy_proxy_filtered_requests_total {self.filtered_requests_total}",
                "# HELP privacy_proxy_filtered_tokens_total Estimated number of model-tokenized tokens filtered.",
                "# TYPE privacy_proxy_filtered_tokens_total counter",
                f"privacy_proxy_filtered_tokens_total {self.filtered_tokens_total}",
                "# HELP privacy_proxy_filtered_spans_total Number of PII spans filtered.",
                "# TYPE privacy_proxy_filtered_spans_total counter",
                f"privacy_proxy_filtered_spans_total {self.filtered_spans_total}",
            ]
            for label, count in sorted(self.filtered_by_label.items()):
                safe = label.replace('"', '\\"')
                lines.append(f'privacy_proxy_filtered_spans_by_label_total{{label="{safe}"}} {count}')
            return "\n".join(lines) + "\n"


metrics = GlobalMetrics()


@dataclass
class RedactionStats:
    tokens: int = 0
    spans: int = 0
    labels: Dict[str, int] = field(default_factory=dict)

    def add(self, label: str, token_count: int) -> None:
        self.tokens += token_count
        self.spans += 1
        self.labels[label] = self.labels.get(label, 0) + 1


class RedactionContext:
    """
    Stable placeholders inside one request.
    Same detected original span + label => same placeholder.
    """

    def __init__(self) -> None:
        self.by_value: Dict[Tuple[str, str], str] = {}
        self.next_index: Dict[str, int] = {}
        self.token_counts: Dict[str, int] = {}

    def placeholder(self, label: str, value: str) -> str:
        label = normalize_label(label)
        key = (label, value)
        if key in self.by_value:
            return self.by_value[key]

        self.next_index[label] = self.next_index.get(label, 0) + 1
        ph = f"[{label.upper()}_{self.next_index[label]}]"
        self.by_value[key] = ph
        return ph


def normalize_label(label: str) -> str:
    label = label or "private"
    label = label.replace("B-", "").replace("I-", "").replace("E-", "").replace("S-", "")
    return label.lower()


class PrivacySanitizer:
    def __init__(self) -> None:
        self.tokenizer = None
        self.classifier = None
        self._last_used_at = 0.0
        self._executor = None
        self._pending = 0
        self._closed = False

    async def _run(self, func, *args):
        # All model access uses one dedicated thread, including loading and unloading.
        # Keep admission occupied until the thread finishes even if its caller disconnects.
        if self._closed or self._pending >= settings.inference_max_pending:
            raise HTTPException(status_code=503, detail="privacy_filter_busy", headers={"Retry-After": "1"})
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="privacy-filter")
        self._pending += 1
        try:
            future = asyncio.get_running_loop().run_in_executor(self._executor, func, *args)
        except BaseException:
            self._pending -= 1
            raise

        def finished(done):
            self._pending -= 1
            # Retrieve exceptions from work whose HTTP caller has been cancelled.
            if not done.cancelled():
                done.exception()

        future.add_done_callback(finished)
        return await asyncio.shield(future)

    async def idle_watchdog(self) -> None:
        while True:
            timeout = settings.model_idle_unload_seconds
            await asyncio.sleep(min(30.0, max(0.1, timeout / 2)) if timeout > 0 else 30.0)
            if self._closed:
                return
            if timeout > 0 and self._executor is not None and self._pending == 0:
                # No suspension between this check and _run's admission. A newly
                # admitted request can follow maintenance in FIFO and reload if needed.
                try:
                    await self._run(self.unload_if_idle)
                except HTTPException as exc:
                    if exc.status_code != 503:
                        log.exception("Privacy model idle maintenance failed; will retry")
                    else:
                        log.debug("Privacy model idle maintenance deferred: worker busy")
                except Exception:
                    log.exception("Privacy model idle maintenance failed; will retry")

    async def aclose(self) -> None:
        self._closed = True
        executor = self._executor
        if executor is not None:
            # FIFO: release the model only after all admitted inferences finish.
            try:
                await asyncio.get_running_loop().run_in_executor(executor, self._unload)
            finally:
                try:
                    await asyncio.to_thread(executor.shutdown, wait=True)
                finally:
                    self._executor = None

    def _touch(self) -> None:
        self._last_used_at = time.monotonic()

    def unload_if_idle(self) -> None:
        if self.classifier is None:
            return

        timeout = settings.model_idle_unload_seconds
        if timeout <= 0:
            return

        idle_for = time.monotonic() - self._last_used_at
        if idle_for < timeout:
            return

        log.info("Unloading privacy model after %.1fs of inactivity", idle_for)
        self._unload()

    def _unload(self) -> None:
        self.classifier = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_loaded(self) -> None:
        if self.classifier is not None:
            return

        log.info("Loading privacy model: %s", settings.privacy_model_id)

        dtype = None
        if settings.torch_dtype == "bf16":
            dtype = torch.bfloat16
        elif settings.torch_dtype == "fp16":
            dtype = torch.float16
        elif settings.torch_dtype == "fp32":
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(settings.privacy_model_id)

        model_kwargs = {}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        if settings.device == "cuda":
            model_kwargs["device_map"] = "cuda"
        elif settings.device == "cpu":
            model_kwargs["device_map"] = "cpu"
        else:
            model_kwargs["device_map"] = "auto"

        model = AutoModelForTokenClassification.from_pretrained(
            settings.privacy_model_id,
            **model_kwargs,
        )
        model.eval()

        self.classifier = pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )

        self.tokenizer = tokenizer
        log.info("Privacy model loaded")
        self._touch()

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return max(1, len(text.split()))

    def _infer(self, texts: List[str]) -> Dict[str, list]:
        if not texts:
            return {}
        self.ensure_loaded()
        batch_size = settings.inference_batch_size if self.tokenizer.pad_token_id is not None else 1
        result = {}
        batch = []
        chars = 0

        def flush():
            # Bound padding costs and keep DataLoader workers out of the model thread.
            with torch.inference_mode():
                predictions = self.classifier(batch, batch_size=len(batch), num_workers=0)
            result.update(zip(batch, predictions))

        try:
            for text in texts:
                if batch and (len(batch) >= batch_size
                              or chars + len(text) > settings.inference_batch_max_chars):
                    flush()
                    batch = []
                    chars = 0
                batch.append(text)
                chars += len(text)
            if batch:
                flush()
        except Exception:
            log.exception("Privacy model inference failed")
            raise HTTPException(status_code=500, detail="privacy_filter_failed") from None
        finally:
            self._touch()
        return result

    @staticmethod
    def _validate_text(text: str) -> None:
        if len(text) > settings.max_string_chars:
            raise HTTPException(status_code=413, detail="privacy_string_too_long")

    async def sanitize_text(self, text: str, ctx: RedactionContext, stats: RedactionStats) -> str:
        self._validate_text(text)
        if not text:
            return text
        return await self._run(self._sanitize_text_sync, text, ctx, stats)

    def _sanitize_text_sync(self, text, ctx, stats, entities=None):
        if entities is None:
            entities = self._infer([text])[text]

        spans = []
        for ent in entities:
            score = float(ent.get("score", 0.0))
            if score < settings.min_entity_score:
                continue

            start = ent.get("start")
            end = ent.get("end")
            label = ent.get("entity_group") or ent.get("entity") or "private"

            if start is None or end is None:
                word = ent.get("word", "")
                if word:
                    idx = text.find(word)
                    if idx >= 0:
                        start, end = idx, idx + len(word)

            if start is None or end is None:
                continue

            start, end = int(start), int(end)
            if start < 0 or end <= start or end > len(text):
                continue

            spans.append((start, end, normalize_label(label), score))

        if not spans:
            return text

        # Merge overlapping spans, keep earlier/larger one.
        spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        merged = []
        for span in spans:
            if not merged or span[0] >= merged[-1][1]:
                merged.append(span)
            else:
                prev = merged[-1]
                if span[1] > prev[1]:
                    merged[-1] = (prev[0], span[1], prev[2], max(prev[3], span[3]))

        out = []
        last = 0
        for start, end, label, _score in merged:
            original = text[start:end]
            replacement = ctx.placeholder(label, original)
            out.append(text[last:start])
            out.append(replacement)
            last = end

            if original not in ctx.token_counts:
                ctx.token_counts[original] = self.count_tokens(original)
            stats.add(label, ctx.token_counts[original])

        out.append(text[last:])
        return "".join(out)

    async def sanitize_payload(self, payload: Any) -> Tuple[Any, RedactionStats]:
        return await self._run(self._sanitize_payload_sync, payload)

    def _sanitize_payload_sync(self, payload: Any) -> Tuple[Any, RedactionStats]:
        # Cache only inside this operation: no PII retained across requests.
        texts = {}

        def collect(value, parent_key=None):
            if parent_key in settings.skip_json_keys:
                return
            if isinstance(value, str):
                self._validate_text(value)
                if value:
                    texts[value] = None
            elif isinstance(value, list):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    collect(item, str(key))

        collect(payload)
        predictions = self._infer(list(texts))
        ctx = RedactionContext()
        stats = RedactionStats()
        sanitized = self._sanitize_any(payload, ctx, stats, None, predictions)
        return sanitized, stats

    def _sanitize_any(
        self,
        value: Any,
        ctx: RedactionContext,
        stats: RedactionStats,
        parent_key: Optional[str],
        predictions: Dict[str, list],
    ) -> Any:
        if parent_key in settings.skip_json_keys:
            return value
        if isinstance(value, str):
            if not value:
                return value
            return self._sanitize_text_sync(value, ctx, stats, predictions[value])
        if isinstance(value, list):
            return [self._sanitize_any(v, ctx, stats, None, predictions) for v in value]
        if isinstance(value, dict):
            return {k: self._sanitize_any(v, ctx, stats, str(k), predictions) for k, v in value.items()}
        return value


sanitizer = PrivacySanitizer()


class NoUpstreamCookies(DefaultCookiePolicy):
    def set_ok(self, cookie, request):
        # The pool is shared by different users, never persist upstream sessions.
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sanitizer
    if sanitizer._closed:
        sanitizer = PrivacySanitizer()
    owner = sanitizer
    limits = httpx.Limits(
        max_connections=settings.upstream_max_connections,
        max_keepalive_connections=min(settings.upstream_max_keepalive_connections, settings.upstream_max_connections),
    )
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=30.0, pool=30.0), limits=limits,
        cookies=CookieJar(policy=NoUpstreamCookies()),
    ) as client:
        app.state.upstream_client = client
        watchdog = asyncio.create_task(owner.idle_watchdog())
        try:
            yield
        finally:
            watchdog.cancel()
            try:
                with suppress(asyncio.CancelledError):
                    await watchdog
            finally:
                try:
                    await owner.aclose()
                finally:
                    app.state.upstream_client = None


app = FastAPI(title="OpenAI Privacy Filter Proxy", version="1.0.0", lifespan=lifespan)


def extract_bearer(req: Request) -> str:
    auth = req.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return ""
    return auth.split(" ", 1)[1].strip()


def require_auth(req: Request, *, metrics_auth: bool = False) -> None:
    if metrics_auth and not settings.metrics_require_auth:
        return

    if not settings.inbound_api_keys:
        return

    token = extract_bearer(req)
    if token not in settings.inbound_api_keys:
        raise HTTPException(status_code=401, detail="invalid_or_missing_api_token")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": settings.privacy_model_id,
        "upstream": settings.upstream_base_url,
        "filter_output": settings.filter_output,
        "model_suffix": settings.model_suffix,
    }


@app.get("/metrics")
async def get_metrics(req: Request) -> PlainTextResponse:
    require_auth(req, metrics_auth=True)
    return PlainTextResponse(await metrics.prometheus(), media_type="text/plain")


def build_upstream_headers(req: Request) -> Dict[str, str]:
    excluded = {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }

    headers = {
        k: v
        for k, v in req.headers.items()
        if k.lower() not in excluded
    }

    if settings.upstream_api_key:
        headers["authorization"] = f"Bearer {settings.upstream_api_key}"

    headers["content-type"] = "application/json"
    return headers


def headers_for_modified_body(resp: Response | httpx.Response) -> Dict[str, str]:
    excluded = {
        "content-length",
        "content-encoding",
        "transfer-encoding",
        "connection",
    }
    return {
        key: value
        for key, value in resp.headers.items()
        if key.lower() not in excluded
    }


def response_models_endpoint(full_path: str) -> bool:
    return full_path == "models" or full_path.startswith("models/")


def add_response_model_suffixes(upstream_resp: Response, full_path: str) -> Response:
    ctype = upstream_resp.headers.get("content-type", "")
    if "application/json" not in ctype:
        return upstream_resp

    try:
        response_payload = json.loads(upstream_resp.body)
    except Exception:
        return upstream_resp

    response_payload = rewrite_response_model_ids(
        response_payload,
        models_endpoint=response_models_endpoint(full_path),
    )
    return JSONResponse(
        content=response_payload,
        status_code=upstream_resp.status_code,
        headers=headers_for_modified_body(upstream_resp),
    )


async def forward_request(
    req: Request,
    full_path: str,
    sanitized_payload: Any,
    stream: bool = False,
) -> Response:
    full_path = unsuffix_model_path(full_path)
    url = f"{settings.upstream_base_url}/{full_path}"

    client = getattr(req.app.state, "upstream_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="upstream_client_not_ready")
    upstream_req = client.build_request(
        method=req.method,
        url=url,
        headers=build_upstream_headers(req),
        params=req.query_params.multi_items(),
        json=sanitized_payload,
    )
    try:
        upstream = await client.send(upstream_req, stream=stream)
    except httpx.PoolTimeout:
        raise HTTPException(status_code=503, detail="upstream_busy", headers={"Retry-After": "1"}) from None
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="upstream_timeout") from None
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="upstream_unavailable") from None

    content_type = upstream.headers.get("content-type", "application/json")
    headers = headers_for_modified_body(upstream)
    if stream:
        async def body():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await asyncio.shield(upstream.aclose())

        return UpstreamStreamingResponse(
            upstream,
            body(),
            status_code=upstream.status_code,
            headers=headers,
            media_type=content_type,
        )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=headers,
        media_type=content_type,
    )


class UpstreamStreamingResponse(StreamingResponse):
    def __init__(self, upstream: httpx.Response, *args, **kwargs):
        self.upstream = upstream
        super().__init__(*args, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Also covers disconnects/send failures before the body iterator starts.
            await asyncio.shield(self.upstream.aclose())


def buffered_completion_response(payload: Any, full_path: str, headers: Dict[str, str], include_usage: bool) -> Response:
    """Emit SSE only after the entire completion has passed output filtering.

    Asking upstream for JSON avoids forwarding PII split across arbitrary SSE deltas,
    including streamed tool arguments. This deliberately delays the first content chunk.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        raise HTTPException(status_code=502, detail="invalid_upstream_completion")
    chat = full_path.rstrip("/") == "chat/completions"
    base = {key: value for key, value in payload.items() if key not in {"choices", "usage"}}
    if chat:
        base["object"] = "chat.completion.chunk"
    chunks = []
    finished = []
    for choice in payload["choices"]:
        if not isinstance(choice, dict):
            raise HTTPException(status_code=502, detail="invalid_upstream_completion")
        if chat:
            message = choice.get("message")
            if not isinstance(message, dict):
                raise HTTPException(status_code=502, detail="invalid_upstream_completion")
            delta = dict(message)
            calls = delta.get("tool_calls")
            if isinstance(calls, list):
                if any(not isinstance(call, dict) for call in calls):
                    raise HTTPException(status_code=502, detail="invalid_upstream_completion")
                delta["tool_calls"] = [{**call, "index": i} for i, call in enumerate(calls)]
            chunk = {**choice, "delta": delta, "finish_reason": None}
            chunk.pop("message", None)
            end = {"index": choice.get("index", 0), "delta": {}, "finish_reason": choice.get("finish_reason")}
        else:
            chunk = {**choice, "finish_reason": None}
            end = {"index": choice.get("index", 0), "text": "", "finish_reason": choice.get("finish_reason")}
        chunks.append(chunk)
        finished.append(end)

    async def events():
        for choices in (chunks, finished):
            yield "data: " + json.dumps({**base, "choices": choices}, ensure_ascii=False) + "\n\n"
        if include_usage and "usage" in payload:
            yield "data: " + json.dumps({**base, "choices": [], "usage": payload["usage"]}, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    headers = {key: value for key, value in headers.items() if key.lower() not in {"content-type", "etag", "content-md5"}}
    headers["x-privacy-stream-mode"] = "buffered"
    return StreamingResponse(events(), headers=headers, media_type="text/event-stream")


@app.api_route("/v1/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_openai(req: Request, full_path: str) -> Response:
    if req.method == "OPTIONS":
        return Response(status_code=204)

    require_auth(req)

    if req.method in ("GET", "DELETE"):
        # Pas de body JSON à filtrer.
        upstream_resp = await forward_request(req, full_path, sanitized_payload=None)
        return add_response_model_suffixes(upstream_resp, full_path)

    try:
        payload = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="expected_json_body")

    # Reject invalid requests before paying for model loading/inference.
    payload = rewrite_request_model_ids(payload)
    wants_stream = bool(isinstance(payload, dict) and payload.get("stream") is True)
    buffered_stream = wants_stream and settings.filter_output
    if buffered_stream and full_path.rstrip("/") not in {"chat/completions", "completions"}:
        raise HTTPException(
            status_code=400,
            detail="filtered_streaming_supported_for_chat_and_text_completions_only",
        )

    start = time.perf_counter()
    sanitized_payload, in_stats = await sanitizer.sanitize_payload(payload)
    input_ms = (time.perf_counter() - start) * 1000
    await metrics.add(in_stats.tokens, in_stats.spans, in_stats.labels)

    if buffered_stream:
        sanitized_payload = {**sanitized_payload, "stream": False}
        sanitized_payload.pop("stream_options", None)

    upstream_start = time.perf_counter()
    upstream_resp = await forward_request(req, full_path, sanitized_payload, stream=wants_stream and not buffered_stream)
    upstream_ms = (time.perf_counter() - upstream_start) * 1000
    upstream_resp.headers["x-privacy-filtered-tokens"] = str(in_stats.tokens)
    upstream_resp.headers["x-privacy-filtered-spans"] = str(in_stats.spans)
    upstream_resp.headers["x-privacy-filter-input-latency-ms"] = str(round(input_ms, 2))
    upstream_resp.headers["x-privacy-filter-latency-ms"] = str(round(input_ms, 2))
    upstream_resp.headers["x-upstream-latency-ms"] = str(round(upstream_ms, 2))
    if wants_stream and not buffered_stream:
        return upstream_resp

    ctype = upstream_resp.headers.get("content-type", "")
    if "application/json" not in ctype:
        if buffered_stream:
            raise HTTPException(status_code=502, detail="expected_json_completion_for_filtered_stream")
        return upstream_resp

    try:
        response_payload = json.loads(upstream_resp.body)
    except (ValueError, UnicodeError):
        if buffered_stream:
            raise HTTPException(status_code=502, detail="invalid_upstream_completion") from None
        return upstream_resp

    response_payload = rewrite_response_model_ids(
        response_payload, models_endpoint=response_models_endpoint(full_path),
    )
    out_stats = RedactionStats()
    output_start = time.perf_counter()
    output_ms = 0.0
    if settings.filter_output:
        response_payload, out_stats = await sanitizer.sanitize_payload(response_payload)
        output_ms = (time.perf_counter() - output_start) * 1000
        await metrics.add(out_stats.tokens, out_stats.spans, out_stats.labels, count_request=False)

    headers = headers_for_modified_body(upstream_resp)
    if settings.filter_output:
        headers["x-privacy-filtered-output-tokens"] = str(out_stats.tokens)
        headers["x-privacy-filtered-output-spans"] = str(out_stats.spans)
        headers["x-privacy-filter-output-latency-ms"] = str(round(output_ms, 2))
        headers["x-privacy-filter-latency-ms"] = str(round(input_ms + output_ms, 2))
    if buffered_stream and 200 <= upstream_resp.status_code < 300:
        stream_options = payload.get("stream_options")
        include_usage = isinstance(stream_options, dict) and stream_options.get("include_usage") is True
        return buffered_completion_response(
            response_payload, full_path, headers, include_usage,
        )
    return JSONResponse(content=response_payload, status_code=upstream_resp.status_code, headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
