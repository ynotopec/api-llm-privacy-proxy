import os
import gc
import json
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import httpx
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask
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

    placeholder_style: str = os.getenv("PLACEHOLDER_STYLE", "typed_index")
    skip_json_keys: set = field(
        default_factory=lambda: {
            x.strip()
            for x in os.getenv(
                "SKIP_JSON_KEYS",
                "model,role,type,stream,temperature,max_tokens,top_p,tools,tool_choice,name",
            ).split(",")
            if x.strip()
        }
    )

    metrics_require_auth: bool = os.getenv("METRICS_REQUIRE_AUTH", "true").lower() in ("1", "true", "yes", "on")


settings = Settings()


def suffix_model_id(model_id: str) -> str:
    if not settings.model_suffix or model_id.endswith(settings.model_suffix):
        return model_id
    return f"{model_id}{settings.model_suffix}"


def unsuffix_model_id(model_id: str) -> str:
    if settings.model_suffix and model_id.endswith(settings.model_suffix):
        return model_id[: -len(settings.model_suffix)]
    return model_id


def unsuffix_model_path(full_path: str) -> str:
    parts = full_path.split("/")
    if len(parts) >= 2 and parts[0] == "models":
        parts[1] = unsuffix_model_id(parts[1])
    return "/".join(parts)


def rewrite_request_model_ids(value: Any) -> Any:
    if isinstance(value, list):
        return [rewrite_request_model_ids(item) for item in value]

    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "model" and isinstance(item, str):
                out[key] = unsuffix_model_id(item)
            else:
                out[key] = rewrite_request_model_ids(item)
        return out

    return value


def rewrite_response_model_ids(value: Any, *, models_endpoint: bool = False) -> Any:
    if isinstance(value, list):
        return [rewrite_response_model_ids(item, models_endpoint=models_endpoint) for item in value]

    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(item, str) and (key == "model" or (models_endpoint and key == "id")):
                out[key] = suffix_model_id(item)
            else:
                out[key] = rewrite_response_model_ids(item, models_endpoint=models_endpoint)
        return out

    return value


class GlobalMetrics:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.requests_total = 0
        self.filtered_requests_total = 0
        self.filtered_tokens_total = 0
        self.filtered_spans_total = 0
        self.filtered_by_label: Dict[str, int] = {}

    async def add(self, tokens: int, spans: int, labels: Dict[str, int]) -> None:
        async with self.lock:
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
        self._load_lock = asyncio.Lock()

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
        self.classifier = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def ensure_loaded(self) -> None:
        self.unload_if_idle()
        if self.classifier is not None:
            return

        async with self._load_lock:
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

            self.tokenizer = AutoTokenizer.from_pretrained(settings.privacy_model_id)

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
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
            )

            log.info("Privacy model loaded")
            self._touch()

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return max(1, len(text.split()))

    async def sanitize_text(self, text: str, ctx: RedactionContext, stats: RedactionStats) -> str:
        if not text or len(text) > settings.max_string_chars:
            return text

        await self.ensure_loaded()

        self._touch()

        try:
            entities = self.classifier(text)
        except Exception as e:
            log.exception("Privacy model inference failed")
            raise HTTPException(status_code=500, detail=f"privacy_filter_failed: {e}")

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

            stats.add(label, self.count_tokens(original))

        out.append(text[last:])
        return "".join(out)

    async def sanitize_payload(self, payload: Any) -> Tuple[Any, RedactionStats]:
        ctx = RedactionContext()
        stats = RedactionStats()
        sanitized = await self._sanitize_any(payload, ctx, stats, parent_key=None)
        return sanitized, stats

    async def _sanitize_any(
        self,
        value: Any,
        ctx: RedactionContext,
        stats: RedactionStats,
        parent_key: Optional[str],
    ) -> Any:
        if isinstance(value, str):
            if parent_key in settings.skip_json_keys:
                return value
            return await self.sanitize_text(value, ctx, stats)

        if isinstance(value, list):
            return [await self._sanitize_any(v, ctx, stats, parent_key=None) for v in value]

        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                out[k] = await self._sanitize_any(v, ctx, stats, parent_key=str(k))
            return out

        return value


sanitizer = PrivacySanitizer()

app = FastAPI(title="OpenAI Privacy Filter Proxy", version="1.0.0")


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


def headers_for_modified_body(resp: Response) -> Dict[str, str]:
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

    timeout = httpx.Timeout(600.0, connect=30.0)

    if stream:
        client = httpx.AsyncClient(timeout=timeout)
        upstream_req = client.build_request(
            method=req.method,
            url=url,
            headers=build_upstream_headers(req),
            params=dict(req.query_params),
            json=sanitized_payload,
        )
        upstream_stream = await client.send(upstream_req, stream=True)
        async def close_upstream() -> None:
            await upstream_stream.aclose()
            await client.aclose()

        content_type = upstream_stream.headers.get("content-type", "application/json")
        headers = {
            k: v
            for k, v in upstream_stream.headers.items()
            if k.lower() not in {"content-length", "connection"}
        }
        return StreamingResponse(
            upstream_stream.aiter_bytes(),
            status_code=upstream_stream.status_code,
            headers=headers,
            media_type=content_type,
            background=BackgroundTask(close_upstream),
        )

    async with httpx.AsyncClient(timeout=timeout) as client:
        upstream = await client.request(
            method=req.method,
            url=url,
            headers=build_upstream_headers(req),
            params=dict(req.query_params),
            json=sanitized_payload,
        )

        content_type = upstream.headers.get("content-type", "application/json")

        excluded_resp_headers = {
            "content-length",
            "content-encoding",
            "transfer-encoding",
            "connection",
        }

        headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in excluded_resp_headers
        }

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=headers,
            media_type=content_type,
        )


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

    start = time.perf_counter()

    sanitized_payload, in_stats = await sanitizer.sanitize_payload(payload)
    sanitized_payload = rewrite_request_model_ids(sanitized_payload)
    await metrics.add(in_stats.tokens, in_stats.spans, in_stats.labels)

    wants_stream = bool(isinstance(payload, dict) and payload.get("stream") is True)
    if wants_stream:
        return await forward_request(req, full_path, sanitized_payload, stream=True)

    upstream_resp = await forward_request(req, full_path, sanitized_payload, stream=False)

    # Ajout headers utiles.
    upstream_resp.headers["x-privacy-filtered-tokens"] = str(in_stats.tokens)
    upstream_resp.headers["x-privacy-filtered-spans"] = str(in_stats.spans)
    upstream_resp.headers["x-privacy-filter-latency-ms"] = str(round((time.perf_counter() - start) * 1000, 2))

    rewritten_resp = add_response_model_suffixes(upstream_resp, full_path)
    ctype = rewritten_resp.headers.get("content-type", "")
    if "application/json" not in ctype:
        return rewritten_resp

    try:
        response_payload = json.loads(rewritten_resp.body)
    except Exception:
        return rewritten_resp

    out_stats = RedactionStats()
    if settings.filter_output:
        response_payload, out_stats = await sanitizer.sanitize_payload(response_payload)
        await metrics.add(out_stats.tokens, out_stats.spans, out_stats.labels)

    final = JSONResponse(
        content=response_payload,
        status_code=rewritten_resp.status_code,
        headers=headers_for_modified_body(rewritten_resp),
    )
    if settings.filter_output:
        final.headers["x-privacy-filtered-output-tokens"] = str(out_stats.tokens)
        final.headers["x-privacy-filtered-output-spans"] = str(out_stats.spans)
    return final


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
