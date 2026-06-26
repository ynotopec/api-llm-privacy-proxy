# API LLM Privacy Proxy — Architecture

Proxy OpenAI-compatible (`/v1/*`) qui filtre les PII avec `openai/privacy-filter`
avant transmission vers un backend LLM.

---

## Flux de requête

```mermaid
sequenceDiagram
    autonumber
    actor Client as Client OpenAI<br>(SDK curl HTTP)
    participant Proxy as api-llm-privacy-proxy<br>FastAPI :8088
    participant Sanitizer as Privacy Sanitizer<br>openai/privacy-filter
    participant Rewriter as Model ID Rewriter<br>gpt-4o to gpt-4o-anonym
    participant Forwarder as Forwarder<br>httpx.AsyncClient
    participant Upstream as Upstream LLM<br>Backend LLM compatible /v1

    Client->>Proxy: POST /v1/chat/completions
    Proxy->>Proxy: Auth Bearer Token INBOUND_API_KEYS

    alt auth fail
        Proxy-->>Client: 401 invalid_or_missing_api_token
    else auth ok
        Proxy->>Proxy: Parse JSON body

        Proxy->>Sanitizer: sanitize_text(entrées PII)
        Sanitizer->>Sanitizer: NLP token classification<br>B-PERSON B-EMAIL
        Sanitizer-->>Proxy: text redacté [PRIVATE_PERSON_1]

        Proxy->>Rewriter: unsuffix_model_id gpt-4o-anonym
        Rewriter-->>Proxy: model_id unsuffix

        Proxy->>Forwarder: forward_request sanitized
        Forwarder->>Upstream: POST /v1/chat/completions<br>Bearer UPSTREAM_API_KEY
        Note over Upstream: LLM traite la requête
        Upstream-->>Forwarder: Reponse JSON LLM
        Forwarder-->>Proxy: upstream response

        alt FILTER_OUTPUT = true
            Proxy->>Sanitizer: sanitize_text(sortie LLM)
            Sanitizer->>Sanitizer: NLP détection PII réponse
            Sanitizer-->>Proxy: réponse redactée
        end

        Proxy->>Rewriter: suffix_model_id gpt-4o
        Rewriter-->>Proxy: model_id suffix

        Proxy-->>Client: Reponse redactée + headers
    end
```

## Architecture des composants

```mermaid
graph LR
    subgraph EXTERNAL[Monde Externe]
        C[Client OpenAI<br>SDK curl HTTP]
    end

    subgraph PROXY[api-llm-privacy-proxy FastAPI :8088]
        AUTH[Auth N-Turn<br>Bearer Token<br>INBOUND_API_KEYS]
        SAN1[Sanitiser Entree<br>PII detection]
        MODEL[(openai/privacy-filter<br>Token Classifier)]
        REWRITE_IN[Rewriter Entrée<br>unsuffix model ID]
        FWD[Forwarder<br>httpx.AsyncClient]
        REWRITE_OUT[Rewriter Sortie<br>suffix model ID]
        METRICS[Metrics<br>Prometheus]
        HEALTH[Health<br>GET /health]
    end

    subgraph UPSTREAM[Upstream LLM]
        LLM[Backend LLM<br>v1-compatible]
    end

    C -->|POST /v1/*| AUTH
    AUTH -->|token ok| SAN1
    AUTH -->|token fail| REJECT[401 Unauthorized]

    SAN1 -->|text| MODEL
    MODEL -->|classification| REWRITE_IN
    REWRITE_IN -->|unsuffix model| FWD
    FWD -->|POST upstream| LLM
    LLM -->|response| FWD
    FWD --> REWRITE_OUT
    REWRITE_OUT -->|FILTER_OUTPUT| OUT_SAN[Sanitiser Sortie<br>PII reponse]
    OUT_SAN --> REWRITE_OUT2[Resuffix model ID]
    REWRITE_OUT2 --> C

    REWRITE_IN -.->|stats| METRICS
    OUT_SAN -.->|stats| METRICS
    AUTH -.->|stats| METRICS
    HEALTH -.->|status ok| C

    style EXTERNAL fill:#1e293b,stroke:#94a3b8
    style PROXY fill:#334155,stroke:#cbd5e1
    style UPSTREAM fill:#14532d,stroke:#34d399
    style C fill:#0c3544,stroke:#22d3ee
    style AUTH fill:#881337,stroke:#fb7185
    style METRICS fill:#4c1d95,stroke:#a78bfa
    style LLM fill:#064e3b,stroke:#34d399
    style MODEL fill:#881337,stroke:#fb7185
    style REJECT fill:#1e293b,stroke:#fb7185
```

## Pipeline de redaction PII

```mermaid
graph TD
    A[JSON brut] --> B{Traversée<br>récurseive JSON}
    B -->|clef skip| C[Pas filtre]
    B -->|string| D[Valeur string]
    B -->|dict| E{Enfant dict}
    B -->|list| F{Item liste}

    D --> G{Max chars<br>atteint}
    G -->|oui| C
    G -->|non| H[Tokenisation]

    H --> I[NLP Inference<br>token-classification]
    I --> J[Collecter spans<br>score > threshold]

    J --> K{Spans<br>vides}
    K -->|oui| C
    K -->|non| L[Merged spans<br>fusionner overlaps]

    L --> M[Generer placeholders<br>stable par requête]
    M --> N[Substituer spans<br>PRIVATE_EMAIL_1]
    N --> O[Stats tokens/spans/labels]

    O --> P[JSON redacté]
    C --> P
    E --> B
    F --> B

    style A fill:#1e293b,stroke:#94a3b8
    style I fill:#881337,stroke:#fb7185
    style M fill:#881337,stroke:#fb7185
    style N fill:#881337,stroke:#fb7185
    style P fill:#34d399,stroke:#34d399
```

## Configuration

```mermaid
graph LR
    subgraph ENV[.env - Variables]
        A[INBOUND_API_KEYS<br>change-me<br>Tokens comma-separes]
        B[UPSTREAM_BASE_URL<br>http://127.0.0.1:8000/v1<br>Backend LLM]
        C[UPSTREAM_API_KEY<br>cl-xxx<br>Token upstream]
        D[PRIVACY_MODEL_ID<br>openai/privacy-filter<br>Modele PII]
        E[DEVICE<br>auto cuda cpu<br>Dispositif inference]
        F[TORCH_DTYPE<br>auto fp32 fp16 bf16]
        G[FILTER_OUTPUT<br>true false<br>Filtrer reponses LLM]
        H[MODEL_IDLE_UNLOAD_SECONDS<br>300<br>0 = jamais]
        I[MIN_ENTITY_SCORE<br>0.50<br>Seuil detection PII]
        J[MAX_STRING_CHARS<br>200000<br>Anti-abus par string]
        K[MODEL_SUFFIX<br>-anonym<br>Suffix model id]
        L[PLACEHOLDER_STYLE<br>typed_index]
        M[SKIP_JSON_KEYS<br>model,role,type,...]
    end

    ENV --> SETTINGS[Settings dataclass Python]
    SETTINGS --> RUNTIME[Runtime FastAPI]

    style ENV fill:#78350f,stroke:#fbbf24
    style SETTINGS fill:#064e3b,stroke:#34d399
```

## Observabilite

```mermaid
graph LR
    subgraph METRICS[Metric Prometheus]
        A1[privacy_proxy_requests_total<br>Total requetes proxiees]
        A2[privacy_proxy_filtered_requests_total<br>Requetes avec au moins 1 PII]
        A3[privacy_proxy_filtered_tokens_total<br>Tokens models filtres]
        A4[privacy_proxy_filtered_spans_total<br>PII spans filtres]
        A5[privacy_proxy_filtered_spans_by_label_total<br>PII par type]
    end

    subgraph HEADERS[Headers Reponse]
        B1[x-privacy-filtered-tokens<br>Tokens filtres entree]
        B2[x-privacy-filtered-spans<br>PII spans detectes]
        B3[x-privacy-filter-latency-ms<br>Latence filtrage ms]
    end

    subgraph ENDPOINTS[Endpoints]
        C1[GET /health<br>Status OK + config]
        C2[GET /metrics<br>Prometheus format]
        C3[OPTIONS /v1/*<br>CORS preflight]
    end

    METRICS --> C2
    HEADERS --> C2
    METRICS --> C1
    ENDPOINTS --> C3

    style METRICS fill:#4c1d95,stroke:#a78bfa
    style HEADERS fill:#064e3b,stroke:#34d399
    style ENDPOINTS fill:#1e293b,stroke:#94a3b8
```

## Gestion memoire GPU Idle Unload

```mermaid
sequenceDiagram
    participant S as PrivacySanitizer
    participant TM as Last Used<br>time.monotonic
    participant ID as Idle Check<br>MODEL_IDLE_UNLOAD_SECONDS
    participant GC as Garbage Collection
    participant GPU as GPU VRAM
    participant RL as Recharge<br>lazy load

    S->>TM: _last_used_at = time.monotonic
    Note right of TM: chaque appel sanitize_text

    ID->>ID: idle_for = now - _last_used_at

    alt idle_for > timeout 300s
        ID->>GC: classifier = None<br>tokenizer = None<br>gc.collect<br>torch.cuda.empty_cache
        GC->>GPU: Libere VRAM GPU
        Note right of GPU: Le modele est detruit
    else idle_for <= timeout
        ID->>S: Continue utiliser
    end

    S->>RL: ensure_loaded

    alt model is unloaded
        RL->>RL: Load from HuggingFace<br>AutoModelForTokenClassification<br>+ AutoTokenizer
        RL->>TM: _last_used_at = time.monotonic
        Note right of RL: Recharge differee
    end
```

## Deploiement

```mermaid
graph TD
    subgraph PREP[Preparation]
        A[./install.sh<br>uv venv + pip install]
        B[cp .env.example .env<br>Configurer variables]
    end

    subgraph RUNTIME[Runtime]
        C[source run.sh 0.0.0.0 8088<br>uvicorn app:app --port 8088]
    end

    subgraph PROD[Production]
        D[systemd service]
        E[sudo systemctl enable --now<br>api-llm-privacy-proxy]
        F[journalctl -u<br>api-llm-privacy-proxy -f]
    end

    A --> B
    B --> C
    C -->|developement| C
    C -->|systemd| D
    D --> E
    E --> F

    style PREP fill:#78350f,stroke:#fbbf24
    style RUNTIME fill:#064e3b,stroke:#34d399
    style PROD fill:#4c1d95,stroke:#a78bfa
```

## Exemple de redaction

| Entrée | Sortie |
|---|---|
| `Alice Smith, alice@example.com` | `[PRIVATE_PERSON_1], [PRIVATE_EMAIL_1]` |
| `+33 6 12 34 56 78` | `[PRIVATE_PHONE_NUMBER_1]` |
| `123 Main Street, Paris` | `[PRIVATE_ADDRESS_1]` |
| `1234567890` (SSN US) | `[PRIVATE_US_SSN_1]` |

Les placeholders sont stables : le même span PII détecté dans la même requête obtient toujours le même placeholder.

---

*API LLM Privacy Proxy • FastAPI + HuggingFace Transformers + PyTorch*
