cat > README.md <<'EOF'
# OpenAI Privacy Filter Proxy

Proxy OpenAI-compatible `/v1/*` qui filtre les PII avec `openai/privacy-filter` avant transmission vers un backend LLM.

## Fonctionnement

Client OpenAI-compatible  
→ `api-llm-privacy-proxy`  
→ redaction PII : `[PRIVATE_EMAIL_1]`, `[PRIVATE_PERSON_1]`, etc.  
→ upstream OpenAI-compatible

## Installation

```bash
./install.sh
cp .env.example .env
nano .env
source run.sh 0.0.0.0 8088
````

## Variables importantes

```bash
INBOUND_API_KEYS='change-me'
UPSTREAM_BASE_URL='http://127.0.0.1:8000/v1'
UPSTREAM_API_KEY=''
PRIVACY_MODEL_ID='openai/privacy-filter'
DEVICE=auto
TORCH_DTYPE=auto
FILTER_OUTPUT=true
MODEL_SUFFIX='-anonym'
```

## Test

```bash
pytest -q
```

## Appel OpenAI-compatible

```bash
curl -s http://127.0.0.1:8088/v1/chat/completions \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-anonym",
    "messages": [
      {
        "role": "user",
        "content": "My name is Alice Smith and my email is alice@example.com"
      }
    ]
  }' | jq .
```

## Metrics

```bash
curl -s http://127.0.0.1:8088/metrics \
  -H 'Authorization: Bearer change-me'
```

## Notes production

* Par défaut, le proxy filtre les entrées envoyées au LLM et les réponses du LLM (`FILTER_OUTPUT=true`).
* Les modèles exposés au client sont suffixés avec `-anonym` (`MODEL_SUFFIX`) et seul le champ `model` OpenAI de premier niveau est désuffixé avant envoi à l’upstream.
* Les configurations utilisateur comme `thinking` / `reasoning` sont préservées telles quelles par défaut.
* `FILTER_OUTPUT=false` permet de désactiver le filtrage des réponses si la latence est prioritaire.
* Le modèle peut rater des PII, surtout hors anglais ou avec formats métier spécifiques.
* Pour contexte gouvernement / médical / RH / finance, valider sur corpus interne et ajouter éventuellement règles regex métier ou fine-tuning.
  EOF

````

---

## 10. Service systemd exemple

```bash
sudo tee /etc/systemd/system/api-llm-privacy-proxy.service >/dev/null <<'EOF'
[Unit]
Description=OpenAI Privacy Filter Proxy
After=network-online.target
Wants=network-online.target

[Service]
User=ailab
WorkingDirectory=/home/ailab/api-llm-privacy-proxy
Environment=VENV_DIR=/home/ailab/venv/api-llm-privacy-proxy
ExecStart=/bin/bash -lc 'source ./run.sh 0.0.0.0 8088'
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now api-llm-privacy-proxy
sudo journalctl -u api-llm-privacy-proxy -f
```

[1]: https://huggingface.co/openai/privacy-filter "openai/privacy-filter · Hugging Face"
