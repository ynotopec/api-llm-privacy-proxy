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
nano .env
source run.sh 0.0.0.0 8088
```

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

## Installation avec pip et tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
```

`requirements.txt` reste la source des dépendances de production ; les outils de test sont dans l’extra `dev`.

## Docker

```bash
docker build -t api-llm-privacy-proxy .
docker run --rm --name privacy-proxy -p 8088:8088 \
  --env-file .env \
  -v privacy-hf-cache:/data/huggingface \
  api-llm-privacy-proxy
```

Créer `.env` depuis `.env.example`, puis régler les tokens et `UPSTREAM_BASE_URL` avant le lancement. Dans le conteneur, `127.0.0.1` désigne le conteneur lui-même : utiliser le nom DNS ou l’adresse accessible du backend.

Le conteneur tourne avec l’UID/GID `10001`. Un répertoire hôte monté pour le cache doit être inscriptible par cet utilisateur. Pour un GPU NVIDIA, ajouter `--gpus all` avec le NVIDIA Container Toolkit installé.

Le port interne est `8088`. Pour le modifier, remplacer la commande :

```bash
docker run --rm --env-file .env -p 8090:8090 api-llm-privacy-proxy \
  uvicorn app:app --host 0.0.0.0 --port 8090
```

Derrière un reverse proxy, configurer `--forwarded-allow-ips` avec les adresses des proxies de confiance.

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

* Par défaut, le proxy filtre les entrées envoyées au LLM et les réponses JSON du LLM (`FILTER_OUTPUT=true`).
* Les modèles exposés au client sont suffixés avec `-anonym` (`MODEL_SUFFIX`) et seul le champ `model` OpenAI de premier niveau est désuffixé avant envoi à l’upstream.
* Les configurations utilisateur comme `thinking` / `reasoning` sont préservées telles quelles par défaut.
* `FILTER_OUTPUT=false` permet de désactiver le filtrage des réponses si la latence est prioritaire.
* **Streaming :** avec `FILTER_OUTPUT=true`, `stream=true` est refusé en HTTP 400 avant l’inférence. Pour conserver le streaming SSE avec filtrage des entrées seulement, configurer `FILTER_OUTPUT=false`. Auparavant, le streaming contournait silencieusement le filtrage de sortie. Le flux amont reste transmis sans réécriture des identifiants de modèle.
* Une chaîne filtrable dépassant `MAX_STRING_CHARS` est refusée en HTTP 413, au lieu d’être transmise sans filtrage. Cette limite en caractères ne remplace pas la fenêtre de contexte du modèle ni une limite HTTP globale au reverse proxy.
* Le modèle peut rater des PII, surtout hors anglais ou avec formats métier spécifiques.
* Pour contexte gouvernement / médical / RH / finance, valider sur corpus interne et ajouter éventuellement règles regex métier ou fine-tuning.

### Charge et mémoire

Le chargement, l’inférence, la tokenisation et le déchargement passent par un seul thread dédié par worker HTTP. Les routes HTTP et les flux actifs restent disponibles pendant le travail du modèle. `INFERENCE_MAX_PENDING=8` borne le nombre d’opérations actives et en attente ; la saturation renvoie HTTP 503 avec `Retry-After: 1`. Une déconnexion ne libère cette place qu’après la fin du travail déjà soumis.

Les chaînes identiques ne sont inférées qu’une fois par opération de filtrage. Les textes distincts sont regroupés par lots (`INFERENCE_BATCH_SIZE=4`, `INFERENCE_BATCH_MAX_CHARS=32768`) ; les compteurs conservent chaque occurrence et les placeholders leur ordre initial. Aucun cache de textes ne persiste entre requêtes. Un tokenizer sans token de padding utilise automatiquement des lots d’une seule chaîne. Réduire la taille des lots à 1 si la mémoire disponible est limitée ; le budget en caractères ne garantit pas une consommation GPU fixe.

Une tâche de surveillance décharge réellement le modèle après `MODEL_IDLE_UNLOAD_SECONDS` d’inactivité, avec une vérification au plus toutes les 30 secondes et sans interrompre l’inférence. La prochaine utilisation recharge le modèle. Choisir `MODEL_IDLE_UNLOAD_SECONDS=0` pour le garder chargé et éviter ces démarrages à froid. Le premier chargement reste différé jusqu’à la première chaîne à filtrer.

Un client HTTP partagé réutilise les connexions amont pendant la vie du worker (`UPSTREAM_MAX_CONNECTIONS=100`, `UPSTREAM_MAX_KEEPALIVE_CONNECTIONS=20`). Les cookies amont ne sont pas mémorisés entre utilisateurs ; un cookie fourni explicitement par le client est toujours transmis. Les réponses streaming libèrent leur connexion à la fin du flux ou à sa fermeture.

Chaque worker Uvicorn possède son propre modèle, sa file et ses métriques : augmenter `--workers` multiplie aussi la consommation RAM/VRAM. Commencer avec un worker et mesurer sur le matériel cible.

### Mesures de latence

* `x-privacy-filter-input-latency-ms` : filtrage d’entrée, incluant son attente dans la file et le chargement éventuel.
* `x-privacy-filter-output-latency-ms` : filtrage de sortie JSON lorsqu’il est activé.
* `x-privacy-filter-latency-ms` : somme des filtrages d’entrée et de sortie, **sans le temps du LLM amont** (correction du comportement précédent).
* `x-upstream-latency-ms` : appel amont ; en streaming, mesure seulement jusqu’à la réception des en-têtes.

Les tests incluent la concurrence, les déconnexions, le cycle de vie, les lots et un petit modèle Transformers local sans téléchargement. Ils ne constituent pas un benchmark du modèle `openai/privacy-filter` sur GPU.

---

## Service systemd exemple

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
