# Telemetry Backend

Receives privacy-preserving spam signals from the SpamShield Android app, exposes them for the
retraining pipeline, and serves the trained TFLite classifier through an authenticated API.
Uses the native `mongodb` driver and MongoDB Atlas.

## Privacy rule

Telemetry stores raw text when the final label is **spam**. Ham text is stored only for a
deliberate user correction with explicit text-sharing consent; all other ham telemetry is
hash-only. The server-side prediction endpoint processes submitted text in memory and does not
persist it.

## Setup

Install Node dependencies with `npm install`. Run the complete service in Docker so the Python
TFLite runtime is present; the same Dockerfile is used by Render.

`.env` needs:

```
MONGODB_URI=<your Atlas connection string>
PORT=3000
API_KEY=<shared secret — must match the Android app's local.properties TELEMETRY_API_KEY>
```

Generate a key with `node -e "console.log(require('crypto').randomBytes(24).toString('hex'))"`.

## Run the trained model

The project already contains a trained model in the separate `SMSSpam-Model` repository; the
backend loads that pinned `.tflite` artifact and its metadata at startup, verifies its SHA-256,
then keeps one TFLite interpreter running for requests. Training should remain in the `Model/`
pipeline, not in the production web process. To replace the model, publish a newly gated artifact
at a durable HTTPS URL and set `MODEL_URL`, `MODEL_SHA256`, and `MODEL_META_URL` on Render.

The backend runtime needs Python 3.11 plus `tflite-runtime`. Use the included Dockerfile for both
local container runs and Render deployments. On Render, set the service runtime to **Docker** and
the Dockerfile path to `./Dockerfile`; keep the existing environment variables and add/verify:

- `MONGODB_URI` — MongoDB Atlas connection string.
- `API_KEY` — strong random secret shared with the Android app.
- `PUBLISH_KEY` — separate strong secret for model administration; never reuse `API_KEY`.
- `PORT` — Render supplies this automatically; the service listens on it.

`MODEL_URL`, `MODEL_SHA256`, `MODEL_META_URL`, and `MODEL_CACHE_DIR` are optional unless you are
serving a different model. The defaults point to a pinned, verified model repository commit.
The model and metadata are downloaded on service startup, so Render needs outbound HTTPS access.

### Prediction endpoint

`POST /api/model/predict` requires the usual `x-api-key` header and a JSON body containing a
non-empty `text` value (maximum 5,000 characters). Message text is only processed in memory by
the predictor and is not stored or logged.

Example response:

```json
{
  "label": "spam",
  "isSpam": true,
  "spamProbability": 0.94,
  "threshold": 0.68,
  "model": {
    "featureVersion": 1,
    "numBuckets": 20000,
    "maxFeatures": 256,
    "sha256": "verified-model-checksum"
  }
}
```

The endpoint is limited to 60 requests per 15 minutes per client IP. Existing Android-side
inference and model OTA endpoints continue to work independently.

## Endpoints

Telemetry and prediction endpoints require an `x-api-key` header matching `API_KEY`. Model
administration endpoints use the separate `x-publish-key` header matching `PUBLISH_KEY`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check, no auth |
| POST | `/api/telemetry` | Report one screening result (see body shape below) |
| GET | `/api/telemetry/stats` | Counters: total spam patterns, false-positive signals, device reports |
| GET | `/api/telemetry/export?since=&minDeviceCount=&format=json\|csv` | Pulls spam samples for retraining (consumed by `Model/fetch_telemetry.py`) |
| POST | `/api/model/predict` | Classifies text with the server-side TFLite model |
| GET | `/api/model/latest` | Returns the active model manifest for the Android OTA updater |
| GET | `/api/model/retrain-status` | Checks whether enough trusted reports exist to retrain |
| POST | `/api/model/publish` | Publishes a model manifest (publish key only) |

### POST /api/telemetry body

```jsonc
{
  "spamHash": "sha256 hex of the message body",
  "label": "spam" | "ham",          // the FINAL verdict, after any user correction
  "source": "model" | "user_correction",
  "confidence": 0.94,
  "timestamp": 1732900000000,
  "appVersion": "1.0",
  "messageText": "only present, and only accepted, when label is \"spam\""
}
```

Requests are additionally rate-limited (120 / 15 min per IP) since this endpoint is reachable by
any installed copy of the app.

## Data model

- `spam_signals` — one doc per unique `spamHash` that was ever labeled spam: `text`, `confidence`,
  `deviceCount` (how many devices independently reported it), `firstSeen`, `lastSeen`.
- `false_positive_signals` — hash-only doc per message a user corrected from spam back to ham.
  No text is stored here, ever.
