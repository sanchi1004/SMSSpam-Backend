"""Persistent JSON-lines TFLite worker for the Express API."""

import hashlib
import json
import os
import sys
import tempfile
import unicodedata
from urllib.request import Request, urlopen

# Model is pinned to the model repository commit verified when this worker was added.
MODEL_URL_DEFAULT = (
    "https://raw.githubusercontent.com/sanchi1004/SMSSpam-Model/"
    "02a045c5bf83796c42f5d4f30b42e2e2d7e37018/models/spam_detector.tflite"
)
MODEL_META_URL_DEFAULT = (
    "https://raw.githubusercontent.com/sanchi1004/SMSSpam-Model/"
    "02a045c5bf83796c42f5d4f30b42e2e2d7e37018/models/model_meta.json"
)
MODEL_SHA256_DEFAULT = "4d525bd543f685f68f2775cf8d5d38a5cf5f5b11b0137b89a6466b0f075a3db2"

NUM_BUCKETS = 20000
MAX_FEATURES = 256
FEATURE_VERSION = 1
CHAR_NGRAMS = (3, 4)
USE_WORD_BIGRAMS = True
PAD_ID = 0
FNV_OFFSET = 0x811C9DC5
FNV_PRIME = 0x01000193
MASK32 = 0xFFFFFFFF


def normalize(text):
    """Mirror Model/text_features.py: preserve Unicode letters, marks and numbers."""
    output = []
    previous_was_space = True
    for char in text.lower():
        if unicodedata.category(char)[0] in ("L", "M", "N"):
            output.append(char)
            previous_was_space = False
        elif not previous_was_space:
            output.append(" ")
            previous_was_space = True
    return "".join(output).strip()


def fnv1a(text):
    value = FNV_OFFSET
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * FNV_PRIME) & MASK32
    return value


def bucket(token):
    return 1 + fnv1a(token) % (NUM_BUCKETS - 1)


def featurize(text):
    words = normalize(text).split()
    if not words:
        features = []
    else:
        features = [bucket(word) for word in words]
        if USE_WORD_BIGRAMS:
            features.extend(
                bucket(words[index] + " " + words[index + 1])
                for index in range(len(words) - 1)
            )
        for word in words:
            padded = "<" + word + ">"
            for ngram_size in CHAR_NGRAMS:
                if len(padded) >= ngram_size:
                    features.extend(
                        bucket(padded[index:index + ngram_size])
                        for index in range(len(padded) - ngram_size + 1)
                    )
    features = features[:MAX_FEATURES]
    return features + [PAD_ID] * (MAX_FEATURES - len(features))


def fetch_bytes(url, max_bytes):
    request = Request(url, headers={"User-Agent": "spamshield-backend/1.0"})
    with urlopen(request, timeout=45) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Downloaded model asset exceeds the allowed size")
    return data


def load_model():
    import numpy as np
    from tflite_runtime.interpreter import Interpreter

    model_url = os.environ.get("MODEL_URL", MODEL_URL_DEFAULT)
    meta_url = os.environ.get("MODEL_META_URL", MODEL_META_URL_DEFAULT)
    expected_sha = os.environ.get("MODEL_SHA256", MODEL_SHA256_DEFAULT).lower()
    cache_dir = os.environ.get("MODEL_CACHE_DIR", tempfile.gettempdir())
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "spam_detector.tflite")

    model_bytes = None
    cached_sha = None
    if os.path.isfile(model_path):
        with open(model_path, "rb") as model_file:
            model_bytes = model_file.read(20 * 1024 * 1024 + 1)
        if len(model_bytes) > 20 * 1024 * 1024:
            model_bytes = None
        else:
            cached_sha = hashlib.sha256(model_bytes).hexdigest()

    if cached_sha != expected_sha:
        model_bytes = fetch_bytes(model_url, 20 * 1024 * 1024)

    actual_sha = hashlib.sha256(model_bytes).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("Model SHA-256 does not match MODEL_SHA256")

    if cached_sha != actual_sha:
        temporary_path = model_path + ".download"
        with open(temporary_path, "wb") as model_file:
            model_file.write(model_bytes)
        os.replace(temporary_path, model_path)

    metadata = json.loads(fetch_bytes(meta_url, 64 * 1024).decode("utf-8"))
    if (metadata.get("num_buckets") != NUM_BUCKETS or
            metadata.get("max_features") != MAX_FEATURES or
            metadata.get("feature_version") != FEATURE_VERSION or
            tuple(metadata.get("char_ngrams", [])) != CHAR_NGRAMS or
            metadata.get("use_word_bigrams") is not USE_WORD_BIGRAMS):
        raise ValueError("Model metadata does not match the backend feature contract")
    if not 0.0 < float(metadata.get("threshold", 0)) < 1.0:
        raise ValueError("Model metadata contains an invalid decision threshold")

    interpreter = Interpreter(model_path=model_path, num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if (tuple(input_detail["shape"]) != (1, MAX_FEATURES) or
            input_detail["dtype"] != np.int32 or tuple(output_detail["shape"]) != (1, 1)):
        raise ValueError("TFLite input/output shape or type does not match this backend")

    info = {
        "threshold": float(metadata["threshold"]),
        "featureVersion": FEATURE_VERSION,
        "numBuckets": NUM_BUCKETS,
        "maxFeatures": MAX_FEATURES,
        "sha256": actual_sha,
    }
    return interpreter, input_detail, output_detail, info


def main():
    import numpy as np

    interpreter, input_detail, output_detail, model_info = load_model()
    print(json.dumps({"ready": True, "model": model_info}), flush=True)

    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            request_id = request.get("id")
            text = request.get("text")
            if not isinstance(text, str):
                raise ValueError("text must be a string")

            vector = np.asarray([featurize(text)], dtype=np.int32)
            interpreter.set_tensor(input_detail["index"], vector)
            interpreter.invoke()
            probability = float(interpreter.get_tensor(output_detail["index"])[0][0])
            threshold = model_info["threshold"]
            result = {
                "label": "spam" if probability > threshold else "ham",
                "isSpam": probability > threshold,
                "spamProbability": probability,
                "threshold": threshold,
                "model": model_info,
            }
            print(json.dumps({"id": request_id, "result": result}), flush=True)
        except Exception as error:  # return a request-scoped error without terminating the worker
            print(json.dumps({"id": request.get("id"), "error": str(error)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"ready": False, "error": str(error)}), flush=True)
        sys.exit(1)