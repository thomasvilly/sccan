"""
ocr_service.py — Vision LLM extraction for fitness forms.
Supports two backends, selected via config.toml [ocr] backend setting:
  - "openrouter" : Cloud API calls via OpenRouter (GPT-4o, Claude, Gemini, etc.)
  - "ollama"     : Local CPU inference via Ollama (Qwen2.5-VL, Llama 3.2 Vision, Moondream)

Functional style. No try/except. Logs everything.
"""

import asyncio
import base64
import json
import logging
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# SCHEMAS FOR VISION MODELS
# ---------------------------------------------------------------------------

CARDIO_SCHEMA = {
    "type": "object",
    "properties": {
        "cccare_id": {"type": ["string", "null"]},
        "target_hr": {"type": ["string", "null"]},
        "target_rpe": {"type": ["string", "null"]},
        "equipment_settings": {
            "type": "object",
            "properties": {
                "nustep_arms": {"type": ["string", "null"]},
                "nustep_seat": {"type": ["string", "null"]},
                "leg_stab": {"type": ["string", "null"]},
                "recumbent_bike_seat": {"type": ["string", "null"]},
                "upright_bike_seat": {"type": ["string", "null"]},
            },
        },
        "sessions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": ["string", "null"]},
                    "activity": {"type": ["string", "null"]},
                    "time_minutes": {"type": ["integer", "null"]},
                    "work_rate": {"type": ["string", "null"]},
                    "heart_rate_range": {"type": ["string", "null"]},
                    "rpe": {"type": ["integer", "null"]},
                    "comments": {"type": ["string", "null"]},
                    "watch": {"type": ["string", "null"]},
                    "kinesiologist_signed": {"type": "boolean"},
                },
            },
        },
    },
    "required": ["cccare_id", "sessions"],
}

STRENGTH_SCHEMA = {
    "type": "object",
    "properties": {
        "cccare_id": {"type": ["string", "null"]},
        "year": {"type": ["integer", "null"]},
        "sessions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": ["string", "null"]},
                    "exercises": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "exercise_name": {"type": "string"},
                                "sets": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "reps": {"type": ["integer", "null"]},
                                            "weight": {"type": ["integer", "null"]},
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "stretches_completed": {"type": "array", "items": {"type": "string"}},
                    "kinesiologist_signed": {"type": "boolean"},
                },
            },
        },
    },
    "required": ["cccare_id", "sessions"],
}

MERGED_SCHEMA = {
    "type": "object",
    "properties": {
        "cccare_id": {"type": "string", "description": "Final chosen CCCARE ID"},
        "id_mismatch": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "cardio_data": {"type": "object"},
        "strength_data": {"type": "object"},
        "notes": {"type": ["string", "null"]},
    },
    "required": ["cccare_id", "id_mismatch", "confidence", "cardio_data", "strength_data"],
}

# Models known to support Ollama's format="json" reliably
OLLAMA_JSON_CAPABLE_MODELS = {
    "llama3.2-vision",
    "llama3.2-vision:11b",
    "qwen3-vl:2b",
}


# ---------------------------------------------------------------------------
# IMAGE ENCODING
# ---------------------------------------------------------------------------


def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# PROMPT BUILDERS
# ---------------------------------------------------------------------------

# The JSON instruction suffix appended when the model doesn't support format=json
_JSON_SUFFIX = (
    "\n\nIMPORTANT: Respond with ONLY valid JSON. "
    "No markdown fences, no explanation, no preamble. Just the raw JSON object."
)


def build_cardio_prompt(examples, *, force_json_instruction=False):
    prompt = """Extract all data from this CARDIO RECORDING LOG fitness form.

The form is a landscape page (may appear rotated in the image). It contains:

HEADER AREA (top of form):
- Title: "CARDIO RECORDING LOG"
- CCCARE ID: handwritten ID field (may be blank — return null if blank)
- Equipment Settings: NuStep Arms, Seat, Leg Stab., Recumbent Bike Seat, Upright Bike Seat
- Target HR (bpm) and Target RPE (e.g. "11-13")
- Year: 2026

SESSION TABLE — each row is one workout session with these columns:
- Date (e.g. "Jan 5", "Jan 7")
- Activity (NS = NuStep, RB = Recumbent Bike, UB = Upright Bike, TM = Treadmill, E = Elliptical, ROW, LAPS)
- Time (total min) — integer
- Work Rate / Speed/Elevation (e.g. "4.0 - 1%", "4 @ 50", "3 @ 75")
- Heart Rate Range: low to high (e.g. "87-105")
- RPE (6-20 scale) — integer
- Comments (handwritten notes, e.g. "Too hard", "Tried hard, waited to do 20")
- Watch # — the watch number used
- Kinesiologist signature — set kinesiologist_signed to true if a signature/initials are present in that row

IMPORTANT READING NOTES:
- The form may be rotated 90 degrees — read it in landscape orientation
- CCCARE ID may be blank on this page — return null, do NOT invent one
- Work Rate field format varies by activity: treadmill uses "speed - incline%", NuStep uses "level @ resistance", bike uses "level @ watts"
- Read ALL rows that have any handwritten data
- Comments field is often blank (null), only include if text is actually written

"""
    for i, ex in enumerate(examples, 1):
        prompt += f"\nExample {i}:\n{ex['image_description']}\nCorrect extraction:\n{json.dumps(ex['expected_json'], indent=2)}\n"
    prompt += "\nNow extract from this new form image. Return JSON matching the schema exactly. Empty/blank fields should be null.\n"
    if force_json_instruction:
        prompt += _JSON_SUFFIX
    return prompt


def build_strength_prompt(examples, *, force_json_instruction=False):
    prompt = """Extract all data from this STRENGTHENING EXERCISES fitness form.

The form is a landscape page (may appear rotated in the image). It contains:

HEADER AREA:
- CCCARE ID: handwritten (e.g. "11-59.VEL")
- Year: 2026

FORM LAYOUT — the table has EXERCISES as rows and DATES as columns:
- Each DATE column represents one session day (e.g. "Jan 5", "Jan 7", etc.)
- Under each date, the client writes their reps and weight for each exercise
- Each exercise row has sub-columns: Reps and Wt (weight in lbs)
- If weight is 0 or blank, the exercise is bodyweight — return weight as 0
- If both reps and weight are blank for an exercise on a given date, return empty sets array []

EXERCISES (rows, in order on the form):
1. Thoracic rotation (standing against wall)
2. Dynamic hip mobility lateral lunge
3. Squats (grip feet, spread floor)
4. Chest press on bench — has TWO set rows (Wt/Reps twice)
5. Bent knee hip raise - heels on step (risers) or heels on glute bench — has TWO set rows
6. Vertical traction machine (seat #) — has Wt and Reps sub-columns
7. Airplane exercise
8. Front bridge with forearms on wedge (#) - progress to floor — has Sec (seconds) instead of Reps

STRETCHES (bottom of form, per session):
- Checkmarks for: 1. Quad, 2. Hamstring, 3. Glute, 4. Hip flexor, 5. Calf, 6. Chest, 7. Upper back
- Only include stretches that have a visible checkmark for that date
- If no stretches are checked for a date, return empty array []

KINESIOLOGIST SIGNATURE: at the bottom of each date column, check if initials/signature present

IMPORTANT READING NOTES:
- The form may be rotated 90 degrees — read in landscape orientation
- Each DATE is a separate session object in the output
- Exercises with no data for a date still appear but with empty sets: []
- Use the full exercise name as printed on the form for exercise_name

"""
    for i, ex in enumerate(examples, 1):
        prompt += f"\nExample {i}:\n{ex['image_description']}\nCorrect extraction:\n{json.dumps(ex['expected_json'], indent=2)}\n"
    prompt += "\nNow extract from this new form image. Return JSON matching the schema exactly.\n"
    if force_json_instruction:
        prompt += _JSON_SUFFIX
    return prompt


def build_reconciliation_prompt(cardio_data, strength_data, *, force_json_instruction=False):
    prompt = f"""You are reconciling data from two pages of the same fitness form (one cardio, one strength).

Cardio page extracted data: {json.dumps(cardio_data, indent=2)}
Strength page extracted data: {json.dumps(strength_data, indent=2)}

CCCARE IDs found:
- Cardio page: {cardio_data.get('cccare_id')}
- Strength page: {strength_data.get('cccare_id')}

RULES:
1. If one page has a CCCARE ID and the other is null/blank, use the non-null ID
2. If both have IDs but they differ, examine both images and pick the clearer one
3. If both are null, set cccare_id to null and confidence to "low"
4. Set id_mismatch to true only if both pages had non-null IDs that differ

Return merged JSON with:
- cccare_id: the single correct ID (or null if neither page had one)
- id_mismatch: boolean
- confidence: "high", "medium", or "low"
- cardio_data: the full cardio extraction as-is
- strength_data: the full strength extraction as-is
- notes: any reconciliation notes (e.g. "Cardio ID was blank, used strength ID") or null
"""
    if force_json_instruction:
        prompt += _JSON_SUFFIX
    return prompt


# ---------------------------------------------------------------------------
# RESPONSE PARSING
# ---------------------------------------------------------------------------


def parse_json_response(text):
    """
    Parse JSON from a model response, stripping markdown fences if present.
    """
    cleaned = text.strip()
    # Strip ```json ... ``` fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Drop first line (```json) and last line (```)
        inner_lines = []
        started = False
        for line in lines:
            if not started and line.strip().startswith("```"):
                started = True
                continue
            if started and line.strip() == "```":
                break
            if started:
                inner_lines.append(line)
        cleaned = "\n".join(inner_lines).strip()
    logging.debug(f"Parsing JSON response ({len(cleaned)} chars)")
    return json.loads(cleaned)


# ===========================================================================
#  BACKEND: OPENROUTER (Cloud API)
# ===========================================================================


def _get_api_key(config):
    key = os.getenv(config["api"]["openrouter_api_key_env"], "")
    if not key:
        logging.error("OPENROUTER_API_KEY not set!")
    return key


def _api_headers(config):
    return {
        "Authorization": f"Bearer {_get_api_key(config)}",
        "HTTP-Referer": config["api"]["http_referer"],
        "X-Title": config["api"]["app_title"],
        "Content-Type": "application/json",
    }


async def _openrouter_extract_page(client, image_b64, prompt, schema, page_type, model, config):
    """Send a single page to OpenRouter for structured extraction."""
    logging.info(f"[openrouter] Extracting {page_type} with model={model}")
    t0 = time.time()

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": f"{page_type}_page", "schema": schema, "strict": True},
        },
    }

    response = await client.post(
        f"{config['api']['openrouter_base_url']}/chat/completions",
        json=payload,
        headers=_api_headers(config),
        timeout=config["api"]["api_timeout_seconds"],
    )
    response.raise_for_status()
    result = response.json()

    elapsed = time.time() - t0
    logging.info(f"[openrouter] {page_type} done in {elapsed:.2f}s")
    logging.debug(f"[openrouter] Raw response ({page_type}): {result}")

    return json.loads(result["choices"][0]["message"]["content"])


async def _openrouter_reconcile(cardio_data, strength_data, cardio_b64, strength_b64, config):
    """Reconcile via OpenRouter."""
    model = config["models"]["reconciliation_model"]
    logging.info(f"[openrouter] Reconciling with model={model}")
    t0 = time.time()

    prompt = build_reconciliation_prompt(cardio_data, strength_data)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cardio_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{strength_b64}"}},
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "merged_form", "schema": MERGED_SCHEMA, "strict": True},
        },
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{config['api']['openrouter_base_url']}/chat/completions",
            json=payload,
            headers=_api_headers(config),
            timeout=config["api"]["api_timeout_seconds"],
        )
    response.raise_for_status()
    result = response.json()

    elapsed = time.time() - t0
    logging.info(f"[openrouter] Reconciliation done in {elapsed:.2f}s")
    return json.loads(result["choices"][0]["message"]["content"])


async def openrouter_pipeline(cardio_path, strength_path, config, examples):
    """Full OCR pipeline using OpenRouter cloud API."""
    logging.info(f"[openrouter] Pipeline starting: cardio={cardio_path} strength={strength_path}")

    cardio_b64 = encode_image_base64(cardio_path)
    strength_b64 = encode_image_base64(strength_path)

    cardio_prompt = build_cardio_prompt(examples.get("cardio_examples", []))
    strength_prompt = build_strength_prompt(examples.get("strength_examples", []))

    async with httpx.AsyncClient() as client:
        cardio_task = _openrouter_extract_page(
            client, cardio_b64, cardio_prompt, CARDIO_SCHEMA, "cardio",
            config["models"]["cardio_extraction_model"], config,
        )
        strength_task = _openrouter_extract_page(
            client, strength_b64, strength_prompt, STRENGTH_SCHEMA, "strength",
            config["models"]["strength_extraction_model"], config,
        )
        cardio_data, strength_data = await asyncio.gather(cardio_task, strength_task)

    merged = await _openrouter_reconcile(cardio_data, strength_data, cardio_b64, strength_b64, config)
    return merged


# ===========================================================================
#  BACKEND: OLLAMA (Local CPU inference)
# ===========================================================================


def _ollama_model_supports_json(model_name):
    """Check if the Ollama model is known to support format='json'."""
    # Normalize: strip tag suffixes for matching
    base = model_name.split(":")[0].lower()
    full = model_name.lower()
    return base in OLLAMA_JSON_CAPABLE_MODELS or full in OLLAMA_JSON_CAPABLE_MODELS


def _ollama_extract_page_sync(image_path, prompt, page_type, config):
    """
    Extract a single page using the Ollama Python library (synchronous).
    Ollama's Python client is sync-only, so we call this from threads.
    """
    import ollama

    model = config["ollama"]["extraction_model"]
    host = config["ollama"].get("host", "http://localhost:11434")
    logging.info(f"[ollama] Extracting {page_type} with model={model} at {host}")
    t0 = time.time()

    client = ollama.Client(host=host)

    use_json_format = _ollama_model_supports_json(model)
    if not use_json_format:
        logging.info(f"[ollama] Model {model} not in JSON-capable list, using prompt-based JSON")

    # Build the message with image
    # Ollama vision models accept images as file paths or raw bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    options = {}
    ollama_timeout = config["ollama"].get("timeout_seconds", 120)
    # num_ctx controls context window; vision models need headroom for image tokens
    if config["ollama"].get("num_ctx"):
        options["num_ctx"] = config["ollama"]["num_ctx"]

    kwargs = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_bytes],
            }
        ],
        "options": options,
    }

    if use_json_format:
        kwargs["format"] = "json"

    response = client.chat(**kwargs)

    elapsed = time.time() - t0
    raw_text = response["message"]["content"]
    logging.info(f"[ollama] {page_type} done in {elapsed:.2f}s ({len(raw_text)} chars)")
    logging.debug(f"[ollama] Raw response ({page_type}): {raw_text[:500]}")

    return parse_json_response(raw_text)


def _ollama_reconcile_sync(cardio_data, strength_data, cardio_path, strength_path, config):
    """Reconcile both pages using Ollama."""
    import ollama

    model = config["ollama"]["reconciliation_model"]
    host = config["ollama"].get("host", "http://localhost:11434")
    logging.info(f"[ollama] Reconciling with model={model} at {host}")
    t0 = time.time()

    client = ollama.Client(host=host)
    use_json_format = _ollama_model_supports_json(model)

    prompt = build_reconciliation_prompt(
        cardio_data, strength_data,
        force_json_instruction=not use_json_format,
    )

    with open(cardio_path, "rb") as f:
        cardio_bytes = f.read()
    with open(strength_path, "rb") as f:
        strength_bytes = f.read()

    options = {}
    if config["ollama"].get("num_ctx"):
        options["num_ctx"] = config["ollama"]["num_ctx"]

    kwargs = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [cardio_bytes, strength_bytes],
            }
        ],
        "options": options,
    }

    if use_json_format:
        kwargs["format"] = "json"

    response = client.chat(**kwargs)

    elapsed = time.time() - t0
    raw_text = response["message"]["content"]
    logging.info(f"[ollama] Reconciliation done in {elapsed:.2f}s")
    logging.debug(f"[ollama] Reconciliation raw: {raw_text[:500]}")

    return parse_json_response(raw_text)


async def ollama_pipeline(cardio_path, strength_path, config, examples):
    """
    Full OCR pipeline using local Ollama.
    Ollama's Python client is synchronous, so we run extractions in threads.
    Both pages are extracted in parallel threads, then reconciled.
    """
    logging.info(f"[ollama] Pipeline starting: cardio={cardio_path} strength={strength_path}")

    use_json = _ollama_model_supports_json(config["ollama"]["extraction_model"])
    force_json = not use_json

    cardio_prompt = build_cardio_prompt(
        examples.get("cardio_examples", []),
        force_json_instruction=force_json,
    )
    strength_prompt = build_strength_prompt(
        examples.get("strength_examples", []),
        force_json_instruction=force_json,
    )

    loop = asyncio.get_event_loop()

    # Run both extractions in parallel threads (Ollama client is sync/blocking)
    cardio_future = loop.run_in_executor(
        None, _ollama_extract_page_sync, cardio_path, cardio_prompt, "cardio", config
    )
    strength_future = loop.run_in_executor(
        None, _ollama_extract_page_sync, strength_path, strength_prompt, "strength", config
    )

    cardio_data, strength_data = await asyncio.gather(cardio_future, strength_future)

    # Reconciliation (sequential — needs results from both)
    merged = await loop.run_in_executor(
        None, _ollama_reconcile_sync, cardio_data, strength_data, cardio_path, strength_path, config
    )

    return merged


# ===========================================================================
#  UNIFIED PIPELINE ENTRY POINT
# ===========================================================================


async def process_form_ocr(cardio_path, strength_path, config, examples):
    """
    Run the full OCR pipeline using whichever backend is configured.
    Saves the final merged JSON to disk and returns it.

    Backend is selected by config["ocr"]["backend"]: "openrouter" or "ollama".
    """
    backend = config.get("ocr", {}).get("backend", "openrouter")
    logging.info(f"OCR pipeline: backend={backend} cardio={cardio_path} strength={strength_path}")
    t0 = time.time()

    if backend == "ollama":
        merged = await ollama_pipeline(cardio_path, strength_path, config, examples)
    elif backend == "openrouter":
        merged = await openrouter_pipeline(cardio_path, strength_path, config, examples)
    else:
        logging.error(f"Unknown OCR backend: {backend!r}. Must be 'openrouter' or 'ollama'.")
        raise ValueError(f"Unknown OCR backend: {backend!r}")

    # Save result to disk
    ts = int(time.time())
    result_path = os.path.join(config["paths"]["result_dir"], f"result_{ts}.json")
    with open(result_path, "w") as f:
        json.dump(merged, f, indent=2)

    elapsed = time.time() - t0
    logging.info(f"OCR pipeline complete ({backend}) in {elapsed:.2f}s → {result_path}")
    return merged


# ===========================================================================
#  OLLAMA HEALTH / SETUP HELPERS
# ===========================================================================


def check_ollama_health(config):
    """
    Check if Ollama server is running and the configured model is available.
    Returns (healthy: bool, message: str).
    """
    host = config.get("ollama", {}).get("host", "http://localhost:11434")
    model = config.get("ollama", {}).get("extraction_model", "qwen2.5-vl:7b")

    logging.info(f"Checking Ollama health at {host} for model {model}")

    # Check server is up
    response = httpx.get(f"{host}/api/tags", timeout=5.0)
    response.raise_for_status()
    data = response.json()

    available = [m["name"] for m in data.get("models", [])]
    logging.info(f"Ollama models available: {available}")

    # Check if our model is pulled
    # Model names from /api/tags include the tag (e.g. "qwen2.5-vl:7b")
    # Normalize for matching
    model_base = model.split(":")[0]
    found = any(model_base in m for m in available)

    if found:
        return True, f"Ollama OK. Model '{model}' is available."
    else:
        return False, (
            f"Ollama is running but model '{model}' not found. "
            f"Available: {available}. Run: ollama pull {model}"
        )


def pull_ollama_model(config):
    """
    Pull the configured Ollama model. Blocks until download completes.
    Useful for first-time setup.
    """
    import ollama

    host = config.get("ollama", {}).get("host", "http://localhost:11434")
    model = config.get("ollama", {}).get("extraction_model", "qwen2.5-vl:7b")

    logging.info(f"Pulling Ollama model: {model} from {host}")
    client = ollama.Client(host=host)

    # Stream progress
    for progress in client.pull(model, stream=True):
        status = progress.get("status", "")
        completed = progress.get("completed", 0)
        total = progress.get("total", 0)
        if total > 0:
            pct = (completed / total) * 100
            logging.info(f"[ollama pull] {status}: {pct:.1f}% ({completed}/{total})")
        else:
            logging.info(f"[ollama pull] {status}")

    logging.info(f"Model {model} pull complete")


# ===========================================================================
#  CLI — run directly to test OCR on local files
# ===========================================================================
#
#  uv run python ocr_service.py cardio.jpg strength.jpg
#  uv run python ocr_service.py --single form.jpg
#  uv run python ocr_service.py --check
#

if __name__ == "__main__":
    import sys
    import tomli as _tomli

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open("config.toml", "rb") as _f:
        _config = _tomli.load(_f)

    _examples_path = _config["paths"]["example_data_file"]
    _examples = {}
    if os.path.exists(_examples_path):
        with open(_examples_path) as _f:
            _examples = json.load(_f)

    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print("Usage:")
        print("  uv run python ocr_service.py <cardio.jpg> <strength.jpg>   # full pipeline")
        print("  uv run python ocr_service.py --single <image.jpg>          # one page only")
        print("  uv run python ocr_service.py --check                       # ollama health check")
        print("  uv run python ocr_service.py --pull                        # pull ollama model")
        sys.exit(0)

    if args[0] == "--check":
        ok, msg = check_ollama_health(_config)
        print(f"{'✅' if ok else '❌'} {msg}")
        sys.exit(0 if ok else 1)

    if args[0] == "--pull":
        pull_ollama_model(_config)
        sys.exit(0)

    if args[0] == "--single":
        # Single page extraction — no reconciliation
        image_path = args[1]
        backend = _config["ocr"]["backend"]
        model = _config["ollama"]["extraction_model"] if backend == "ollama" else _config["models"]["cardio_extraction_model"]
        print(f"Backend: {backend} | Model: {model} | Image: {image_path}")

        prompt = build_cardio_prompt(_examples.get("cardio_examples", []),
                                     force_json_instruction=(backend == "ollama" and not _ollama_model_supports_json(model)))

        if backend == "ollama":
            result = _ollama_extract_page_sync(image_path, prompt, "page", _config)
        else:
            async def _run():
                b64 = encode_image_base64(image_path)
                async with httpx.AsyncClient() as c:
                    return await _openrouter_extract_page(c, b64, prompt, CARDIO_SCHEMA, "page", model, _config)
            result = asyncio.run(_run())

        print(json.dumps(result, indent=2))

    else:
        # Two args = cardio + strength pair → full pipeline
        cardio_path, strength_path = args[0], args[1]
        print(f"Backend: {_config['ocr']['backend']}")
        print(f"Cardio:   {cardio_path}")
        print(f"Strength: {strength_path}")
        result = asyncio.run(process_form_ocr(cardio_path, strength_path, _config, _examples))
        print(json.dumps(result, indent=2))
