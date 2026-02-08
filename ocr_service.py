"""
ocr_service.py — Vision LLM extraction for fitness forms.
Supports two backends, selected via config.toml [ocr] backend setting:
  - "openrouter" : Cloud API calls via OpenRouter (GPT-4o, Claude, Gemini, etc.)
  - "ollama"     : Local CPU inference via Ollama (Qwen2.5-VL, Llama 3.2 Vision, Moondream)

Each image gets the SAME prompt — the LLM figures out which page type it is.
No more assumption about scan order.

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

_JSON_SUFFIX = (
    "\n\nIMPORTANT: Respond with ONLY valid JSON. "
    "No markdown fences, no explanation, no preamble. Just the raw JSON object."
)


def build_page_prompt(examples, *, force_json_instruction=False):
    """Single prompt for EITHER page type. The LLM identifies which form it sees."""
    prompt = """You are extracting data from a fitness form image. First, identify which type of form this is, then extract all data.

THERE ARE TWO POSSIBLE FORM TYPES:

=== TYPE 1: CARDIO RECORDING LOG ===
Title says "CARDIO RECORDING LOG" at the top.

Return JSON with this structure:
{
  "page_type": "cardio",
  "cccare_id": string or null,
  "target_hr": string or null,
  "target_rpe": string or null,
  "equipment_settings": { "nustep_arms": ..., "nustep_seat": ..., "leg_stab": ..., "recumbent_bike_seat": ..., "upright_bike_seat": ... },
  "sessions": [
    {
      "date": "Jan 5",
      "activity": "TM",
      "time_minutes": 30,
      "work_rate": "4.0 - 1%",
      "heart_rate_range": "87-105",
      "rpe": 13,
      "comments": null,
      "watch": "01",
      "kinesiologist_signed": true
    }
  ]
}

Activity codes: NS=NuStep, RB=Recumbent Bike, UB=Upright Bike, TM=Treadmill, E=Elliptical, ROW, LAPS
Work Rate format: treadmill="speed - incline%", NuStep="level @ resistance", bike="level @ watts"
RPE is 6-20 scale, integer. Comments null if blank. Watch is the watch number string.

=== TYPE 2: STRENGTHENING EXERCISES ===
Has exercise names as rows and dates as columns. Exercises include Thoracic rotation, Dynamic hip mobility, Squats, Chest press, Bent knee hip raise, Vertical traction, Airplane, Front bridge.

Return JSON with this structure:
{
  "page_type": "strength",
  "cccare_id": string or null,
  "year": 2026,
  "sessions": [
    {
      "date": "Jan 5",
      "exercises": [
        { "exercise_name": "Thoracic rotation (standing against wall)", "sets": [{"reps": 15, "weight": 0}] },
        { "exercise_name": "Chest press on bench", "sets": [{"reps": 10, "weight": 20}, {"reps": 10, "weight": 20}] },
        { "exercise_name": "Airplane exercise", "sets": [] }
      ],
      "stretches_completed": ["Quad", "Hamstring"],
      "kinesiologist_signed": true
    }
  ]
}

Chest press and Bent knee hip raise have TWO set rows each.
Airplane/Front bridge may have no data (empty sets []).
Weight 0 means bodyweight.
Stretches: 1. Quad, 2. Hamstring, 3. Glute, 4. Hip flexor, 5. Calf, 6. Chest, 7. Upper back — only include checked ones.

=== GENERAL RULES ===
- The form is landscape and may be rotated 90 degrees in the image
- CCCARE ID may be blank — return null, do NOT invent one
- "page_type" MUST be either "cardio" or "strength"
- Read ALL rows/columns that have handwritten data

"""
    cardio_examples = examples.get("cardio_examples", [])
    strength_examples = examples.get("strength_examples", [])
    ex_num = 1
    for ex in cardio_examples:
        prompt += f"\nExample {ex_num} (cardio):\n{ex['image_description']}\nCorrect extraction:\n{json.dumps(ex['expected_json'], indent=2)}\n"
        ex_num += 1
    for ex in strength_examples:
        prompt += f"\nExample {ex_num} (strength):\n{ex['image_description']}\nCorrect extraction:\n{json.dumps(ex['expected_json'], indent=2)}\n"
        ex_num += 1

    prompt += "\nNow extract from this new form image. Return JSON with page_type and all extracted data.\n"
    if force_json_instruction:
        prompt += _JSON_SUFFIX
    return prompt


def build_reconciliation_prompt(page1_data, page2_data, *, force_json_instruction=False):
    prompt = f"""You are reconciling data from two pages of the same fitness form.

Page 1 extracted data: {json.dumps(page1_data, indent=2)}
Page 2 extracted data: {json.dumps(page2_data, indent=2)}

CCCARE IDs found:
- Page 1: {page1_data.get('cccare_id')}
- Page 2: {page2_data.get('cccare_id')}

RULES:
1. If one page has a CCCARE ID and the other is null/blank, use the non-null ID
2. If both have IDs but they differ, pick the clearer/more complete one
3. If both are null, set cccare_id to null and confidence to "low"
4. Set id_mismatch to true only if both pages had non-null IDs that differ

Return merged JSON with:
- cccare_id: the single correct ID (or null)
- id_mismatch: boolean
- confidence: "high", "medium", or "low"
- cardio_data: the extraction from whichever page was the cardio page (based on page_type field)
- strength_data: the extraction from whichever page was the strength page (based on page_type field)
- notes: any reconciliation notes or null
"""
    if force_json_instruction:
        prompt += _JSON_SUFFIX
    return prompt


# ---------------------------------------------------------------------------
# RESPONSE PARSING
# ---------------------------------------------------------------------------


def parse_json_response(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
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


async def _openrouter_extract_page(client, image_b64, prompt, page_label, model, config):
    """Send a single page to OpenRouter. The prompt handles type detection."""
    logging.info(f"[openrouter] Extracting {page_label} with model={model}")
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
        "response_format": {"type": "json_object"},
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
    logging.info(f"[openrouter] {page_label} done in {elapsed:.2f}s")
    logging.debug(f"[openrouter] Raw response ({page_label}): {result}")

    return json.loads(result["choices"][0]["message"]["content"])


async def _openrouter_reconcile(page1_data, page2_data, page1_b64, page2_b64, config):
    model = config["models"]["reconciliation_model"]
    logging.info(f"[openrouter] Reconciling with model={model}")
    t0 = time.time()

    prompt = build_reconciliation_prompt(page1_data, page2_data)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{page1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{page2_b64}"}},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
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


async def openrouter_pipeline(page1_path, page2_path, config, examples):
    """Full OCR pipeline using OpenRouter. Same prompt for both images."""
    logging.info(f"[openrouter] Pipeline starting: page1={page1_path} page2={page2_path}")

    page1_b64 = encode_image_base64(page1_path)
    page2_b64 = encode_image_base64(page2_path)

    prompt = build_page_prompt(examples)
    model = config["models"]["cardio_extraction_model"]  # same model for both

    async with httpx.AsyncClient() as client:
        task1 = _openrouter_extract_page(client, page1_b64, prompt, "page1", model, config)
        task2 = _openrouter_extract_page(client, page2_b64, prompt, "page2", model, config)
        page1_data, page2_data = await asyncio.gather(task1, task2)

    logging.info(f"[openrouter] Page 1 detected as: {page1_data.get('page_type')}")
    logging.info(f"[openrouter] Page 2 detected as: {page2_data.get('page_type')}")

    merged = await _openrouter_reconcile(page1_data, page2_data, page1_b64, page2_b64, config)
    return merged


# ===========================================================================
#  BACKEND: OLLAMA (Local CPU inference)
# ===========================================================================


def _ollama_model_supports_json(model_name):
    base = model_name.split(":")[0].lower()
    full = model_name.lower()
    return base in OLLAMA_JSON_CAPABLE_MODELS or full in OLLAMA_JSON_CAPABLE_MODELS


def _ollama_extract_page_sync(image_path, prompt, page_label, config):
    """Extract a single page using Ollama (synchronous). Same prompt for any page type."""
    import ollama

    model = config["ollama"]["extraction_model"]
    host = config["ollama"].get("host", "http://localhost:11434")
    logging.info(f"[ollama] Extracting {page_label} with model={model} at {host}")
    t0 = time.time()

    client = ollama.Client(host=host)

    use_json_format = _ollama_model_supports_json(model)
    if not use_json_format:
        logging.info(f"[ollama] Model {model} not in JSON-capable list, using prompt-based JSON")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    options = {}
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
    logging.info(f"[ollama] {page_label} done in {elapsed:.2f}s ({len(raw_text)} chars)")
    logging.debug(f"[ollama] Raw response ({page_label}): {raw_text[:500]}")

    return parse_json_response(raw_text)


def _ollama_reconcile_sync(page1_data, page2_data, page1_path, page2_path, config):
    import ollama

    model = config["ollama"]["reconciliation_model"]
    host = config["ollama"].get("host", "http://localhost:11434")
    logging.info(f"[ollama] Reconciling with model={model} at {host}")
    t0 = time.time()

    client = ollama.Client(host=host)
    use_json_format = _ollama_model_supports_json(model)

    prompt = build_reconciliation_prompt(
        page1_data, page2_data,
        force_json_instruction=not use_json_format,
    )

    with open(page1_path, "rb") as f:
        page1_bytes = f.read()
    with open(page2_path, "rb") as f:
        page2_bytes = f.read()

    options = {}
    if config["ollama"].get("num_ctx"):
        options["num_ctx"] = config["ollama"]["num_ctx"]

    kwargs = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [page1_bytes, page2_bytes],
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


async def ollama_pipeline(page1_path, page2_path, config, examples):
    """Full OCR pipeline using local Ollama. Same prompt for both images."""
    logging.info(f"[ollama] Pipeline starting: page1={page1_path} page2={page2_path}")

    use_json = _ollama_model_supports_json(config["ollama"]["extraction_model"])
    force_json = not use_json

    prompt = build_page_prompt(examples, force_json_instruction=force_json)

    loop = asyncio.get_event_loop()

    future1 = loop.run_in_executor(
        None, _ollama_extract_page_sync, page1_path, prompt, "page1", config
    )
    future2 = loop.run_in_executor(
        None, _ollama_extract_page_sync, page2_path, prompt, "page2", config
    )

    page1_data, page2_data = await asyncio.gather(future1, future2)

    logging.info(f"[ollama] Page 1 detected as: {page1_data.get('page_type')}")
    logging.info(f"[ollama] Page 2 detected as: {page2_data.get('page_type')}")

    merged = await loop.run_in_executor(
        None, _ollama_reconcile_sync, page1_data, page2_data, page1_path, page2_path, config
    )

    return merged


# ===========================================================================
#  UNIFIED PIPELINE ENTRY POINT
# ===========================================================================


async def process_form_ocr(page1_path, page2_path, config, examples):
    """
    Run the full OCR pipeline. Takes two image paths (order doesn't matter).
    The LLM identifies which is cardio and which is strength.
    Saves the final merged JSON to disk and returns it.
    """
    backend = config.get("ocr", {}).get("backend", "openrouter")
    logging.info(f"OCR pipeline: backend={backend} page1={page1_path} page2={page2_path}")
    t0 = time.time()

    if backend == "ollama":
        merged = await ollama_pipeline(page1_path, page2_path, config, examples)
    elif backend == "openrouter":
        merged = await openrouter_pipeline(page1_path, page2_path, config, examples)
    else:
        logging.error(f"Unknown OCR backend: {backend!r}. Must be 'openrouter' or 'ollama'.")
        raise ValueError(f"Unknown OCR backend: {backend!r}")

    # Save result to disk
    ts = int(time.time())
    result_path = os.path.join(config["paths"]["result_dir"], f"result_{ts}.json")
    with open(result_path, "w") as f:
        json.dump(merged, f, indent=2)

    elapsed = time.time() - t0
    logging.info(f"OCR pipeline complete ({backend}) in {elapsed:.2f}s -> {result_path}")
    return merged


# ===========================================================================
#  OLLAMA HEALTH / SETUP HELPERS
# ===========================================================================


def check_ollama_health(config):
    host = config.get("ollama", {}).get("host", "http://localhost:11434")
    model = config.get("ollama", {}).get("extraction_model", "qwen2.5-vl:7b")

    logging.info(f"Checking Ollama health at {host} for model {model}")

    response = httpx.get(f"{host}/api/tags", timeout=5.0)
    response.raise_for_status()
    data = response.json()

    available = [m["name"] for m in data.get("models", [])]
    logging.info(f"Ollama models available: {available}")

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
    import ollama

    host = config.get("ollama", {}).get("host", "http://localhost:11434")
    model = config.get("ollama", {}).get("extraction_model", "qwen2.5-vl:7b")

    logging.info(f"Pulling Ollama model: {model} from {host}")
    client = ollama.Client(host=host)

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
#  uv run python ocr_service.py page1.jpg page2.jpg    # full pipeline (order doesn't matter)
#  uv run python ocr_service.py --single form.jpg       # one page only
#  uv run python ocr_service.py --check                  # ollama health check
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
        print("  uv run python ocr_service.py <page1.jpg> <page2.jpg>   # full pipeline (any order)")
        print("  uv run python ocr_service.py --single <image.jpg>      # one page only")
        print("  uv run python ocr_service.py --check                   # ollama health check")
        print("  uv run python ocr_service.py --pull                    # pull ollama model")
        sys.exit(0)

    if args[0] == "--check":
        ok, msg = check_ollama_health(_config)
        print(f"{'OK' if ok else 'FAIL'} {msg}")
        sys.exit(0 if ok else 1)

    if args[0] == "--pull":
        pull_ollama_model(_config)
        sys.exit(0)

    if args[0] == "--single":
        image_path = args[1]
        backend = _config["ocr"]["backend"]
        model = _config["ollama"]["extraction_model"] if backend == "ollama" else _config["models"]["cardio_extraction_model"]
        print(f"Backend: {backend} | Model: {model} | Image: {image_path}")

        force_json = (backend == "ollama" and not _ollama_model_supports_json(model))
        prompt = build_page_prompt(_examples, force_json_instruction=force_json)

        if backend == "ollama":
            result = _ollama_extract_page_sync(image_path, prompt, "page", _config)
        else:
            async def _run():
                b64 = encode_image_base64(image_path)
                async with httpx.AsyncClient() as c:
                    return await _openrouter_extract_page(c, b64, prompt, "page", model, _config)
            result = asyncio.run(_run())

        with open('results/output.json', 'w') as f:
            json.dump(result, f)
    else:
        # Two images — full pipeline, order doesn't matter
        path1, path2 = args[0], args[1]
        print(f"Backend: {_config['ocr']['backend']}")
        print(f"Page 1: {path1}")
        print(f"Page 2: {path2}")
        result = asyncio.run(process_form_ocr(path1, path2, _config, _examples))
        with open('results/output.json', 'w') as f:
            json.dump(result, f)
