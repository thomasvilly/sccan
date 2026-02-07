"""
test_ollama_ocr.py — Standalone CLI to test local Ollama vision OCR on a single image.
Useful for verifying Ollama is working before running the full kiosk.

Usage:
    uv run python test_ollama_ocr.py path/to/image.jpg
    uv run python test_ollama_ocr.py path/to/image.jpg --model llama3.2-vision:11b
    uv run python test_ollama_ocr.py --check    # Just check if Ollama is running

Requires: ollama server running (ollama serve)
"""

import json
import logging
import sys
import time

import tomli

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_config(path="config.toml"):
    with open(path, "rb") as f:
        return tomli.load(f)


def check_health(config):
    from ocr_service import check_ollama_health
    healthy, msg = check_ollama_health(config)
    print(f"{'✅' if healthy else '❌'} {msg}")
    return healthy


def pull_model(config):
    from ocr_service import pull_ollama_model
    print(f"Pulling model: {config['ollama']['extraction_model']}...")
    pull_ollama_model(config)
    print("Done.")


def test_single_image(image_path, config, model_override=None):
    import ollama as ollama_lib
    from ocr_service import _ollama_model_supports_json, parse_json_response

    model = model_override or config["ollama"]["extraction_model"]
    host = config["ollama"].get("host", "http://localhost:11434")

    print(f"Model:  {model}")
    print(f"Host:   {host}")
    print(f"Image:  {image_path}")
    print(f"JSON mode: {_ollama_model_supports_json(model)}")
    print("-" * 60)

    client = ollama_lib.Client(host=host)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = (
        "Extract ALL text and data you can see in this image. "
        "If it's a form or table, preserve the structure. "
        "Return the result as a JSON object."
    )

    use_json = _ollama_model_supports_json(model)
    if not use_json:
        prompt += (
            "\n\nIMPORTANT: Respond with ONLY valid JSON. "
            "No markdown fences, no explanation, no preamble."
        )

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_bytes]}],
        "options": {},
    }
    if config["ollama"].get("num_ctx"):
        kwargs["options"]["num_ctx"] = config["ollama"]["num_ctx"]
    if use_json:
        kwargs["format"] = "json"

    print("Sending to Ollama... (this may take a while on CPU)")
    t0 = time.time()
    response = client.chat(**kwargs)
    elapsed = time.time() - t0

    raw = response["message"]["content"]
    print(f"\n⏱  Completed in {elapsed:.1f}s")
    print(f"📏 Response length: {len(raw)} chars")
    print("-" * 60)

    # Try to parse as JSON
    print("\n📋 RAW RESPONSE:")
    print(raw)

    print("\n" + "-" * 60)
    parsed = parse_json_response(raw)
    print("✅ PARSED JSON:")
    print(json.dumps(parsed, indent=2))


def main():
    config = load_config()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  uv run python test_ollama_ocr.py <image_path>           # Test OCR on image")
        print("  uv run python test_ollama_ocr.py <image_path> --model X # Use specific model")
        print("  uv run python test_ollama_ocr.py --check                # Health check")
        print("  uv run python test_ollama_ocr.py --pull                 # Pull configured model")
        print()
        print(f"Current config: backend={config.get('ocr', {}).get('backend')}")
        print(f"  Ollama model: {config.get('ollama', {}).get('extraction_model')}")
        print(f"  Ollama host:  {config.get('ollama', {}).get('host')}")
        sys.exit(1)

    if sys.argv[1] == "--check":
        check_health(config)
        sys.exit(0)

    if sys.argv[1] == "--pull":
        pull_model(config)
        sys.exit(0)

    image_path = sys.argv[1]
    model_override = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model_override = sys.argv[idx + 1]

    test_single_image(image_path, config, model_override)


if __name__ == "__main__":
    main()
