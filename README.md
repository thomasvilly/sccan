# 💪 Fitness Form Capture Kiosk

Self-service camera-based kiosk for digitizing handwritten two-page weekly fitness forms. Uses OpenCV for real-time quality validation and vision LLMs for structured data extraction.

**Two OCR backends:**
- **Ollama** (local CPU, no API key needed) — Qwen2.5-VL, Llama 3.2 Vision, or Moondream
- **OpenRouter** (cloud API) — GPT-4o, Claude, Gemini, etc.

## Quick Start (WSL)

```bash
cd kiosk_capture

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env if using OpenRouter (not needed for Ollama)
```

### Option A: Local Ollama (default)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start the Ollama server
ollama serve &

# 3. Pull the default model (Qwen2.5-VL — best for OCR)
ollama pull qwen2.5vl:7b

# If downloads are slow, try the HF mirror:
# HF_ENDPOINT=https://hf-mirror.com ollama pull qwen2.5-vl:7b

# 4. Verify it's working
uv run python test_ollama_ocr.py --check

# 5. Test on a single image (optional)
uv run python test_ollama_ocr.py path/to/form.jpg

# 6. Run the kiosk
uv run streamlit run main.py
```

### Option B: OpenRouter (cloud API)

```bash
# 1. Edit config.toml: set backend = "openrouter"
# 2. Add your API key to .env
# 3. Run
uv run streamlit run main.py
```

## Switching Backends

Edit `config.toml`:

```toml
[ocr]
backend = "ollama"      # Local CPU inference
# backend = "openrouter"  # Cloud API
```

Restart the kiosk after changing.

## Ollama Model Options

| Model | Config value | RAM | Speed (CPU) | Best for |
|-------|-------------|-----|-------------|----------|
| Qwen 2.5-VL 7B | `qwen2.5-vl:7b` | ~6GB | 5-10s | Dense text, tables (default) |
| Llama 3.2 Vision 11B | `llama3.2-vision:11b` | ~8GB | 5-10s | General reasoning |
| Moondream 2 | `moondream:1.8b` | ~2GB | <1s | Speed over accuracy |

Change in `config.toml`:

```toml
[ollama]
extraction_model = "qwen2.5-vl:7b"       # or llama3.2-vision:11b or moondream:1.8b
reconciliation_model = "qwen2.5-vl:7b"
```

## Project Structure

```
kiosk_capture/
├── main.py                  # State machine, OpenCV, Streamlit UI
├── ocr_service.py           # OCR extraction (Ollama + OpenRouter) — also runs standalone
├── app.py                   # HuggingFace Spaces entry point
├── config.toml              # All tunable parameters
├── few_shot_examples.json   # Ground truth examples for OCR prompting
├── .env.example             # Environment variable template
├── pyproject.toml           # UV/pip dependencies
├── requirements.txt         # HF Spaces dependencies
├── tests/
│   └── test_kiosk.py        # Unit tests
├── captures/                # Saved page images
├── orphans/                 # Orphaned single pages
└── results/                 # OCR output JSON files
```

## Testing OCR on Local Files

`ocr_service.py` runs directly — no kiosk or camera needed:

```bash
# Health check (is Ollama running?)
uv run python ocr_service.py --check

# Pull the configured model
uv run python ocr_service.py --pull

# Extract a single page
uv run python ocr_service.py --single ~/scans/form_photo.jpg

# Full pipeline on a cardio + strength pair
uv run python ocr_service.py ~/scans/cardio.jpg ~/scans/strength.jpg
```

Results save as JSON to `./results/`.

## Running Tests

```bash
uv run pytest tests/ -v
```

## State Machine

```
READY → (form detected 10 frames) → PROCESSING → (3 good frames) → CAPTURE
  ↑                                      ↓ (3s failures)
  │                                   ISSUES → (5 good frames) → PROCESSING
  │
  ├── CAPTURE → detect page type → AWAITING_SECOND_PAGE → (form detected) → PROCESSING
  │                                       ↓ (120s timeout)
  │                                  Save orphan → READY
  │
  └── DONE → (3s display) → READY
       ↑
       └── Both pages captured (or orphan matched)
```

## WSL Camera Setup

```powershell
# Windows PowerShell (admin):
winget install usbipd
usbipd list
usbipd bind --busid <BUS_ID>
usbipd attach --wsl --busid <BUS_ID>
```

```bash
# WSL:
ls /dev/video*   # Verify camera
```
