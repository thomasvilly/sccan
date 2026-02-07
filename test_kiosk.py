"""
tests/test_kiosk.py — Unit and integration tests for the kiosk capture system.
Run with: uv run pytest tests/ -v
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from main import (
    CONFIG,
    READY,
    PROCESSING,
    ISSUES,
    AWAITING_SECOND_PAGE,
    DONE,
    CapturedPage,
    buffer_add,
    buffer_clear,
    buffer_get_sharpest,
    cleanup_old_orphans,
    detect_page_type,
    find_orphan_match,
    form_detected,
    make_frame_buffer,
    save_as_orphan,
    validate_frame,
    ERROR_MESSAGES,
)

from ocr_service import (
    build_cardio_prompt,
    build_strength_prompt,
    build_reconciliation_prompt,
    parse_json_response,
    CARDIO_SCHEMA,
    STRENGTH_SCHEMA,
    MERGED_SCHEMA,
    _ollama_model_supports_json,
)

# ---------------------------------------------------------------------------
# TEST HELPERS
# ---------------------------------------------------------------------------


def make_white_image(h=1080, w=1920):
    return np.ones((h, w, 3), dtype=np.uint8) * 240


def make_dark_image(h=1080, w=1920):
    return np.ones((h, w, 3), dtype=np.uint8) * 20


def make_textured_image(h=1080, w=1920):
    img = np.ones((h, w, 3), dtype=np.uint8) * 220
    for y in range(100, h, 60):
        cv2.line(img, (100, y), (w - 100, y), (0, 0, 0), 1)
    for x in range(100, w, 200):
        cv2.line(img, (x, 100), (x, h - 100), (0, 0, 0), 1)
    for _ in range(50):
        cx = np.random.randint(150, w - 150)
        cy = np.random.randint(150, h - 150)
        cv2.putText(img, "123", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
    return img


def make_blurry_image(h=1080, w=1920):
    img = make_textured_image(h, w)
    return cv2.GaussianBlur(img, (51, 51), 0)


def make_glary_image(h=1080, w=1920):
    img = make_textured_image(h, w)
    img[200:800, 400:1500] = 255
    return img


def make_form_image(h=1080, w=1920):
    img = np.ones((h, w, 3), dtype=np.uint8) * 60
    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
    cv2.rectangle(img, (margin_x, margin_y), (w - margin_x, h - margin_y), (240, 240, 240), -1)
    cv2.rectangle(img, (margin_x, margin_y), (w - margin_x, h - margin_y), (0, 0, 0), 3)
    return img


def make_temp_dir():
    return tempfile.mkdtemp()


def _save_temp_image(img, tmpdir, name="test.jpg"):
    path = os.path.join(tmpdir, name)
    cv2.imwrite(path, img)
    return path


# ---------------------------------------------------------------------------
# CONFIG TESTS
# ---------------------------------------------------------------------------


def test_config():
    for section in ("camera", "state_transitions", "timeouts", "opencv_thresholds",
                    "form_detection", "page_detection", "buffer", "paths", "api",
                    "models", "logging", "ocr", "ollama"):
        assert section in CONFIG, f"Missing config section: {section}"


def test_config_ocr_backend():
    assert CONFIG["ocr"]["backend"] in ("openrouter", "ollama")


# ---------------------------------------------------------------------------
# FORM DETECTION TESTS
# ---------------------------------------------------------------------------


def test_form_detection_with_form():
    assert form_detected(make_form_image(), CONFIG) is True


def test_form_detection_no_form():
    assert form_detected(make_dark_image(), CONFIG) is False


def test_form_detection_plain_white():
    assert form_detected(make_white_image(), CONFIG) is False


# ---------------------------------------------------------------------------
# FRAME VALIDATION TESTS
# ---------------------------------------------------------------------------


def test_validate_dark_image():
    passed, reason, _ = validate_frame(make_dark_image(), CONFIG)
    assert passed is False
    assert reason == "no_form_detected"


def test_validate_blurry_image():
    passed, reason, _ = validate_frame(make_blurry_image(), CONFIG)
    assert passed is False
    assert reason in ("image_blurry", "no_form_detected", "contrast_low")


def test_validate_glary_image():
    passed, reason, _ = validate_frame(make_glary_image(), CONFIG)
    assert passed is False


def test_validate_good_image():
    passed, reason, scores = validate_frame(make_textured_image(), CONFIG)
    assert passed is True
    assert reason is None
    assert "sharpness" in scores
    assert "contrast" in scores
    assert "glare_ratio" in scores


# ---------------------------------------------------------------------------
# PAGE TYPE DETECTION
# ---------------------------------------------------------------------------


def test_page_type_returns_valid():
    tmpdir = make_temp_dir()
    path = _save_temp_image(make_textured_image(), tmpdir)
    result = detect_page_type(path, CONFIG)
    assert result in ("cardio", "strength", None)
    shutil.rmtree(tmpdir)


def test_page_type_blank():
    tmpdir = make_temp_dir()
    path = _save_temp_image(make_white_image(), tmpdir)
    result = detect_page_type(path, CONFIG)
    assert result in (None, "strength")
    shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# FRAME BUFFER
# ---------------------------------------------------------------------------


def test_frame_buffer_basic():
    buf = make_frame_buffer(3)
    f1 = np.zeros((10, 10, 3), dtype=np.uint8)
    f2 = np.ones((10, 10, 3), dtype=np.uint8)
    f3 = np.ones((10, 10, 3), dtype=np.uint8) * 2

    buffer_add(buf, f1, sharpness=10.0)
    buffer_add(buf, f2, sharpness=50.0)
    buffer_add(buf, f3, sharpness=30.0)

    assert np.array_equal(buffer_get_sharpest(buf), f2)


def test_frame_buffer_overflow():
    buf = make_frame_buffer(2)
    f1 = np.zeros((5, 5, 3), dtype=np.uint8)
    f2 = np.ones((5, 5, 3), dtype=np.uint8)
    f3 = np.ones((5, 5, 3), dtype=np.uint8) * 2

    buffer_add(buf, f1, 100.0)
    buffer_add(buf, f2, 50.0)
    buffer_add(buf, f3, 75.0)

    assert len(buf["frames"]) == 2
    assert np.array_equal(buffer_get_sharpest(buf), f3)


def test_frame_buffer_clear():
    buf = make_frame_buffer(5)
    buffer_add(buf, np.zeros((5, 5, 3), dtype=np.uint8), 10.0)
    buffer_clear(buf)
    assert len(buf["frames"]) == 0
    assert buffer_get_sharpest(buf) is None


# ---------------------------------------------------------------------------
# ORPHAN MANAGEMENT
# ---------------------------------------------------------------------------


def test_orphan_creation():
    tmpdir = make_temp_dir()
    config = {**CONFIG, "paths": {**CONFIG["paths"], "orphan_dir": tmpdir}}

    src = os.path.join(tmpdir, "page_test.jpg")
    cv2.imwrite(src, make_textured_image(100, 100))

    orphan_path = save_as_orphan(src, "cardio", config)
    assert os.path.exists(orphan_path)
    assert "orphan_cardio_" in orphan_path
    assert os.path.exists(orphan_path.replace(".jpg", ".json"))
    shutil.rmtree(tmpdir)


def test_orphan_matching():
    tmpdir = make_temp_dir()
    config = {**CONFIG, "paths": {**CONFIG["paths"], "orphan_dir": tmpdir}, "timeouts": {**CONFIG["timeouts"]}}

    ts = int(time.time())
    orphan_path = os.path.join(tmpdir, f"orphan_cardio_{ts}.jpg")
    cv2.imwrite(orphan_path, make_textured_image(100, 100))

    assert find_orphan_match("strength", config) is not None
    assert find_orphan_match("cardio", config) is None
    shutil.rmtree(tmpdir)


def test_orphan_expired():
    tmpdir = make_temp_dir()
    config = {**CONFIG, "paths": {**CONFIG["paths"], "orphan_dir": tmpdir},
              "timeouts": {**CONFIG["timeouts"], "orphan_timeout_seconds": 120}}

    old_ts = int(time.time()) - 300
    orphan_path = os.path.join(tmpdir, f"orphan_cardio_{old_ts}.jpg")
    cv2.imwrite(orphan_path, make_textured_image(100, 100))

    assert find_orphan_match("strength", config) is None
    shutil.rmtree(tmpdir)


def test_orphan_cleanup():
    tmpdir = make_temp_dir()
    config = {**CONFIG, "paths": {**CONFIG["paths"], "orphan_dir": tmpdir}}

    old_ts = int(time.time()) - 7200
    for ext in (".jpg", ".json"):
        with open(os.path.join(tmpdir, f"orphan_cardio_{old_ts}{ext}"), "w") as f:
            f.write("test")

    new_ts = int(time.time())
    for ext in (".jpg", ".json"):
        with open(os.path.join(tmpdir, f"orphan_strength_{new_ts}{ext}"), "w") as f:
            f.write("test")

    cleanup_old_orphans(config, max_age_seconds=3600)

    remaining = os.listdir(tmpdir)
    assert not any(str(old_ts) in f for f in remaining)
    assert any(str(new_ts) in f for f in remaining)
    shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# ERROR MESSAGES
# ---------------------------------------------------------------------------


def test_error_messages_coverage():
    expected = ["no_form_detected", "image_blurry", "contrast_low",
                "glare_detected", "page_type_unreadable", "duplicate_page", "generic"]
    for reason in expected:
        assert reason in ERROR_MESSAGES


# ---------------------------------------------------------------------------
# OCR SERVICE TESTS (prompts, parsing, schema)
# ---------------------------------------------------------------------------


def test_cardio_prompt_builder():
    from main import EXAMPLES
    prompt = build_cardio_prompt(EXAMPLES.get("cardio_examples", []))
    assert "CARDIO RECORDING LOG" in prompt
    assert "CCCARE ID" in prompt


def test_strength_prompt_builder():
    from main import EXAMPLES
    prompt = build_strength_prompt(EXAMPLES.get("strength_examples", []))
    assert "STRENGTHENING EXERCISES" in prompt
    assert "Squats" in prompt


def test_reconciliation_prompt_builder():
    prompt = build_reconciliation_prompt(
        {"cccare_id": "123", "sessions": []},
        {"cccare_id": "123", "exercises": []},
    )
    assert "123" in prompt
    assert "cccare_id" in prompt


def test_prompt_force_json_instruction():
    prompt = build_cardio_prompt([], force_json_instruction=True)
    assert "ONLY valid JSON" in prompt

    prompt_no = build_cardio_prompt([], force_json_instruction=False)
    assert "ONLY valid JSON" not in prompt_no


def test_parse_json_clean():
    raw = '{"key": "value"}'
    assert parse_json_response(raw) == {"key": "value"}


def test_parse_json_with_fences():
    raw = '```json\n{"key": "value"}\n```'
    assert parse_json_response(raw) == {"key": "value"}


def test_parse_json_with_fences_no_lang():
    raw = '```\n{"a": 1}\n```'
    assert parse_json_response(raw) == {"a": 1}


def test_schemas_have_required_fields():
    assert "cccare_id" in CARDIO_SCHEMA["required"]
    assert "cccare_id" in STRENGTH_SCHEMA["required"]
    assert "cccare_id" in MERGED_SCHEMA["required"]


def test_ollama_json_capable_detection():
    assert _ollama_model_supports_json("qwen2.5vl:7b") is True
    assert _ollama_model_supports_json("qwen2.5vl") is True
    assert _ollama_model_supports_json("llama3.2-vision:11b") is True
    assert _ollama_model_supports_json("moondream:1.8b") is False
    assert _ollama_model_supports_json("some-random-model") is False
