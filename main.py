"""
Fitness Form Capture Kiosk — main.py
Self-service camera-based kiosk for digitizing handwritten fitness forms.
State machine + OpenCV validation + Streamlit UI.
OCR extraction lives in ocr_service.py.

Functional style. No try/except. Logs everything.

Run with: uv run streamlit run main.py
"""

import asyncio
import glob
import json
import logging
import os
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path
import pygame

import cv2
import numpy as np
import streamlit as st
import tomli
from dotenv import load_dotenv

from ocr_service import process_form_ocr

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

load_dotenv()


def load_config(config_path="config.toml"):
    with open(config_path, "rb") as f:
        config = tomli.load(f)
    return config


CONFIG = load_config()

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------


def setup_logging(config):
    level = getattr(logging, config["logging"]["log_level"].upper(), logging.INFO)
    handlers = []
    if config["logging"]["log_to_console"]:
        handlers.append(logging.StreamHandler())
    handlers.append(logging.FileHandler(config["logging"]["log_file"]))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


setup_logging(CONFIG)

# ---------------------------------------------------------------------------
# DIRECTORY SETUP
# ---------------------------------------------------------------------------

for dir_key in ("capture_dir", "orphan_dir", "result_dir", "temp_dir"):
    Path(CONFIG["paths"][dir_key]).mkdir(parents=True, exist_ok=True)

pygame.mixer.init()

def play_shutter(track):
    if track == "START":
        sound = pygame.mixer.Sound("sounds/Place in paper.mp3")
    elif track =="SCAN":
        sound = pygame.mixer.Sound("sounds/Scanning.mp3")
    elif track =="ISSUES":
        sound = pygame.mixer.Sound("sounds/Try again.mp3")
    elif track =="FLIP":
        sound = pygame.mixer.Sound("sounds/Flip.mp3")
    elif track =="DONE":
        sound = pygame.mixer.Sound("sounds/Scan complete.mp3")
    sound.play()
    return None

# ---------------------------------------------------------------------------
# DATA TYPES
# ---------------------------------------------------------------------------

CapturedPage = namedtuple("CapturedPage", ["filepath", "page_type", "timestamp"])

# ---------------------------------------------------------------------------
# FEW-SHOT EXAMPLES
# ---------------------------------------------------------------------------


def load_examples(path):
    if not os.path.exists(path):
        logging.warning(f"Few-shot examples file not found: {path}")
        return {"cardio_examples": [], "strength_examples": []}
    with open(path, "r") as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data.get('cardio_examples', []))} cardio and {len(data.get('strength_examples', []))} strength examples")
    return data


EXAMPLES = load_examples(CONFIG["paths"]["example_data_file"])

# ---------------------------------------------------------------------------
# OPENCV: FORM DETECTION (READY state)
# ---------------------------------------------------------------------------


def form_detected(frame, config):
    """
    Detect if a rectangular form is present in the frame via contour analysis.
    Returns True if a landscape quadrilateral of appropriate size is found.
    """
    fd = config["form_detection"]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = fd["gaussian_kernel"]
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(blurred, fd["canny_low"], fd["canny_high"])
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    frame_area = frame.shape[0] * frame.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, fd["contour_approx_epsilon"] * perimeter, True)

        if 4 <= len(approx) <= 8:
            area = cv2.contourArea(approx)
            area_ratio = area / frame_area

            if fd["area_ratio_min"] <= area_ratio <= fd["area_ratio_max"]:
                rect = cv2.minAreaRect(approx)
                w, h = rect[1]
                if min(w, h) == 0:
                    continue
                aspect = max(w, h) / min(w, h)
                if fd["aspect_ratio_min"] <= aspect <= fd["aspect_ratio_max"]:
                    logging.debug(f"Form detected: area_ratio={area_ratio:.3f} aspect={aspect:.2f}")
                    return True

    return False


# ---------------------------------------------------------------------------
# OPENCV: FRAME VALIDATION (PROCESSING state)
# ---------------------------------------------------------------------------


def validate_frame(frame, config):
    """
    Run image quality checks on a frame.
    Returns (passed: bool, failure_reason: str|None, scores: dict)
    """
    th = config["opencv_thresholds"]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scores = {}

    # 1. Mean intensity — is anything even there?
    mean_intensity = float(gray.mean())
    scores["mean_intensity"] = mean_intensity
    if mean_intensity < th["mean_intensity_min"]:
        return False, "no_form_detected", scores

    # 2. Blur (Laplacian variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    scores["sharpness"] = laplacian_var
    if laplacian_var < th["blur_threshold"]:
        return False, "image_blurry", scores

    # 3. Contrast (std dev of grayscale)
    contrast = float(gray.std())
    scores["contrast"] = contrast
    if contrast < th["contrast_threshold"]:
        return False, "contrast_low", scores

    # 4. Glare (blown-out pixel ratio)
    glare_ratio = float(np.sum(gray > 250) / gray.size)
    scores["glare_ratio"] = glare_ratio
    if glare_ratio > th["glare_threshold"]:
        return False, "glare_detected", scores

    return True, None, scores


# ---------------------------------------------------------------------------
# PAGE TYPE DETECTION (OpenCV heuristic)
# ---------------------------------------------------------------------------


def detect_page_type(image_path, config):
    """
    BYPASS: Always return 'generic_form' so we capture whatever is visible.
    We assume the user follows the workflow: Scan Side A -> Flip -> Scan Side B.
    The LLM will figure out which image is which later.
    """
    # Verify the image file actually exists and isn't empty
    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        return None
        
    # We just return a placeholder string. 
    # The exact string doesn't matter as long as it's not None.
    return "generic_form"


# ---------------------------------------------------------------------------
# ORPHAN PAGE MANAGEMENT
# ---------------------------------------------------------------------------


def save_as_orphan(page_filepath, page_type, config):
    """Move a captured page into the orphan directory."""
    ts = int(time.time())
    orphan_filename = f"orphan_{page_type}_{ts}.jpg"
    orphan_path = os.path.join(config["paths"]["orphan_dir"], orphan_filename)
    os.rename(page_filepath, orphan_path)

    metadata = {
        "filename": orphan_filename,
        "page_type": page_type,
        "timestamp": ts,
        "created_at": datetime.now().isoformat(),
    }
    meta_path = orphan_path.replace(".jpg", ".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    logging.warning(f"Created orphan: {orphan_path}")
    return orphan_path


def find_orphan_match(new_page_type, config):
    """
    Look for an orphaned page of the OPPOSITE type created within the orphan timeout window.
    Returns the filepath of the matching orphan, or None.
    """
    cutoff = time.time() - config["timeouts"]["orphan_timeout_seconds"]
    needed_type = "strength" if new_page_type == "cardio" else "cardio"

    pattern = os.path.join(config["paths"]["orphan_dir"], f"orphan_{needed_type}_*.jpg")
    orphan_files = glob.glob(pattern)

    for orphan_file in sorted(orphan_files, reverse=True):
        basename = os.path.basename(orphan_file)
        parts = basename.replace(".jpg", "").split("_")
        orphan_ts = int(parts[-1])
        if orphan_ts >= cutoff:
            logging.info(f"Orphan match found: {orphan_file} (age={time.time()-orphan_ts:.0f}s)")
            return orphan_file

    return None


def cleanup_old_orphans(config, max_age_seconds=3600):
    """Remove orphan files older than max_age_seconds."""
    cutoff = time.time() - max_age_seconds
    pattern = os.path.join(config["paths"]["orphan_dir"], "orphan_*")
    for f in glob.glob(pattern):
        basename = os.path.basename(f)
        parts = basename.replace(".jpg", "").replace(".json", "").split("_")
        ts = int(parts[-1])
        if ts < cutoff:
            os.remove(f)
            logging.info(f"Cleaned up old orphan: {f}")


def is_duplicate_image(img1_source, img2_source, threshold=15):
    """
    Compares two images. Inputs can be file paths (str) or numpy arrays (frame).
    """
    def load_gray_thumb(source):
        # If it's a file path, load it
        if isinstance(source, str):
            img = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
        # If it's a numpy array (live frame), convert to gray
        else:
            img = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if len(source.shape) == 3 else source
            
        if img is None: return None
        return cv2.resize(img, (64, 64)).astype("float")

    try:
        thumb1 = load_gray_thumb(img1_source)
        thumb2 = load_gray_thumb(img2_source)
        
        if thumb1 is None or thumb2 is None: return False

        diff = np.mean(np.abs(thumb1 - thumb2))
        return diff < threshold
    except Exception:
        return False
        
# ---------------------------------------------------------------------------
# FRAME BUFFER
# ---------------------------------------------------------------------------


def make_frame_buffer(max_size):
    return {"frames": [], "max_size": max_size}


def buffer_add(buf, frame, sharpness):
    buf["frames"].append({"frame": frame, "sharpness": sharpness})
    if len(buf["frames"]) > buf["max_size"]:
        buf["frames"].pop(0)


def buffer_get_sharpest(buf):
    if not buf["frames"]:
        return None
    return max(buf["frames"], key=lambda x: x["sharpness"])["frame"]


def buffer_clear(buf):
    buf["frames"].clear()


# ---------------------------------------------------------------------------
# ERROR MESSAGES
# ---------------------------------------------------------------------------

ERROR_MESSAGES = {
    "no_form_detected": "Please place your form in the tray",
    "image_blurry": "Image is blurry — please hold still",
    "contrast_low": "Lighting is too dim — check the desk lamp",
    "glare_detected": "Glare detected — adjust the lamp angle",
    "page_type_unreadable": "Cannot read form — please reposition",
    "duplicate_page": "Same page detected — please flip to the other side",
    "generic": "Please adjust your form and try again",
}


# ---------------------------------------------------------------------------
# STATE MACHINE
# ---------------------------------------------------------------------------

READY = "READY"
PROCESSING = "PROCESSING"
ISSUES = "ISSUES"
AWAITING_SECOND_PAGE = "AWAITING_SECOND_PAGE"
DONE = "DONE"


def init_session_state():
    defaults = {
        "state": READY,
        "consecutive_detections": 0,
        "consecutive_good": 0,
        "consecutive_bad_start": None,
        "failure_counts": {},
        "frame_buffer": make_frame_buffer(CONFIG["buffer"]["frame_buffer_size"]),
        "first_page": None,
        "issue_reason": "generic",
        "issue_consecutive_good": 0,
        "awaiting_start_time": None,
        "awaiting_consecutive_detections": 0,
        "done_start_time": None,
        "ocr_task_launched": False,
        "last_result": None,
        "camera_opened": False,
        "awaiting_form_gone": False,
        "awaiting_cooldown_until": 0,
        "awaiting_form_gone_count": 0,
        "last_finished_scan": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def transition_to(new_state, **kwargs):
    old = st.session_state.state
    st.session_state.state = new_state
    logging.info(f"State: {old} -> {new_state} {kwargs if kwargs else ''}")

    if new_state == READY:
        st.session_state.consecutive_detections = 0
        st.session_state.first_page = None
        st.session_state.ocr_task_launched = False
        play_shutter("START")
    elif new_state == PROCESSING:
        st.session_state.consecutive_good = 0
        st.session_state.consecutive_bad_start = None
        st.session_state.failure_counts = {}
        buffer_clear(st.session_state.frame_buffer)
        # play_shutter("SCAN")
    elif new_state == ISSUES:
        st.session_state.issue_reason = kwargs.get("reason", "generic")
        st.session_state.issue_consecutive_good = 0
        play_shutter("ISSUES")
    elif new_state == AWAITING_SECOND_PAGE:
        st.session_state.awaiting_start_time = time.time()
        st.session_state.awaiting_consecutive_detections = 0
        st.session_state.awaiting_form_gone = False
        st.session_state.awaiting_cooldown_until = time.time() + 2.0
        st.session_state.awaiting_form_gone_count = 0
        play_shutter("FLIP")
    elif new_state == DONE:
        st.session_state.done_start_time = time.time()
        st.session_state.ocr_task_launched = False
        play_shutter("DONE")


def save_captured_frame(frame, config):
    ts = int(time.time() * 1000)
    filename = f"page_{ts}.jpg"
    filepath = os.path.join(config["paths"]["capture_dir"], filename)
    cv2.imwrite(filepath, frame)
    logging.info(f"Saved captured frame: {filepath}")
    return filepath

def handle_capture(best_frame, config, is_second_page=False):
    """
    Saves the frame and handles the logic for Page 1 vs Page 2.
    Now includes a hard reset to prevent the '22/1 frames' loop.
    """
    # 1. Save the image to disk
    filepath = save_captured_frame(best_frame, config)
    logging.info(f"Captured candidate: {filepath}")

    # 2. IF THIS IS THE SECOND PAGE (The Logic Fix)
    if is_second_page and st.session_state.first_page is not None:
        first = st.session_state.first_page
        
        if is_duplicate_image(first.filepath, filepath, threshold=5):
            logging.warning("Duplicate detected: You haven't flipped the page yet.")
            
            st.session_state.consecutive_good = 0
            
            # Delete the useless duplicate file
            try:
                os.remove(filepath)
            except OSError:
                pass

            # Tell the user why nothing happened
            st.toast("⚠️ Still seeing Page 1. Please flip the paper!", icon="📄")
            return 

        # 3. SUCCESS: We have two DIFFERENT pages
        logging.info(f"Two distinct pages captured. Sending to OCR.")

        st.session_state.last_result = {
            "cardio": first.filepath,
            "strength": filepath
        }
        st.session_state.last_finished_scan = filepath 
        transition_to(DONE)
        return

    # 4. IF THIS IS THE FIRST PAGE
    # Just save it and wait for the second one.
    st.session_state.first_page = CapturedPage(
        filepath=filepath, 
        page_type="generic_form", 
        timestamp=time.time()
    )
    
    # Notify user
    st.toast("✅ Page 1 Captured! Flip to Page 2.", icon="🔄")
    transition_to(AWAITING_SECOND_PAGE)

# ---------------------------------------------------------------------------
# STATE HANDLERS
# ---------------------------------------------------------------------------



def process_frame_ready(frame, config):
    # NEW: If the thing in the camera is the exact same thing we just finished, IGNORE IT.
    if st.session_state.last_finished_scan:
        if is_duplicate_image(st.session_state.last_finished_scan, frame, threshold=15):
            return # Ignore this frame, stay in READY

    if form_detected(frame, config):
        st.session_state.consecutive_detections += 1
        if st.session_state.consecutive_detections >= config["state_transitions"]["form_detection_frames"]:
            transition_to(PROCESSING)
    else:
        st.session_state.consecutive_detections = 0


def process_frame_processing(frame_full, frame, config):
    passed, reason, scores = validate_frame(frame, config)
    logging.debug(f"Validation: passed={passed} reason={reason} scores={scores}")

    if passed:
        buffer_add(st.session_state.frame_buffer, frame_full, scores.get("sharpness", 0))
        st.session_state.consecutive_good += 1
        st.session_state.consecutive_bad_start = None

        if st.session_state.consecutive_good >= config["state_transitions"]["good_capture_frames"]:
            best = buffer_get_sharpest(st.session_state.frame_buffer)
            is_second = st.session_state.first_page is not None
            handle_capture(best, config, is_second_page=is_second)
    else:
        st.session_state.consecutive_good = 0
        st.session_state.failure_counts[reason] = st.session_state.failure_counts.get(reason, 0) + 1

        if st.session_state.consecutive_bad_start is None:
            st.session_state.consecutive_bad_start = time.time()

        elapsed = time.time() - st.session_state.consecutive_bad_start
        if elapsed >= config["timeouts"]["issue_timeout_seconds"]:
            most_common = max(st.session_state.failure_counts, key=st.session_state.failure_counts.get)
            transition_to(ISSUES, reason=most_common)


def process_frame_issues(frame, config):
    passed, _, _ = validate_frame(frame, config)
    if passed:
        st.session_state.issue_consecutive_good += 1
        if st.session_state.issue_consecutive_good >= config["state_transitions"]["issue_retry_frames"]:
            transition_to(PROCESSING)
    else:
        st.session_state.issue_consecutive_good = 0


def process_frame_awaiting(frame, config):
    # Just wait a tiny bit (1s) so the user can start moving their hand
    if time.time() < st.session_state.awaiting_cooldown_until:
        return

    # Timeout: If they take too long (>120s), save as orphan and reset
    elapsed = time.time() - st.session_state.awaiting_start_time
    if elapsed >= config["timeouts"]["orphan_timeout_seconds"]:
        first = st.session_state.first_page
        save_as_orphan(first.filepath, first.page_type, config)
        transition_to(READY)
        return

    # SIMPLE LOGIC: If we see a form, go to validation.
    # The validation step will reject it if it's still Page 1.
    if form_detected(frame, config):
        st.session_state.awaiting_consecutive_detections += 1
        if st.session_state.awaiting_consecutive_detections >= 2: # Fast transition
            transition_to(PROCESSING)
    else:
        st.session_state.awaiting_consecutive_detections = 0

def process_frame_done(config):
    if not st.session_state.ocr_task_launched and st.session_state.last_result:
        st.session_state.ocr_task_launched = True
        result = st.session_state.last_result
        import threading

        def _run_ocr():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                process_form_ocr(result["cardio"], result["strength"], config, EXAMPLES)
            )
            loop.close()

        t = threading.Thread(target=_run_ocr, daemon=True)
        t.start()
        logging.info("OCR background thread launched")

    if time.time() - st.session_state.done_start_time >= config["timeouts"]["success_display_seconds"]:
        cleanup_old_orphans(config)
        time.sleep(1)
        transition_to(READY)


# ---------------------------------------------------------------------------
# UI: STATUS DISPLAY
# ---------------------------------------------------------------------------


def get_status_display(state, config):
    """Returns (icon_svg, title, subtitle, border_color, bg_color, text_color, step) for each state."""

    if state == READY:
        count = st.session_state.consecutive_detections
        needed = config["state_transitions"]["form_detection_frames"]
        sub = f"Detection: {count}/{needed}" if count > 0 else "Place the document on the scanner"
        return "insert", "Insert Page", sub, "transparent", "#f8fafc", "#1e293b", 0

    if state == PROCESSING:
        count = st.session_state.consecutive_good
        needed = config["state_transitions"]["good_capture_frames"]
        side = "second side" if st.session_state.first_page else "first side"
        return "scanning", "Scanning...", f"Processing {side}", "#3b82f6", "#eff6ff", "#1e40af", 1

    if state == ISSUES:
        reason = st.session_state.issue_reason
        msg = ERROR_MESSAGES.get(reason, ERROR_MESSAGES["generic"])
        return "error", "Needs Repositioning", msg, "#f97316", "#fff7ed", "#c2410c", 1

    if state == AWAITING_SECOND_PAGE:
        elapsed = time.time() - st.session_state.awaiting_start_time
        remaining = max(0, config["timeouts"]["orphan_timeout_seconds"] - elapsed)
        if not st.session_state.awaiting_form_gone:
            sub = "Remove the page from the tray"
        else:
            sub = "Place the flipped page on the scanner"
        return "flip", "Flip the Page", sub, "#16a34a", "#f0fdf4", "#166534", 1

    if state == DONE:
        return "complete", "Scan Complete!", "Your document has been scanned", "#16a34a", "#f0fdf4", "#166534", 2

    return "insert", "Unknown State", "", "transparent", "#f8fafc", "#1e293b", 0


# ---------------------------------------------------------------------------
# UI: SVG ICONS (inline, no external dependencies)
# ---------------------------------------------------------------------------

ICONS = {
    "insert": """<svg width="96" height="96" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>
    </svg>""",
    "scanning": """<svg width="96" height="96" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="animation: spin 1.5s linear infinite;">
        <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
    </svg>""",
    "error": """<svg width="96" height="96" viewBox="0 0 24 24" fill="none" stroke="#f97316" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>""",
    "flip": """<svg width="96" height="96" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
        <path d="M9 15l3-3 3 3" opacity="0.5"/><path d="M9 11l3 3 3-3" opacity="0.5"/>
    </svg>""",
    "complete": """<svg width="96" height="96" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
    </svg>""",
}


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------


def run_kiosk():
    st.set_page_config(
        page_title="Fitness Form Kiosk",
        page_icon="💪",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_session_state()
    config = CONFIG

    # --- Full kiosk CSS ---
    st.markdown("""
    <style>
        /* Hide Streamlit chrome */
        #MainMenu, footer, header { visibility: hidden; }
        .stApp { background-color: #f1f5f9; }
        [data-testid="stToolbar"] { display: none; }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        /* Main status card */
        .kiosk-card {
            background: white;
            border-radius: 1.5rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04);
            padding: 3rem 2.5rem;
            min-height: 480px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            transition: border-color 0.3s ease;
            border: 6px solid transparent;
        }
        .kiosk-card-bordered {
            border: 6px solid var(--card-border);
        }

        .kiosk-icon {
            margin-bottom: 1.5rem;
        }
        .kiosk-title {
            font-size: 3rem;
            font-weight: 600;
            margin: 0 0 0.75rem 0;
            line-height: 1.2;
        }
        .kiosk-subtitle {
            font-size: 1.6rem;
            margin: 0;
            opacity: 0.75;
        }
        .kiosk-hint {
            font-size: 1.2rem;
            margin-top: 1.5rem;
            opacity: 0.5;
        }

        /* Progress dots */
        .progress-dots {
            display: flex;
            justify-content: center;
            gap: 0.75rem;
            margin-top: 2rem;
        }
        .progress-dot {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #cbd5e1;
            transition: background 0.3s ease;
        }
        .progress-dot.active {
            background: #3b82f6;
        }
        .progress-dot.done {
            background: #16a34a;
        }

        /* Camera feed container */
        .camera-container {
            border-radius: 1rem;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            background: #1e293b;
        }

        /* Debug bar */
        .debug-bar {
            background: #1e293b;
            color: #94a3b8;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-family: monospace;
            font-size: 0.85rem;
            margin-top: 0.75rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Layout: camera left, status right ---
    col_cam, col_status = st.columns([3, 2], gap="large")

    with col_cam:
        camera_placeholder = st.empty()

    with col_status:
        status_placeholder = st.empty()
        debug_placeholder = st.empty()

    # --- Open camera ---
    cap = cv2.VideoCapture(config["camera"]["camera_index"], cv2.CAP_DSHOW)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
    cap.set(cv2.CAP_PROP_FOCUS, 10) 

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # Manual Mode key

    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["capture_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["capture_height"])

    if not cap.isOpened():
        st.error("❌ Cannot open camera! Check that USB camera is connected and passed through to WSL.")
        st.info("Run `ls /dev/video*` in WSL to verify camera device is available.")
        logging.error("Camera failed to open")
        return

    logging.info(f"Camera opened: index={config['camera']['camera_index']} resolution={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")

    # --- Main loop ---
    while True:
        ret, frame_full = cap.read()
        if not ret:
            logging.warning("Camera read failed, retrying...")
            time.sleep(0.05)
            continue

        frame = frame_full
        state = st.session_state.state

        if state == READY:
            process_frame_ready(frame, config)
        elif state == PROCESSING:
            process_frame_processing(frame_full, frame, config)
        elif state == ISSUES:
            process_frame_issues(frame, config)
        elif state == AWAITING_SECOND_PAGE:
            process_frame_awaiting(frame, config)
        elif state == DONE:
            process_frame_done(config)

        # --- Get display data ---
        icon_key, title, subtitle, border_color, bg_color, text_color, step = get_status_display(st.session_state.state, config)
        icon_svg = ICONS.get(icon_key, ICONS["insert"])

        # --- Render camera feed ---
        if config["ui"]["show_camera_feed"]:
            display_frame = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)
            if config["ui"]["show_debug_overlay"]:
                cv2.putText(display_frame, f"State: {st.session_state.state}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            camera_placeholder.image(display_frame, channels="RGB", width="stretch")

        # --- Render status card ---
        border_style = f"border-color: {border_color};" if border_color != "transparent" else ""
        card_class = "kiosk-card kiosk-card-bordered" if border_color != "transparent" else "kiosk-card"

        # Determine which side we're on for the page label
        side_label = ""
        if st.session_state.state in (READY, PROCESSING, ISSUES) and st.session_state.first_page is None:
            side_label = "Side 1"
        elif st.session_state.state in (PROCESSING, ISSUES) and st.session_state.first_page is not None:
            side_label = "Side 2"
        elif st.session_state.state == AWAITING_SECOND_PAGE:
            side_label = "Side 2"

        status_html = f"""
<div class="{card_class}" style="background: {bg_color}; {border_style}">
    <div class="kiosk-icon">{icon_svg}</div>
    <h1 class="kiosk-title" style="color: {text_color};">{title}</h1>
    <p class="kiosk-subtitle" style="color: {text_color};">{subtitle}</p>
    {"<p class='kiosk-hint' style='color: " + text_color + ";'>" + side_label + "</p>" if side_label else ""}
</div>
"""
        
        status_placeholder.markdown(status_html, unsafe_allow_html=True)

        # --- Debug info ---
        if config["ui"]["show_debug_overlay"]:
            debug_placeholder.markdown(
                f'<div class="debug-bar">'
                f'State: {st.session_state.state} &nbsp;|&nbsp; '
                f'Detections: {st.session_state.consecutive_detections} &nbsp;|&nbsp; '
                f'Good: {st.session_state.consecutive_good}'
                f'</div>',
                unsafe_allow_html=True,
            )

        time.sleep(1.0 / config["camera"]["fps_target"])


if __name__ == "__main__":
    run_kiosk()