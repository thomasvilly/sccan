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
    print(f"Contours found: {len(contours)}")
    
    frame_area = frame.shape[0] * frame.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, fd["contour_approx_epsilon"] * perimeter, True)
        print(f"  Contour: points={len(approx)} area_ratio={cv2.contourArea(approx)/frame_area:.3f}") 

        if 4 <= len(approx) <= 8:
            area = cv2.contourArea(approx)
            area_ratio = area / frame_area

            if fd["area_ratio_min"] <= area_ratio <= fd["area_ratio_max"]:
                rect = cv2.minAreaRect(approx)
                w, h = rect[1]
                if min(w, h) == 0:
                    continue
                aspect = max(w, h) / min(w, h)
                print(f"  QUAD: area_ratio={area_ratio:.3f} aspect={aspect:.2f}")
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

import numpy as np

def is_duplicate_image(image_path_1, image_path_2, threshold=40):
    """
    Returns True if two images are visually similar (ignoring small movements).
    threshold: Lower = stricter (0 = identical, 255 = completely different).
    """
    try:
        # Load as grayscale
        img1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return False

        # Resize to small thumbnails (64x64) to ignore high-res noise
        # and minor alignment shifts
        img1_small = cv2.resize(img1, (64, 64))
        img2_small = cv2.resize(img2, (64, 64))

        # Calculate the Mean Absolute Difference between pixels
        diff = np.mean(np.abs(img1_small.astype("float") - img2_small.astype("float")))
        
        # If the difference is small (e.g. < 15), they are the same page
        logging.info(f"Duplicate Check: Difference Score = {diff:.2f}")
        same = diff < threshold
        print(same)
        return same
        
    except Exception as e:
        logging.error(f"Duplicate check failed: {e}")
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
    "no_form_detected": "📋 Please place your form in the tray",
    "image_blurry": "📸 Image is blurry — please hold still",
    "contrast_low": "💡 Lighting is too dim — please check the desk lamp",
    "glare_detected": "✨ Glare detected — please adjust the lamp angle",
    "page_type_unreadable": "❓ Cannot read form header — please reposition the form",
    "duplicate_page": "🔄 This page was already scanned — please flip to the other side",
    "generic": "⚠️ Please adjust your form and try again",
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def transition_to(new_state, **kwargs):
    old = st.session_state.state
    st.session_state.state = new_state
    logging.info(f"State: {old} → {new_state} {kwargs if kwargs else ''}")

    if new_state == READY:
        st.session_state.consecutive_detections = 0
        st.session_state.first_page = None
        st.session_state.ocr_task_launched = False
    elif new_state == PROCESSING:
        st.session_state.consecutive_good = 0
        st.session_state.consecutive_bad_start = None
        st.session_state.failure_counts = {}
        buffer_clear(st.session_state.frame_buffer)
    elif new_state == ISSUES:
        st.session_state.issue_reason = kwargs.get("reason", "generic")
        st.session_state.issue_consecutive_good = 0
    elif new_state == AWAITING_SECOND_PAGE:
        st.session_state.awaiting_start_time = time.time()
        st.session_state.awaiting_consecutive_detections = 0
        st.session_state.awaiting_form_gone = False
    elif new_state == DONE:
        st.session_state.done_start_time = time.time()
        st.session_state.ocr_task_launched = False


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
        
        # COMPARE: Is this just Page 1 again?
        # threshold=5 is strict. 
        #   >5 means "These are different images" (Good)
        #   <5 means "This is the same image" (Bad)
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
        
        # Send both to the LLM. 
        # We don't know which is cardio/strength, so we just label them 1 and 2.
        st.session_state.last_result = {
            "cardio": first.filepath,
            "strength": filepath
        }
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
    elapsed = time.time() - st.session_state.awaiting_start_time
    if elapsed >= config["timeouts"]["orphan_timeout_seconds"]:
        first = st.session_state.first_page
        save_as_orphan(first.filepath, first.page_type, config)
        transition_to(READY)
        return

    detected = form_detected(frame, config)

    # Phase 1: Wait for the form to DISAPPEAR (person lifts it to flip)
    if not st.session_state.awaiting_form_gone:
        if not detected:
            st.session_state.awaiting_form_gone = True
            logging.info("Form removed from tray — ready for second page")
        return  # Don't look for new form until old one is gone

    # Phase 2: Form was gone, now look for it to reappear
    if detected:
        st.session_state.awaiting_consecutive_detections += 1
        if st.session_state.awaiting_consecutive_detections >= config["state_transitions"]["form_detection_frames"]:
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
        transition_to(READY)


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------


def get_status_display(state, config):
    if state == READY:
        count = st.session_state.consecutive_detections
        needed = config["state_transitions"]["form_detection_frames"]
        sub = f"Detection: {count}/{needed}" if count > 0 else ""
        return "📋", "Place your form on the tray", sub, "blue"

    if state == PROCESSING:
        count = st.session_state.consecutive_good
        needed = config["state_transitions"]["good_capture_frames"]
        return "🔍", "Hold still... validating", f"Quality check: {count}/{needed} good frames", "orange"

    if state == ISSUES:
        reason = st.session_state.issue_reason
        msg = ERROR_MESSAGES.get(reason, ERROR_MESSAGES["generic"])
        return "⚠️", "Adjustment Needed", msg, "red"

    if state == AWAITING_SECOND_PAGE:
        first = st.session_state.first_page
        elapsed = time.time() - st.session_state.awaiting_start_time
        remaining = max(0, config["timeouts"]["orphan_timeout_seconds"] - elapsed)
        page_name = first.page_type.title() if first else "Page"
        return "✅", f"{page_name} page captured!", f"Please flip and scan the other side ({remaining:.0f}s remaining)", "green"

    if state == DONE:
        return "🎉", "Complete! Processing your form...", "Both pages captured successfully", "green"

    return "❓", "Unknown State", "", "gray"


def run_kiosk():
    st.set_page_config(
        page_title="Fitness Form Kiosk",
        page_icon="💪",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_session_state()
    config = CONFIG

    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp { background-color: #0e1117; }
        .status-box {
            padding: 2rem; border-radius: 1rem;
            text-align: center; margin: 1rem 0; font-size: 1.5rem;
        }
        .status-ready { background: #1e3a5f; color: #7eb8f7; }
        .status-processing { background: #3d3200; color: #ffc107; }
        .status-issues { background: #5f1e1e; color: #f77e7e; }
        .status-awaiting { background: #1e5f2e; color: #7ef79e; }
        .status-done { background: #1e5f2e; color: #7ef79e; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<h1 style='text-align:center; color:white;'>{config['ui']['title']}</h1>", unsafe_allow_html=True)

    col_cam, col_status = st.columns([3, 2])
    with col_cam:
        camera_placeholder = st.empty()
    with col_status:
        status_placeholder = st.empty()
        debug_placeholder = st.empty()
    
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

        emoji, title, msg, color = get_status_display(st.session_state.state, config)

        if config["ui"]["show_camera_feed"]:
            display_frame = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)
            if config["ui"]["show_debug_overlay"]:
                cv2.putText(display_frame, f"State: {st.session_state.state}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            camera_placeholder.image(display_frame, channels="RGB", width="stretch")

        css_class = f"status-{st.session_state.state.lower().replace('_', '-')}"
        if st.session_state.state == AWAITING_SECOND_PAGE:
            css_class = "status-awaiting"
        status_placeholder.markdown(f"""
        <div class="status-box {css_class}">
            <div style="font-size:3rem;">{emoji}</div>
            <div style="font-size:2rem; font-weight:bold; margin:0.5rem 0;">{title}</div>
            <div style="font-size:1.2rem;">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

        if config["ui"]["show_debug_overlay"]:
            debug_placeholder.text(
                f"State: {st.session_state.state} | "
                f"Detections: {st.session_state.consecutive_detections} | "
                f"Good frames: {st.session_state.consecutive_good}"
            )

        time.sleep(1.0 / config["camera"]["fps_target"])


if __name__ == "__main__":
    run_kiosk()
