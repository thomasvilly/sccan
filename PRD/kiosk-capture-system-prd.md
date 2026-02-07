# Product Requirements Document: Self-Service Fitness Form Capture Kiosk

**Version:** 1.1  
**Date:** February 7, 2026  
**Project:** Elderly Fitness Form Digitization System

---

## IMPORTANT IMPLEMENTATION NOTES

**For the developer (Opus):**

1. **Code Style:** Write FUNCTIONAL code. Avoid classes unless absolutely necessary (only for genuine primitives). Keep code in minimal files - prefer single `main.py` if possible.

2. **Error Handling:** Do NOT use try/except blocks. Let errors crash with full tracebacks. LOG EVERYTHING extensively.

3. **Configuration:** ALL tunable parameters (frame counts, thresholds, timeouts) MUST be in `config.toml` for easy testing of different configurations.

4. **Dependencies:** Use UV package manager. Works on Windows and Linux/WSL.

5. **API Flexibility:** Code should support testing different vision models via OpenRouter. Don't hardcode to one model.

6. **Decision Making:** Make implementation decisions where the PRD is ambiguous. Prefer simplicity and speed.

---

## 1. Overview

### 1.1 Purpose
Build a no-touch, camera-based self-service kiosk that allows elderly fitness clients to digitize their handwritten two-page weekly fitness forms after completing their exercise session. The system captures both pages (cardio log and strengthening exercises), validates photo quality in real-time using OpenCV, and sends images to vision-capable LLMs for structured data extraction.

### 1.2 Success Criteria
- Zero user input required (no buttons, no keyboard, no mouse)
- 95%+ successful capture rate on first attempt
- <10 second average capture time per page
- Handles messy handwriting, lighting variations, and form placement within physical tray
- Matches orphaned pages from incomplete previous sessions

### 1.3 Technical Stack
- **Language:** Python 3.10+
- **Package Manager:** UV
- **Camera:** USB webcam (1080p minimum)
- **Computer Vision:** OpenCV 4.x (full resolution processing, no downsampling)
- **OCR Options:** 
  - Primary: DeepSeek-OCR or similar vision models via OpenRouter
  - Alternative: Direct vision-capable LLMs (GPT-4o, Claude 3.5 Sonnet, etc.)
  - Page detection: Quick text search on raw image (no separate OCR engine needed)
- **API:** OpenRouter (unified interface for multiple LLM providers)
- **Display:** Streamlit (deployable to HuggingFace Spaces)
- **Physical Constraint:** CAD-designed tray ensures precise form placement

---

## 2. Form Specifications

### 2.1 Physical Form Details
- **Format:** Two-sided landscape document (11" × 8.5")
- **Page 1 (Cardio):** "CARDIO RECORDING LOG" header, workout tracking table
- **Page 2 (Strength):** "Strengthening Exercises" header, exercise checklist and stretches
- **Client Identifier:** "CCCARE ID" field (handwritten) appears at top of both pages
- **Reference:** See attached `weekly-start-fit.pdf`

### 2.2 Scan Requirements
- Both pages must be captured (order-independent: cardio→strength OR strength→cardio both valid)
- Pages must be matched by CCCARE ID
- Forms are always landscape orientation
- Users may scan pages minutes apart (orphaned page handling required)

---

## 3. System Architecture

### 3.1 State Machine

The system operates as a continuous loop with 5 states:

```
┌──────────────────────────────────────────────────────────────┐
│                         READY                                │
│  "Place your form on the surface"                           │
│  Camera: Running | Form Detection: Active                   │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 │ Form detected for 10 consecutive frames
                 ↓
┌──────────────────────────────────────────────────────────────┐
│                       PROCESSING                             │
│  "Hold still... validating"                                  │
│  Validation: 8 checks running | Buffer: Last 5 frames       │
└─────┬──────────────────────────┬─────────────────────────────┘
      │                          │
      │ 3 good frames            │ 3 seconds of failures
      ↓                          ↓
┌─────────────────┐      ┌──────────────────────────────────────┐
│  Capture Page   │      │           ISSUES                     │
│  Detect Type    │      │  "Please adjust your form..."        │
│  Check Orphans  │      │  Camera: Still running               │
└────┬────────────┘      └────────┬─────────────────────────────┘
     │                            │
     │                            │ 5 consecutive good frames
     │                            ↓
     │                      Return to PROCESSING
     │
     ↓
┌──────────────────────────────────────────────────────────────┐
│                  AWAITING_SECOND_PAGE                        │
│  "✓ [Page type] captured. Please flip and scan other side"  │
│  Timeout: 120 seconds → Save as orphan                       │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 │ Second form detected → PROCESSING → Capture
                 │ (if duplicate page detected → ISSUES)
                 ↓
┌──────────────────────────────────────────────────────────────┐
│                         DONE                                 │
│  "✓ Complete! Processing your form..."                      │
│  Send 3 async OCR API calls | Display: 3 seconds            │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 │ Auto-reset after 3 seconds
                 ↓
              READY (loop)
```

### 3.2 Data Flow

```
Camera (1080p, 15-20 FPS)
    ↓
Frame Buffer (in-memory, last 5 frames at 720p for OpenCV)
    ↓
OpenCV Validation (8 checks per frame)
    ↓
Capture Decision (3 consecutive good frames)
    ↓
Save Best Frame to Disk (select sharpest from buffer)
    ↓
Quick Page Type Detection (Tesseract on header)
    ↓
Check for Orphaned Pages (match opposite type from last 120s)
    ↓
After Both Pages Captured
    ↓
3 Async GPT-4o API Calls:
    1. Extract cardio page data
    2. Extract strength page data
    3. Reconcile & merge (handle ID mismatches)
    ↓
Return JSON to backend database
```

---

## 4. Detailed State Specifications

### 4.1 READY State

**Purpose:** Idle state, waiting for form to appear

**Display:**
- "Place your form on the surface"
- Live camera feed (optional, implementer choice)

**Camera:**
- Running at 15-20 FPS
- Capturing frames at 1080p
- Downscaling to 720p for OpenCV processing

**Logic:**
```python
while in READY:
    frame = capture_frame()  # 1080p
    frame_720p = downscale(frame, 720p)
    
    if form_detected(frame_720p):
        consecutive_detections += 1
        if consecutive_detections >= 10:
            transition_to(PROCESSING)
            initialize_frame_buffer()
    else:
        consecutive_detections = 0
```

**Form Detection Check:**
```python
def form_detected(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest quadrilateral
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) == 4:  # Quadrilateral found
            area = cv2.contourArea(approx)
            frame_area = frame.shape[0] * frame.shape[1]
            area_ratio = area / frame_area
            
            # Check size (60-80% of frame)
            if 0.60 <= area_ratio <= 0.80:
                # Check aspect ratio (landscape form ~1.4:1, allow 1.2-1.6)
                rect = cv2.minAreaRect(approx)
                width, height = rect[1]
                aspect_ratio = max(width, height) / min(width, height)
                
                if 1.2 <= aspect_ratio <= 1.6:
                    return True
    
    return False
```

**Transition Condition:**
- Form detected in 10 consecutive frames (~0.5-0.7 seconds at 15-20 FPS)

---

### 4.2 PROCESSING State

**Purpose:** Validate photo quality before capture

**Display:**
- "Hold still... validating"
- Progress indicator (optional)

**Frame Buffer:**
- Keep last 5 frames in memory (numpy arrays, no disk I/O)
- Tag each frame with validation scores
- Rolling window: oldest frame drops when new frame arrives

**Validation Checks (simplified - physical tray constrains placement):**

Since forms are placed in a CAD-designed tray with exact positioning guides, most spatial checks are unnecessary. Focus on image quality only:

```python
def validate_frame(frame):
    """
    Run simplified validation checks on frame.
    Returns: (pass: bool, failure_reason: str, scores: dict)
    """
    scores = {}
    
    # 1. Basic form presence check
    # Since tray constrains placement, we just need to detect *something* in frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean()
    if mean_intensity < 50:  # Too dark, probably no form
        return False, "no_form_detected", scores
    scores['mean_intensity'] = mean_intensity
    
    # 2. Blur Check (Laplacian variance > threshold)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        return False, "image_blurry", scores
    scores['sharpness'] = laplacian_var
    
    # 3. Contrast Check (ensure text is visible)
    contrast = gray.std()
    if contrast < CONTRAST_THRESHOLD:
        return False, "contrast_low", scores
    scores['contrast'] = contrast
    
    # 4. Glare Check (no blown-out regions)
    blown_pixels = np.sum(gray > 250)
    total_pixels = gray.size
    glare_ratio = blown_pixels / total_pixels
    if glare_ratio > GLARE_THRESHOLD:
        return False, "glare_detected", scores
    scores['glare_ratio'] = glare_ratio
    
    return True, None, scores
```

**Note:** Rotation, size, and orientation checks removed since physical tray enforces correct placement.

**Logic:**
```python
consecutive_good = 0
consecutive_bad = 0
failure_counts = {}  # Track failure reasons
start_time = time.time()

while in PROCESSING:
    frame_1080p = capture_frame()
    frame_720p = downscale(frame_1080p, 720p)
    
    passed, failure_reason, scores = validate_frame(frame_720p)
    
    if passed:
        # Add to buffer with score
        frame_buffer.add(frame_1080p, scores['sharpness'])
        consecutive_good += 1
        consecutive_bad = 0
        
        if consecutive_good >= 3:
            # CAPTURE!
            best_frame = frame_buffer.get_sharpest()
            transition_to_capture(best_frame)
    else:
        # Track failures
        failure_counts[failure_reason] = failure_counts.get(failure_reason, 0) + 1
        consecutive_good = 0
        consecutive_bad += 1
        
        # Timeout check (3 seconds)
        if time.time() - start_time > 3.0:
            most_common_failure = max(failure_counts, key=failure_counts.get)
            transition_to(ISSUES, most_common_failure)
```

**Transition Conditions:**
- **To capture:** 3 consecutive frames pass all validation checks
- **To ISSUES:** 3 seconds elapse with continuous validation failures

**Capture Action:**
```python
def transition_to_capture(best_frame):
    # Save to disk
    timestamp = int(time.time())
    filename = f"page_{timestamp}.jpg"
    cv2.imwrite(filename, best_frame)
    
    # Detect page type (quick Tesseract OCR on header)
    page_type = detect_page_type(filename)  # returns "cardio" or "strength"
    
    if page_type is None:
        # Can't determine page type
        os.remove(filename)
        transition_to(ISSUES, "page_type_unreadable")
        return
    
    # Check for orphaned pages
    orphan_match = find_orphan_match(page_type)
    
    if orphan_match:
        # Found matching orphan! Skip to DONE with both pages
        transition_to(DONE, [orphan_match, filename])
    else:
        # Save this as first page, wait for second
        save_first_page(filename, page_type)
        transition_to(AWAITING_SECOND_PAGE, page_type)
```

---

### 4.3 ISSUES State

**Purpose:** Show user what to fix

**Display:**
- Error message based on failure reason
- Camera feed still visible (optional, for context)

**Error Messages (simplified):**
```python
ERROR_MESSAGES = {
    "no_form_detected": "Please place form in tray",
    "image_blurry": "Please ensure camera is steady",
    "contrast_low": "Please check lighting",
    "glare_detected": "Please adjust lighting to reduce glare",
    "page_type_unreadable": "Cannot read form. Please ensure form is properly placed",
    "duplicate_page": "Already scanned this page. Please scan the other side",
    "generic": "Please adjust your form and try again"
}
```

**Logic:**
```python
consecutive_good = 0

while in ISSUES:
    frame_1080p = capture_frame()
    frame_720p = downscale(frame_1080p, 720p)
    
    passed, failure_reason, scores = validate_frame(frame_720p)
    
    if passed:
        consecutive_good += 1
        if consecutive_good >= 5:
            # Issue resolved!
            transition_to(PROCESSING)
    else:
        consecutive_good = 0
```

**Transition Condition:**
- 5 consecutive frames pass all validation checks

---

### 4.4 AWAITING_SECOND_PAGE State

**Purpose:** First page captured, waiting for user to flip form

**Display:**
- "✓ [Cardio/Strength] page captured. Please flip and scan the other side"

**Timeout:**
- 120 seconds (2 minutes)
- If no second page captured, save first page as orphan

**Logic:**
```python
first_page_file = current_captured_page
first_page_type = current_page_type  # "cardio" or "strength"
start_time = time.time()

while in AWAITING_SECOND_PAGE:
    # Check timeout
    if time.time() - start_time > 120:
        # Save as orphan
        orphan_filename = f"orphan_{first_page_type}_{int(time.time())}.jpg"
        os.rename(first_page_file, orphan_filename)
        transition_to(READY)
        break
    
    # Wait for form detection (same as READY state)
    frame = capture_frame()
    frame_720p = downscale(frame, 720p)
    
    if form_detected(frame_720p):
        consecutive_detections += 1
        if consecutive_detections >= 10:
            # Transition to PROCESSING for second page
            transition_to(PROCESSING)
    else:
        consecutive_detections = 0
```

**After Second Page Captured:**
```python
def handle_second_page_capture(second_page_file):
    second_page_type = detect_page_type(second_page_file)
    
    if second_page_type is None:
        # Can't read page type
        os.remove(second_page_file)
        transition_to(ISSUES, "page_type_unreadable")
        return
    
    # Check for duplicate
    if second_page_type == first_page_type:
        # User scanned same side twice!
        os.remove(second_page_file)
        transition_to(ISSUES, "duplicate_page")
        # Stay in AWAITING_SECOND_PAGE, don't reset
        return
    
    # Success! Both pages captured
    transition_to(DONE, [first_page_file, second_page_file])
```

---

### 4.5 DONE State

**Purpose:** Both pages captured, send to OCR pipeline

**Display:**
- "✓ Complete! Processing your form..."
- Success animation (optional)

**OCR Pipeline:**

The system makes **3 API calls** to OpenRouter (can use different models):

```python
import httpx
import base64

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model options to try (configure in config file)
# - openai/gpt-4o
# - anthropic/claude-3.5-sonnet
# - deepseek/deepseek-ocr (if available)
# - google/gemini-2.0-flash-001

async def process_form_ocr(cardio_image_path, strength_image_path, example_data):
    """
    Send 3 API calls via OpenRouter:
    1. Extract cardio page (with few-shot examples)
    2. Extract strength page (with few-shot examples)
    3. Reconcile and merge
    
    example_data: dict with ground truth examples for few-shot prompting
    """
    
    # Encode images
    cardio_b64 = encode_image_base64(cardio_image_path)
    strength_b64 = encode_image_base64(strength_image_path)
    
    # Build prompts with few-shot examples
    cardio_prompt = build_cardio_prompt_with_examples(example_data['cardio_examples'])
    strength_prompt = build_strength_prompt_with_examples(example_data['strength_examples'])
    
    # Call 1 & 2: Extract each page (can run in parallel)
    async with httpx.AsyncClient() as client:
        cardio_task = extract_page(client, cardio_b64, cardio_prompt, CARDIO_SCHEMA, "cardio")
        strength_task = extract_page(client, strength_b64, strength_prompt, STRENGTH_SCHEMA, "strength")
        
        cardio_data, strength_data = await asyncio.gather(cardio_task, strength_task)
    
    # Call 3: Reconcile
    final_data = await reconcile_pages(cardio_data, strength_data, cardio_b64, strength_b64)
    
    return final_data

async def extract_page(client, image_b64, prompt, schema, page_type):
    """Generic page extraction via OpenRouter"""
    
    payload = {
        "model": MODEL_CONFIG[f'{page_type}_extraction_model'],  # From config
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"{page_type}_page",
                "schema": schema,
                "strict": True
            }
        }
    }
    
    response = await client.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/yourusername/fitness-kiosk",  # Required by OpenRouter
            "X-Title": "Fitness Form Kiosk"
        },
        timeout=30.0
    )
    
    response.raise_for_status()
    result = response.json()
    
    logging.info(f"Extracted {page_type} page successfully")
    logging.debug(f"Raw response: {result}")
    
    return json.loads(result['choices'][0]['message']['content'])

def build_cardio_prompt_with_examples(examples):
    """
    Build few-shot prompt with ground truth examples.
    
    examples: list of dicts with 'image_description' and 'expected_json'
    """
    prompt = """Extract data from this CARDIO RECORDING LOG fitness form.

The form contains:
- CCCARE ID (handwritten, top of page)
- Target HR and RPE values
- Equipment settings (NuStep arms/seat, bike seats, etc.)
- Multiple workout sessions in a table with columns:
  - Date
  - Time (total minutes)
  - RPE (6-20 scale)
  - Watch #
  - Activity (NS/RB/UB/TM/E/ROW/LAPS)
  - Work Rate / Speed / Elevation
  - Heart Rate Range
  - Comments

Common handwriting issues:
- Number 1 looks like 7
- Number 5 looks like 6
- Number 0 looks like 6

Here are examples of correctly extracted data:

"""
    
    # Add few-shot examples
    for i, example in enumerate(examples, 1):
        prompt += f"\nExample {i}:\n"
        prompt += f"{example['image_description']}\n"
        prompt += f"Correct extraction:\n{json.dumps(example['expected_json'], indent=2)}\n"
    
    prompt += "\n\nNow extract from this new form. Return data as JSON matching the schema. Empty fields should be null.\n"
    
    return prompt

def build_strength_prompt_with_examples(examples):
    """Similar to cardio but for strength page"""
    prompt = """Extract data from this STRENGTHENING EXERCISES fitness form.

The form contains:
- CCCARE ID (handwritten, top of page)
- Date (2026)
- Exercise table with columns: Exercise name, Reps, Weight (Wt)
- Multiple sets per exercise (up to 4 columns of reps/weight)
- Stretches checklist at bottom (7 types of stretches)

Exercises listed:
1. Squats, 2. Chest press, 3. Bent knee hip raise, 4. Vertical traction
5. Airplane, 6. Front bridge, 7. Leg lowers, 8. Thoracic rotation, 9. Hip mobility

Stretches: Quad, Hamstring, Glute, Hip flexor, Calf, Chest, Upper back

Here are examples of correctly extracted data:

"""
    
    for i, example in enumerate(examples, 1):
        prompt += f"\nExample {i}:\n"
        prompt += f"{example['image_description']}\n"
        prompt += f"Correct extraction:\n{json.dumps(example['expected_json'], indent=2)}\n"
    
    prompt += "\n\nNow extract from this new form. Return only checked stretches.\n"
    
    return prompt

async def reconcile_pages(cardio_data, strength_data, cardio_img, strength_img):
    """Third API call: Reconcile data, handle CCCARE ID mismatches"""
    
    prompt = f"""You are reconciling data from two pages of the same fitness form.

Cardio page data: {json.dumps(cardio_data, indent=2)}
Strength page data: {json.dumps(strength_data, indent=2)}

Check CCCARE IDs:
- Cardio: {cardio_data.get('cccare_id')}
- Strength: {strength_data.get('cccare_id')}

If IDs differ, examine both images and choose the correct one based on handwriting clarity.

Return merged data with single correct CCCARE ID, confidence score, and flag if IDs mismatched.
"""
    
    payload = {
        "model": MODEL_CONFIG['reconciliation_model'],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cardio_img}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{strength_img}"}}
            ]
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "merged_form", "schema": MERGED_SCHEMA, "strict": True}
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://github.com/yourusername/fitness-kiosk",
                "X-Title": "Fitness Form Kiosk"
            },
            timeout=30.0
        )
    
    response.raise_for_status()
    result = response.json()
    
    logging.info("Reconciliation complete")
    
    return json.loads(result['choices'][0]['message']['content'])
```

**Few-Shot Example Data Structure:**

Store ground truth examples in a JSON file:

```json
{
  "cardio_examples": [
    {
      "image_description": "Form with CCCARE ID '12345', Date 1/15/26, Time 30min, RPE 12, Activity NS, Work Rate Level 8",
      "expected_json": {
        "cccare_id": "12345",
        "sessions": [{
          "date": "2026-01-15",
          "time_minutes": 30,
          "rpe": 12,
          "activity": "NS",
          "work_rate": "Level 8"
        }]
      }
    }
  ],
  "strength_examples": [
    {
      "image_description": "Form with CCCARE ID '12345', Squats: 12 reps @ 50lbs, Chest press: 10 reps @ 30lbs",
      "expected_json": {
        "cccare_id": "12345",
        "exercises": [
          {"exercise_name": "Squats", "sets": [{"reps": 12, "weight": 50}]},
          {"exercise_name": "Chest press", "sets": [{"reps": 10, "weight": 30}]}
        ]
      }
    }
  ]
}
```

Load this file at startup and pass to OCR pipeline.

**Transition:**
```python
def transition_to_done(page_files):
    # Start async OCR (don't wait for it)
    asyncio.create_task(process_form_ocr(page_files[0], page_files[1]))
    
    # Display success for 3 seconds
    display_success_message()
    time.sleep(3)
    
    # Clean up
    cleanup_frame_buffer()
    
    # Reset to READY
    transition_to(READY)
```

---

## 5. Page Type Detection

**Purpose:** Determine if captured page is cardio or strength

**Method:** Quick vision model query or simple text search on image

**Option 1 (Recommended): Simple text search on image**
```python
import cv2
import numpy as np

def detect_page_type(image_path):
    """
    Fast page type detection using OpenCV text region detection.
    Returns: "cardio", "strength", or None
    """
    img = cv2.imread(image_path)
    
    # Crop to header region (top 15% of image)
    height = img.shape[0]
    header = img[0:int(height * 0.15), :]
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Look for characteristic patterns
    # Cardio page has dense text in header
    # Strength page has more structured table layout
    
    # Simple heuristic: count dark pixels in header
    # Cardio header is text-heavy, strength header is sparser
    dark_pixel_ratio = np.sum(binary == 0) / binary.size
    
    if dark_pixel_ratio > 0.3:
        return "cardio"  # Dense header text
    elif dark_pixel_ratio > 0.1:
        return "strength"  # Sparser header
    else:
        return None  # Can't determine
    
    # Alternative: Use template matching if needed
    # (load template images of known headers and use cv2.matchTemplate)

**Option 2: Quick vision model call**
```python
async def detect_page_type_with_llm(image_path):
    """
    Use vision model for robust page detection.
    Fast single-token response.
    """
    image_b64 = encode_image_base64(image_path)
    
    response = await openrouter_client.chat.completions.create(
        model=PAGE_DETECTION_MODEL,  # Fast model like gpt-4o-mini or similar
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Is this a CARDIO recording log or STRENGTHENING exercises page? Answer with single word: 'cardio' or 'strength'"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]
        }],
        max_tokens=5
    )
    
    answer = response.choices[0].message.content.strip().lower()
    if "cardio" in answer:
        return "cardio"
    elif "strength" in answer:
        return "strength"
    else:
        return None
```

**Note:** Choose Option 1 (OpenCV) for speed, Option 2 (LLM) for reliability. Can try both and use LLM as fallback.

---

## 6. Orphan Page Management

**Purpose:** Handle cases where user scans one page and walks away

### 6.1 Creating Orphans

When timeout occurs in AWAITING_SECOND_PAGE state (120 seconds):

```python
def save_as_orphan(page_file, page_type):
    timestamp = int(time.time())
    orphan_file = f"orphan_{page_type}_{timestamp}.jpg"
    os.rename(page_file, orphan_file)
    
    # Log orphan metadata
    orphan_metadata = {
        "filename": orphan_file,
        "page_type": page_type,
        "timestamp": timestamp,
        "created_at": datetime.now().isoformat()
    }
    save_orphan_metadata(orphan_metadata)
```

### 6.2 Finding Orphan Matches

When new page is captured, check for orphans:

```python
def find_orphan_match(new_page_type):
    """
    Look for orphaned page of opposite type from last 120 seconds.
    Returns: orphan filename if found, None otherwise
    """
    cutoff_time = time.time() - 120  # 2 minutes ago
    
    # Determine what we're looking for
    needed_type = "strength" if new_page_type == "cardio" else "cardio"
    
    # Search orphan directory
    orphan_files = glob.glob(f"orphan_{needed_type}_*.jpg")
    
    for orphan_file in orphan_files:
        # Extract timestamp from filename
        parts = orphan_file.split('_')
        orphan_timestamp = int(parts[-1].replace('.jpg', ''))
        
        if orphan_timestamp >= cutoff_time:
            # Found a match!
            return orphan_file
    
    return None
```

---

## 7. JSON Schemas for Vision Models

### 7.1 Cardio Page Schema

```python
CARDIO_SCHEMA = {
    "type": "object",
    "properties": {
        "cccare_id": {"type": "string"},
        "target_hr": {"type": "string"},
        "target_rpe": {"type": "string"},
        "equipment_settings": {
            "type": "object",
            "properties": {
                "nustep_arms": {"type": "string"},
                "nustep_seat": {"type": "string"},
                "leg_stab": {"type": "string"},
                "recumbent_bike_seat": {"type": "string"},
                "upright_bike_seat": {"type": "string"}
            }
        },
        "sessions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "time_minutes": {"type": "integer"},
                    "rpe": {"type": "integer"},
                    "watch_number": {"type": "string"},
                    "activity": {"type": "string"},
                    "work_rate": {"type": "string"},
                    "heart_rate_range": {"type": "string"},
                    "comments": {"type": "string"}
                }
            }
        }
    },
    "required": ["cccare_id", "sessions"]
}

### 7.2 Strength Page Schema

```python
STRENGTH_SCHEMA = {
    "type": "object",
    "properties": {
        "cccare_id": {"type": "string"},
        "date": {"type": "string"},
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
                                "reps": {"type": "integer"},
                                "weight": {"type": "integer"}
                            }
                        }
                    }
                }
            }
        },
        "stretches_completed": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["cccare_id", "exercises"]
}

### 7.3 Merged Schema

```python
MERGED_SCHEMA = {
    "type": "object",
    "properties": {
        "cccare_id": {"type": "string", "description": "Final chosen CCCARE ID"},
        "id_mismatch": {"type": "boolean", "description": "True if IDs differed between pages"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "cardio_data": {"type": "object"},  # Embed full cardio schema
        "strength_data": {"type": "object"},  # Embed full strength schema
        "notes": {"type": "string", "description": "Any reconciliation notes or warnings"}
    },
    "required": ["cccare_id", "id_mismatch", "confidence", "cardio_data", "strength_data"]
}
```

---

## 8. Configuration (config.toml or config.json)

**ALL tunable parameters in a single configuration file for easy testing:**

```toml
# config.toml

[camera]
camera_index = 0
capture_width = 1920
capture_height = 1080
fps_target = 20
# No downsampling - process at full resolution for speed

[state_transitions]
# CRITICAL: These are the main knobs to tune during testing
form_detection_frames = 10      # READY → PROCESSING
good_capture_frames = 3         # PROCESSING → capture
issue_retry_frames = 5          # ISSUES → PROCESSING

[timeouts]
issue_timeout_seconds = 3.0     # PROCESSING → ISSUES
orphan_timeout_seconds = 120    # AWAITING → save orphan
success_display_seconds = 3     # DONE → READY

[opencv_thresholds]
# Image quality thresholds
blur_threshold = 100            # Laplacian variance (lower = more blur accepted)
contrast_threshold = 30         # Grayscale std dev
glare_threshold = 0.05          # Max 5% blown-out pixels
mean_intensity_min = 50         # Minimum brightness (detect if form present)

[buffer]
frame_buffer_size = 5           # Keep last N frames in memory

[paths]
capture_dir = "./captures"
orphan_dir = "./orphans"
temp_dir = "./temp"
example_data_file = "./few_shot_examples.json"  # Ground truth examples

[api]
openrouter_api_key_env = "OPENROUTER_API_KEY"  # Environment variable name
openrouter_base_url = "https://openrouter.ai/api/v1"

[models]
# Choose models for each task - can test different combinations
cardio_extraction_model = "openai/gpt-4o"
strength_extraction_model = "openai/gpt-4o"
reconciliation_model = "openai/gpt-4o"
page_detection_model = "openai/gpt-4o-mini"  # Fast/cheap for page type

# Alternative models to try:
# "anthropic/claude-3.5-sonnet"
# "google/gemini-2.0-flash-001"
# "deepseek/deepseek-ocr" (if available)

[logging]
log_level = "INFO"  # DEBUG, INFO, WARNING, ERROR
log_file = "kiosk.log"
log_to_console = true
```

**Usage in code:**
```python
import tomli  # or json for .json config

def load_config(config_path="config.toml"):
    with open(config_path, "rb") as f:
        config = tomli.load(f)
    
    logging.info(f"Loaded config: {config}")
    return config

# Access anywhere
CONFIG = load_config()
BLUR_THRESHOLD = CONFIG['opencv_thresholds']['blur_threshold']
FORM_DETECTION_FRAMES = CONFIG['state_transitions']['form_detection_frames']
```

---

## 9. Implementation Guidelines

### 9.1 Code Structure (Functional, Minimal Files)

**Preferred:** Single file or minimal file split

```
kiosk_capture/
├── main.py                 # Main entry point + all state machine logic (prefer this)
├── config.toml             # Configuration file
├── few_shot_examples.json  # Ground truth examples for prompting
├── schemas.json            # JSON schemas (or inline in main.py)
├── captures/               # Saved page images
├── orphans/                # Orphaned pages
└── kiosk.log              # Log output
```

**Alternative if needed:**
```
kiosk_capture/
├── main.py                 # Entry + state machine
├── validators.py           # OpenCV validation functions (if reused heavily)
├── api_client.py           # OpenRouter API calls (if complex)
├── config.toml
└── ...
```

### 9.2 Code Style

**Prefer:**
- **Functional style** - pure functions with clear inputs/outputs
- **Minimal abstraction** - avoid classes unless genuinely needed
- **Inline small functions** - don't split if <20 lines
- **Direct, readable code** over clever patterns

**Example of preferred style:**
```python
def validate_frame(frame, blur_thresh, contrast_thresh, glare_thresh):
    """Pure function - no side effects"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Check 1: Blur
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < blur_thresh:
        return False, "blurry", {"blur": blur_score}
    
    # Check 2: Contrast
    contrast = gray.std()
    if contrast < contrast_thresh:
        return False, "low_contrast", {"contrast": contrast}
    
    # Check 3: Glare
    glare_ratio = (gray > 250).sum() / gray.size
    if glare_ratio > glare_thresh:
        return False, "glare", {"glare": glare_ratio}
    
    return True, None, {"blur": blur_score, "contrast": contrast, "glare": glare_ratio}

# Use directly
passed, reason, scores = validate_frame(frame, CONFIG['blur_threshold'], ...)
```

**Avoid (unless necessary):**
```python
class FrameValidator:
    def __init__(self, config):
        self.blur_threshold = config['blur_threshold']
        # ...
    
    def validate(self, frame):
        # ...
```

**Exception:** A simple dataclass or named tuple is fine for grouping data:
```python
from collections import namedtuple

CapturedPage = namedtuple('CapturedPage', ['filename', 'page_type', 'timestamp'])
```

### 9.3 Error Handling - NO TRY/EXCEPT, LOG EVERYTHING

**Do NOT use try/except blocks.** Let errors crash and log them.

```python
# WRONG - don't do this
try:
    result = validate_frame(frame)
except Exception as e:
    logging.error(f"Error: {e}")
    result = None

# RIGHT - let it crash, log before operations
logging.debug(f"Validating frame shape={frame.shape}")
result = validate_frame(frame)  # If this fails, we want to see the traceback
logging.info(f"Validation result: {result}")
```

**Why:** In a kiosk, crashes are better than silent failures. Logs tell us exactly what went wrong.

**Logging everywhere:**
```python
logging.info(f"State: READY → PROCESSING (detected for {consecutive_frames} frames)")
logging.info(f"Captured page: {page_type}, saved to {filename}")
logging.warning(f"Created orphan: {orphan_file}")
logging.debug(f"Frame validation scores: {scores}")
logging.info(f"OCR call started for {page_type}")
logging.info(f"OCR completed in {elapsed:.2f}s")
```

### 9.4 Streamlit Integration

Use Streamlit for UI, deployable to HuggingFace Spaces:

```python
import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer

# Minimal Streamlit UI
st.title("Fitness Form Capture Kiosk")

# Camera feed
camera_feed = st.empty()
status_text = st.empty()
message_box = st.empty()

# State
if 'state' not in st.session_state:
    st.session_state.state = "READY"
    st.session_state.consecutive_frames = 0

# Main loop (use st.experimental_rerun for state updates)
cap = cv2.VideoCapture(CONFIG['camera']['camera_index'])

while True:
    ret, frame = cap.read()
    
    # Process frame through state machine
    new_state, display_msg = process_frame(frame, st.session_state.state)
    
    # Update UI
    camera_feed.image(frame, channels="BGR")
    status_text.text(f"Status: {new_state}")
    message_box.info(display_msg)
    
    # Update state
    if new_state != st.session_state.state:
        st.session_state.state = new_state
        logging.info(f"State change: → {new_state}")
```

**Note:** Streamlit's architecture may require adapting the continuous loop. Use `st.experimental_rerun()` or WebRTC streaming components.

---

## 10. Testing Requirements

### 10.1 Unit Tests

Required test coverage:

```python
# validators.py
def test_form_detection_valid():
    """Test that valid form is detected"""
    pass

def test_form_detection_no_form():
    """Test that blank image returns False"""
    pass

def test_blur_detection():
    """Test blur threshold with sharp vs blurry images"""
    pass

def test_glare_detection():
    """Test glare detection with synthetic glare"""
    pass

# page_detector.py
def test_cardio_page_detection():
    """Test that cardio page is correctly identified"""
    pass

def test_strength_page_detection():
    """Test that strength page is correctly identified"""
    pass

def test_unreadable_page():
    """Test that blank/unreadable page returns None"""
    pass

# orphan_manager.py
def test_orphan_creation():
    """Test that orphans are saved correctly"""
    pass

def test_orphan_matching():
    """Test that opposite-type orphan is found"""
    pass

def test_orphan_cleanup():
    """Test that old orphans are deleted"""
    pass
```

### 10.2 Integration Tests

Test full state machine flows:

1. **Happy path:** READY → PROCESSING → capture → AWAITING → PROCESSING → capture → DONE → READY
2. **Blur recovery:** PROCESSING → ISSUES (blur) → PROCESSING → capture
3. **Duplicate page:** Capture cardio → try cardio again → ISSUES (duplicate) → capture strength
4. **Orphan match:** Capture cardio → timeout → orphan saved → new user scans strength → no match → capture cardio → match found → DONE
5. **Timeout:** Capture cardio → wait 120s → orphan saved → READY

### 10.3 Performance Tests

Measure:
- FPS achieved (target: 15-20)
- Frame processing time (target: <50ms per frame at 720p)
- State transition latency (target: <100ms)
- Memory usage (target: <500MB)
- API call latency (log for monitoring, not blocking)

---

## 11. Deployment Requirements

### 11.1 Hardware

**Minimum:**
- Laptop/PC with USB port
- CPU: Intel i5 or AMD equivalent (for OpenCV at full resolution)
- RAM: 8GB minimum (16GB recommended)
- USB webcam: 1080p, 30fps capable
- Display: 1920×1080 minimum
- **Physical tray:** CAD-designed form placement guide (ensures correct positioning)
- Desk lamp for consistent lighting

**Recommended:**
- CPU: Intel i7 or AMD equivalent
- RAM: 16GB
- SSD: For faster I/O
- GPU: Optional, not required (OpenCV uses CPU, vision models are API calls)

### 11.2 Software Dependencies

**Using UV (fast Python package manager):**

```bash
# Install UV (if not already installed)
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/WSL:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project with UV
uv init kiosk_capture
cd kiosk_capture

# Add dependencies
uv add opencv-python pillow numpy httpx streamlit tomli
uv add --dev pytest

# Or use pyproject.toml:
```

**pyproject.toml:**
```toml
[project]
name = "kiosk-capture"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "opencv-python>=4.8.1",
    "pillow>=10.1.0",
    "numpy>=1.24.3",
    "httpx>=0.25.0",          # For OpenRouter API
    "streamlit>=1.28.0",       # UI framework
    "tomli>=2.0.1",           # TOML config parsing
    "python-dotenv>=1.0.0",   # Environment variables
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
]
```

**NO Tesseract required** - using vision models for all OCR

### 11.3 Environment Variables

```bash
# .env file
OPENROUTER_API_KEY=sk-or-v1-...your-key-here...
```

### 11.4 Installation

**Windows:**
```powershell
# Clone/create project
mkdir kiosk_capture
cd kiosk_capture

# Initialize with UV
uv init
uv add opencv-python pillow numpy httpx streamlit tomli python-dotenv

# Create directories
New-Item -ItemType Directory -Force captures, orphans

# Set environment variables
Copy-Item .env.example .env
# Edit .env with your OpenRouter API key

# Run
uv run streamlit run main.py
```

**Linux/WSL:**
```bash
# Clone/create project
mkdir kiosk_capture && cd kiosk_capture

# Initialize with UV
uv init
uv add opencv-python pillow numpy httpx streamlit tomli python-dotenv

# Create directories
mkdir -p captures orphans

# Set environment variables
cp .env.example .env
# Edit .env with your OpenRouter API key

# Run
uv run streamlit run main.py
```

### 11.5 HuggingFace Spaces Deployment

**Create `app.py` (entry point for HF Spaces):**
```python
# app.py
import streamlit as st
from main import run_kiosk

# HuggingFace Spaces expects app.py
if __name__ == "__main__":
    run_kiosk()
```

**Create `requirements.txt` for HF Spaces:**
```txt
opencv-python-headless==4.8.1
pillow==10.1.0
numpy==1.24.3
httpx==0.25.0
streamlit==1.28.0
tomli==2.0.1
python-dotenv==1.0.0
```

**Add `packages.txt` if system dependencies needed:**
```txt
# Empty for this project - no system packages needed
```

**HF Spaces config (add as repository secrets):**
- `OPENROUTER_API_KEY`: Your OpenRouter API key

---

## 12. Success Metrics

Track these KPIs:

1. **Capture success rate:** % of sessions resulting in successful 2-page capture
   - Target: >95%

2. **Average capture time:** Seconds from form placement to DONE state
   - Target: <20 seconds for both pages

3. **Issue state frequency:** % of sessions entering ISSUES state
   - Target: <10%

4. **Orphan rate:** % of sessions creating orphaned pages
   - Target: <5%

5. **OCR accuracy:** % of forms with high-confidence extraction
   - Target: >90% (measured from LLM confidence scores)

6. **User abandonment:** % of sessions where user walks away mid-capture
   - Target: <3%

---

## 13. Known Limitations & Future Enhancements

### 13.1 Current Limitations

- **Single user at a time:** If two people try to use kiosk simultaneously, second must wait
- **Lighting dependent:** Requires decent overhead/desk lighting (mitigated by desk lamp)
- **Form must be reasonably flat:** Severely wrinkled/curled forms may fail validation
- **Physical tray required:** CAD tray must be manufactured for proper form placement
- **CCCARE ID mismatch relies on vision model:** No separate validation logic for ID matching

### 13.2 Future Enhancements

- **Audio feedback:** Beeps/voice prompts for state transitions (to be integrated by team)

---

## 14. Appendix: Full State Diagram

```
                    ┌─────────────────────────┐
                    │        READY            │
                    │  Waiting for form       │
                    └───────────┬─────────────┘
                                │
                                │ Form detected
                                │ 10 frames
                                ↓
                    ┌─────────────────────────┐
            ┌──────→│      PROCESSING         │←──────┐
            │       │  Validating quality     │       │
            │       └──┬──────────────────┬───┘       │
            │          │                  │           │
            │          │ 3 good           │ 3s failed │
            │          │ frames           │           │
  5 good    │          ↓                  ↓           │
  frames    │    ┌──────────┐      ┌─────────────┐   │
            │    │ CAPTURE  │      │   ISSUES    │───┘
            └────│ Page 1   │      │ Show error  │
                 └────┬─────┘      └─────────────┘
                      │
                      │ Detect page type
                      │ Check orphans
                      ↓
         ┌────────────┴──────────────┐
         │                           │
         │ Orphan match?             │ No match
         ↓ YES                       ↓
    ┌─────────┐         ┌──────────────────────────┐
    │  DONE   │         │  AWAITING_SECOND_PAGE    │
    │ (both)  │         │  "Flip and scan back"    │
    └────┬────┘         └────┬────────────┬────────┘
         │                   │            │
         │                   │ Form       │ 120s timeout
         │                   │ detected   │
         │                   │ 10 frames  ↓
         │                   ↓         ┌──────────┐
         │              PROCESSING    │ Save as  │
         │              (page 2)      │ orphan   │
         │                   │        └────┬─────┘
         │                   │ 3 good      │
         │                   │ frames      │
         │                   ↓             │
         │              ┌─────────┐        │
         │              │ CAPTURE │        │
         │              │ Page 2  │        │
         │              └────┬────┘        │
         │                   │             │
         │                   │ Check       │
         │                   │ duplicate?  │
         │                   ↓             │
         │              ┌─────────┐        │
         │              │  DONE   │        │
         │              │ (both)  │        │
         │              └────┬────┘        │
         │                   │             │
         │                   │ 3s display  │
         │                   ↓             │
         └───────────────→ READY ←─────────┘
                         (loop)
```

---

**END OF PRD**

**Version:** 1.1  
**Last Updated:** February 7, 2026  
**Status:** Ready for Implementation
