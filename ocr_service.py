"""
ocr_service.py — Single-Call Vision LLM extraction.
Sends BOTH images in one request to reduce latency and cost.
"""

import asyncio
import base64
import json
import logging
import os
import time
import httpx
from dotenv import load_dotenv
import tomli

load_dotenv()

# ===========================================================================
#  1. CONFIGURATION
# ===========================================================================

# Define your examples here. 
# We use single-page examples to teach the model what each page LOOKS like.
FEW_SHOT_EXAMPLES = [
    {
        "type": "cardio",
        "image_path": "ex3_2.jpg", 
        "json_path": "third_example.json"
    },
    {
        "type": "strength",
        "image_path": "ex3_1.jpg",
        "json_path": "third_example.json"
    }
]

SYSTEM_PROMPT = """You are an expert OCR engine for fitness forms.
You will receive TWO images of a fitness log. 
1. Identify each page (Cardio vs Strength).
2. Extract data from BOTH pages.
3. Merge them into a single JSON response.

=== OUTPUT FORMAT ===
{
  "cccare_id": "string or null (pick the clearest one)",
  "cardio_data": {
     "page_type": "cardio", 
     "sessions": [...] 
  } or null if no cardio page found,
  "strength_data": { 
     "page_type": "strength", 
     "sessions": [...] 
  } or null if no strength page found
}

=== RULES ===
- If both images are the same type (e.g. 2 cardio pages), extract both and merge the sessions list.
- Do not invent data. Use null for illegible fields.
- Return ONLY valid JSON.
"""

# ===========================================================================
#  2. HELPERS
# ===========================================================================

def encode_image_base64(image_path):
    if not os.path.exists(image_path):
        logging.warning(f"Image not found: {image_path}")
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_json_content(json_path):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        return json.load(f)

def clean_and_parse_json(response_text):
    text = response_text.strip()
    # Strip markdown if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("\n", 1)[0]
    if text.startswith("json"):
        text = text[4:]
    return json.loads(text.strip())

# ===========================================================================
#  3. PROMPT BUILDER (Multi-Image)
# ===========================================================================

def build_single_call_messages(img1_b64, img2_b64):
    """
    Constructs the prompt:
    System: Rules
    User: [Example 1 Image]
    Assistant: [Example 1 JSON]
    ...
    User: [Target Image 1] [Target Image 2] "Process these together"
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # 1. Add Single-Page Examples (To teach the model the schema)
    for ex in FEW_SHOT_EXAMPLES:
        b64 = encode_image_base64(ex["image_path"])
        json_data = load_json_content(ex["json_path"])
        
        if b64 and json_data:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Example of a {ex['type']} page:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps(json_data, indent=2)
            })

    # 2. Add BOTH Target Images in one message
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Here are two pages from the same log. Identify, extract, and reconcile them into one JSON."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}}
        ]
    })
    
    return messages

def convert_openai_to_ollama(messages):
    """Converts OpenAI format to Ollama format (images as bytes list)."""
    ollama_messages = []
    for msg in messages:
        new_msg = {"role": msg["role"], "content": ""}
        
        if isinstance(msg["content"], list):
            text_parts = []
            images = []
            for part in msg["content"]:
                if part["type"] == "text":
                    text_parts.append(part["text"])
                elif part["type"] == "image_url":
                    b64_str = part["image_url"]["url"].split(",")[-1]
                    images.append(base64.b64decode(b64_str))
            
            new_msg["content"] = "\n".join(text_parts)
            if images:
                new_msg["images"] = images
        else:
            new_msg["content"] = msg["content"]
            
        ollama_messages.append(new_msg)
    return ollama_messages

# ===========================================================================
#  4. BACKENDS
# ===========================================================================

async def run_openrouter(config, messages):
    headers = {
        "Authorization": f"Bearer {os.getenv(config['api']['openrouter_api_key_env'])}",
        "HTTP-Referer": config["api"]["http_referer"],
        "X-Title": config["api"]["app_title"],
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": config["models"]["cardio_extraction_model"], # Use the best model
        "messages": messages,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{config['api']['openrouter_base_url']}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60, # Longer timeout for double image processing
        )
    
    if response.status_code != 200:
        logging.error(f"API Error: {response.text}")
        response.raise_for_status()
        
    return clean_and_parse_json(response.json()["choices"][0]["message"]["content"])

def run_ollama(config, messages):
    import ollama
    client = ollama.Client(host=config["ollama"].get("host"))
    
    # Convert format
    ollama_msgs = convert_openai_to_ollama(messages)
    
    response = client.chat(
        model=config["ollama"]["extraction_model"],
        messages=ollama_msgs,
        format="json",
        options={"num_ctx": 8192} # Crucial: Increase context for 2 images
    )
    return clean_and_parse_json(response["message"]["content"])

# ===========================================================================
#  5. MAIN
# ===========================================================================

async def process_form_single_pass(path1, path2, config):
    logging.info(f"Processing {path1} and {path2} in SINGLE PASS...")
    
    # 1. Prepare Images
    img1 = encode_image_base64(path1)
    img2 = encode_image_base64(path2)
    
    # 2. Build ONE Prompt with BOTH images
    messages = build_single_call_messages(img1, img2)
    
    # 3. Send to backend
    if config["ocr"]["backend"] == "ollama":
        # Run sync ollama in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_ollama, config, messages)
    else:
        result = await run_openrouter(config, messages)
        
    return result

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    with open("config.toml", "rb") as f:
        config = tomli.load(f)

    if len(sys.argv) < 3:
        print("Usage: python ocr_service.py img1.jpg img2.jpg")
        sys.exit(1)

    res = asyncio.run(process_form_single_pass(sys.argv[1], sys.argv[2], config))
    
    print(json.dumps(res, indent=2))
    
    # Save debug output
    with open("results/latest_result.json", "w") as f:
        json.dump(res, f, indent=2)