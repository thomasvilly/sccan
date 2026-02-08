"""
ocr_service.py — Single-Call Vision LLM extraction.
Sends BOTH images in one request to reduce latency and cost.
Few-Shot Examples now use (2 Images -> 1 JSON) structure.
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

# CORRECTED STRUCTURE:
# Each example consists of TWO images and ONE combined ground-truth JSON.
FEW_SHOT_EXAMPLES = [
    {
        "image_paths": ["examples/ex3_1.jpg", "examples/ex3_2.jpg"], 
        "json_path": "examples/third_example.json"
    }
    # You can add more pairs here if you have them
]

SYSTEM_PROMPT = """You are an expert OCR engine for fitness forms.
You will receive TWO images of a fitness log. 
1. Identify which page is CARDIO and which is STRENGTH.
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
        logging.warning(f"JSON not found: {json_path}")
        return {}
    with open(json_path, "r") as f:
        return json.load(f)

def clean_and_parse_json(response_text):
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("\n", 1)[0]
    if text.startswith("json"):
        text = text[4:]
    return json.loads(text.strip())

# ===========================================================================
#  3. PROMPT BUILDER (Multi-Image Few-Shot)
# ===========================================================================

def build_single_call_messages(target_img1_b64, target_img2_b64):
    """
    Constructs the prompt history:
    System: Rules
    User: [Example 1 Img A] [Example 1 Img B] "Extract these"
    Assistant: [Example 1 JSON]
    User: [Target Img A] [Target Img B] "Extract these"
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # 1. Add Full 2-Page Examples
    for ex in FEW_SHOT_EXAMPLES:
        # Load images
        imgs_b64 = []
        for path in ex["image_paths"]:
            b64 = encode_image_base64(path)
            if b64:
                imgs_b64.append(b64)
        
        json_data = load_json_content(ex["json_path"])
        
        # Only add if we have at least one image and the JSON
        if imgs_b64 and json_data:
            user_content = [{"type": "text", "text": "Here is an example set of forms:"}]
            for b64 in imgs_b64:
                user_content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": json.dumps(json_data, indent=2)})

    # 2. Add the Target Images
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Here are two new pages. Identify, extract, and reconcile them into one JSON."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_img1_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_img2_b64}"}}
        ]
    })
    
    return messages

def convert_openai_to_ollama(messages):
    """Converts OpenAI format to Ollama format."""
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
        "model": config["models"]["cardio_extraction_model"],
        "messages": messages,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{config['api']['openrouter_base_url']}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )
    
    if response.status_code != 200:
        logging.error(f"API Error: {response.text}")
        response.raise_for_status()
        
    return clean_and_parse_json(response.json()["choices"][0]["message"]["content"])

def run_ollama(config, messages):
    import ollama
    client = ollama.Client(host=config["ollama"].get("host"))
    
    ollama_msgs = convert_openai_to_ollama(messages)
    
    response = client.chat(
        model=config["ollama"]["extraction_model"],
        messages=ollama_msgs,
        format="json",
        options={"num_ctx": 16384} # High context for multi-image history
    )
    return clean_and_parse_json(response["message"]["content"])

# ===========================================================================
#  5. MAIN
# ===========================================================================

async def process_form_single_pass(path1, path2, config):
    logging.info(f"Processing {path1} and {path2}...")
    
    img1 = encode_image_base64(path1)
    img2 = encode_image_base64(path2)
    
    if not img1 or not img2:
        logging.error("Failed to load target images.")
        return {"error": "image_load_failed"}

    messages = build_single_call_messages(img1, img2)
    
    if config["ocr"]["backend"] == "ollama":
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
    
    # Save output
    with open("results/latest_result.json", "w") as f:
        json.dump(res, f, indent=2)