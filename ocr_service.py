"""
ocr_service.py — Single-Call Vision LLM extraction with Visual Debugging.
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
import httpx
from dotenv import load_dotenv
import tomli

load_dotenv()

# ===========================================================================
#  1. CONFIGURATION & EXAMPLES
# ===========================================================================

# Ensure these paths exist relative to where you run the command!
FEW_SHOT_EXAMPLES = [
    {
        "image_paths": ["examples/ex3_1.jpg", "examples/ex3_2.jpg"], 
        "json_path": "examples/third_example.json"
    },
    {
        "image_paths": ["examples/ex4_1.jpg", "examples/ex4_2.jpg"], 
        "json_path": "examples/fourth_example.json"
    }
]

SYSTEM_PROMPT = """You are an expert OCR engine for fitness forms.
You will receive TWO images of a fitness log. 
1. Identify which page is CARDIO and which is STRENGTH.
2. Extract data from BOTH pages.
3. Merge them into a single JSON response.

=== OUTPUT FORMAT ===
{
  "cccare_id": "string or null (pick the clearest one)",
  "cardio_data": { "page_type": "cardio", "sessions": [...] },
  "strength_data": { "page_type": "strength", "sessions": [...] }
}
"""

# ===========================================================================
#  2. DEBUGGING TOOLS (HTML GENERATOR)
# ===========================================================================

def create_debug_html(messages, filename="latest_prompt_debug.html"):
    """
    Generates an HTML file showing the exact prompt sent to the LLM, 
    including the actual images rendered from Base64.
    """
    html = """
    <html>
    <head>
        <style>
            body { font-family: sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f0f0f0; }
            .message { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .role { font-weight: bold; text-transform: uppercase; margin-bottom: 10px; color: #555; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            .role.system { color: #d63384; }
            .role.user { color: #0d6efd; }
            .role.assistant { color: #198754; }
            img { max-width: 100%; border: 1px solid #ccc; margin-top: 10px; }
            pre { background: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap;}
        </style>
    </head>
    <body>
        <h1>LLM Prompt Preview</h1>
        <p>Open this file in a browser to see exactly what is being sent to the AI.</p>
    """

    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        html += f"<div class='message'><div class='role {role}'>{role.upper()}</div>"

        if isinstance(content, str):
            html += f"<pre>{content}</pre>"
        
        elif isinstance(content, list):
            for part in content:
                if part['type'] == 'text':
                    html += f"<div>{part['text']}</div>"
                elif part['type'] == 'image_url':
                    url = part['image_url']['url']
                    html += f"<div><img src='{url}' /></div>"
                    # html += f"<div style='font-size:10px; color:#999'>Base64 Data ({len(url)} chars)</div>"

        html += "</div>"

    html += "</body></html>"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
        logging.info(f"✅ DEBUG: Saved visual prompt to {os.path.abspath(filename)}")
    except Exception as e:
        logging.error(f"Failed to save debug HTML: {e}")

# ===========================================================================
#  3. HELPERS
# ===========================================================================

def encode_image_base64(image_path):
    # Robust path checking
    if not os.path.exists(image_path):
        # Try relative to script dir if simple path fails
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, image_path)
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            logging.error(f"❌ Image not found: {image_path} (checked absolute: {os.path.abspath(image_path)})")
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
#  4. PROMPT BUILDER
# ===========================================================================

def build_single_call_messages(target_img1_b64, target_img2_b64):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # 1. Add Examples
    for ex in FEW_SHOT_EXAMPLES:
        imgs_b64 = []
        for path in ex["image_paths"]:
            b64 = encode_image_base64(path)
            if b64: imgs_b64.append(b64)
        
        json_data = load_json_content(ex["json_path"])
        
        if imgs_b64 and json_data:
            user_content = [{"type": "text", "text": "Here is an example set of forms:"}]
            for b64 in imgs_b64:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": json.dumps(json_data, indent=2)})

    # 2. Add Target Images
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
            if images: new_msg["images"] = images
        else:
            new_msg["content"] = msg["content"]
        ollama_messages.append(new_msg)
    return ollama_messages

# ===========================================================================
#  5. BACKENDS
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
            timeout=90,
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
        options={"num_ctx": 16384}
    )
    return clean_and_parse_json(response["message"]["content"])

# ===========================================================================
#  6. MAIN
# ===========================================================================

async def process_form_single_pass(path1, path2, config):
    logging.info(f"Processing {path1} and {path2}...")
    
    img1 = encode_image_base64(path1)
    img2 = encode_image_base64(path2)
    
    if not img1 or not img2:
        logging.error("❌ Failed to load target images. Check paths.")
        return {"error": "image_load_failed"}

    # Build Prompt
    messages = build_single_call_messages(img1, img2)
    
    # --- HERE IS THE FIX: GENERATE HTML DEBUG FILE ---
    create_debug_html(messages, "latest_prompt_debug.html")

    # Send to Backend
    if config["ocr"]["backend"] == "ollama":
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_ollama, config, messages)
    else:
        result = await run_openrouter(config, messages)
        
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Check config exists
    if not os.path.exists("config.toml"):
        logging.error("config.toml not found.")
        sys.exit(1)

    with open("config.toml", "rb") as f:
        config = tomli.load(f)

    if len(sys.argv) < 3:
        print("Usage: uv run python ocr_service.py <image1.jpg> <image2.jpg>")
        sys.exit(1)

    res = asyncio.run(process_form_single_pass(sys.argv[1], sys.argv[2], config))
    
    print(json.dumps(res, indent=2))
    
    with open("results/latest_result.json", "w") as f:
        json.dump(res, f, indent=2)