import os
import json
import base64
import re
from flask import Flask, request, jsonify, send_from_directory
import requests as http_requests

app = Flask(__name__, static_folder="public")
API_KEY = (
    os.environ.get("ANTHROPIC_API_KEY", "")
    .replace("\n", "")
    .replace("\r", "")
    .replace(" ", "")
    .strip()
)
CLAUDE_MODEL = "claude-sonnet-4-5"


def _post_to_claude(messages, max_tokens, model):
    resp = http_requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        },
        timeout=120,
    )
    try:
        data = resp.json()
    except Exception:
        raise Exception(f"API returned non-JSON response: {resp.text[:500]}")
    if not resp.ok:
        raise Exception(f"API {resp.status_code}: {data}")
    return data


def call_claude(messages, max_tokens=4000, model=None, fallback_model=None):
    """
    Calls the Anthropic API. If the model returns stop_reason == "refusal"
    (a normal HTTP 200 — Fable 5's safety classifiers can decline a request
    without raising an error), automatically retry on fallback_model.
    """
    if not API_KEY:
        raise Exception("Missing ANTHROPIC_API_KEY environment variable.")

    use_model = model or CLAUDE_MODEL
    data = _post_to_claude(messages, max_tokens, use_model)

    if data.get("stop_reason") == "refusal":
        details = data.get("stop_details") or {}
        print(f"⚠ {use_model} refused (category: {details.get('category', 'unknown')})")

        if fallback_model and fallback_model != use_model:
            print(f"  Retrying with fallback model: {fallback_model}")
            data = _post_to_claude(messages, max_tokens, fallback_model)
            use_model = fallback_model

            if data.get("stop_reason") == "refusal":
                details2 = data.get("stop_details") or {}
                raise Exception(
                    f"Both {model or CLAUDE_MODEL} and fallback {fallback_model} refused "
                    f"(category: {details2.get('category', 'unknown')})"
                )
        else:
            raise Exception(
                f"{use_model} refused request (category: {details.get('category', 'unknown')})"
            )

    text = "".join(block.get("text", "") for block in data.get("content", []))
    if not text:
        raise Exception(
            f"Empty response from {use_model}: stop_reason={data.get('stop_reason')}"
        )
    return text


def extract_json(raw):
    s = raw.find("{")
    e = raw.rfind("}")
    if s == -1 or e == -1:
        raise Exception("No JSON found: " + raw[:300])
    try:
        return json.loads(raw[s:e + 1])
    except json.JSONDecodeError as je:
        # This usually means the model's response was cut off mid-output
        # (hit max_tokens) before it could finish the JSON structure —
        # common on drawings with many characteristics/dimensions.
        raise Exception(
            f"Response appears truncated or malformed ({je}). "
            f"This drawing may have too many characteristics for the "
            f"current response size limit — try a simpler/smaller drawing "
            f"or contact support to increase the limit."
        )



@app.route("/")
def index():
    if os.path.exists("public/index.html"):
        return send_from_directory("public", "index.html")
    if os.path.exists("Public/index.html"):
        return send_from_directory("Public", "index.html")
    return "index.html not found. Check your public/Public folder name.", 404


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files["file"]
        mime = f.mimetype or "image/jpeg"
        image_bytes = f.read()
        b64 = base64.b64encode(image_bytes).decode()

        # ── Step 1: Extract characteristics ──────────────────────────────
        raw1 = call_claude([
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": """You are an FAI inspection engineer reviewing an engineering drawing.

Extract only characteristics that would appear on a real AS9102 First Article Inspection report.

EXTRACT THESE:
- Linear dimensions: plate length, plate width, plate height, flange height, step depth, slot length, slot width, bend dimension
- Diameters: each unique hole diameter, bore diameter, boss diameter (one characteristic per unique value)
- Radii: each unique bend radius or corner radius value (one characteristic per unique value)
- Angles: bend angles, chamfer angles
- Thicknesses: sheet thickness, wall thickness, flange thickness
- Hole locations: center-to-center spacing, edge-to-center offset, hole pattern dimensions
- Surface finish: Ra value if called out
- Material: if a material cert would be required
- Finish: if a finish cert or visual inspection would be required

DO NOT EXTRACT:
- Counts or quantities ("4X", "6X", "2X" prefix alone is not a characteristic)
- Title block fields: part number, revision letter, drawn by, checked by, date, company name, sheet number
- Drawing border labels, zone letters (A B C D), zone numbers (1 2 3 4 5 6)
- Projection symbol or scale notation
- Metadata that is not a measurable dimension
- Duplicate characteristics — if the same numeric value appears in two views for the same dimension, create ONE entry only
- Do not create separate "width" and "height" entries if they have the same nominal value — that is one dimension shown twice

DEDUPLICATION RULES — CRITICAL:
- If a dimension value appears as both a horizontal AND vertical measurement with the same nominal, it means the part is square/symmetric — create ONE characteristic only
- If the same hole spacing appears in both X and Y directions with the same value, create ONE characteristic labeled "Hole center spacing"
- Never create two rows with the same nominal value and same type

TOLERANCE RULES:
- If the drawing shows a specific tolerance, use that exact value (e.g. "±0.12")
- If no specific tolerance is shown, apply the correct tier from the title block general tolerance table
- Do NOT write "per title block" — always resolve to the actual numeric tolerance
- Format as "±0.20" not "+0.20/-0.20"

HOLE PATTERNS:
- Multiple holes with the same diameter = ONE characteristic, e.g. "Hole diameter (6x)"
- If hole spacing is the same in both directions, ONE characteristic: "Hole center spacing"
- If hole spacing differs in X vs Y, create two: "Hole center spacing (horizontal)" and "Hole center spacing (vertical)"

DESCRIPTIONS — use plain inspector language:
"Plate length", "Plate width", "Sheet thickness", "Hole diameter", "Hole center spacing",
"Flange height", "Bend radius", "Slot width", "Slot height", "Slot corner radius", "Outer diameter"

matchingText — set to the exact text as printed on the drawing:
- Ø34.93 → "Ø34.93"
- 2XR3.5 → "2XR3.5"
- 6XØ12.7 → "6XØ12.7"
- 45.4 → "45.4"
- 103° → "103°"

Return ONLY raw JSON. No markdown. No explanation:
{
  "partInfo": {
    "partNumber": "",
    "partName": "",
    "revision": "",
    "material": "",
    "finish": "",
    "drawingNumber": "",
    "date": "",
    "drawnBy": "",
    "checkedBy": "",
    "approvedBy": "",
    "company": "",
    "units": "mm"
  },
  "generalNotes": [],
  "tolerances": {
    "linear": "",
    "angular": "",
    "hole": "",
    "surfaceRoughness": ""
  },
  "characteristics": [
    {
      "id": 1,
      "type": "diameter",
      "description": "Hole diameter (6x)",
      "nominal": "12.7",
      "unit": "mm",
      "tolerance": "±0.12",
      "view": "front",
      "priority": "standard",
      "matchingText": "6XØ12.7"
    }
  ]
}"""
                    },
                ],
            }
        ], max_tokens=8000)

        extracted = extract_json(raw1)
        if not extracted.get("characteristics"):
            return jsonify({"error": "No characteristics found"}), 422

        # ── Step 2: Balloon placement ──────────────────────────────────────
        #
        # Balloon positions are no longer generated by AI. The frontend
        # presents each characteristic as a draggable chip that the user
        # places directly onto the drawing image. Positions are tracked
        # entirely client-side (in index.html) and don't need to come back
        # through the API — the FAI form, legend, and exports all key off
        # characteristic id, not x/y position. This removes all AI placement
        # guessing, truncation, and refusal-handling complexity.
        #
        # balloons starts empty; the frontend populates positions for
        # display only.
        balloons = []


        # ── Step 3: FAI form ──────────────────────────────────────────────
        raw3 = call_claude([
            {
                "role": "user",
                "content": f"""Fill an AS9102 FAI dimensional results form.

partInfo:
{json.dumps(extracted.get("partInfo", {}))}

generalTolerances:
{json.dumps(extracted.get("tolerances", {}))}

characteristics:
{json.dumps(extracted.get("characteristics", []))}

Rules:
- Create one row for every characteristic.
- For desc: use the characteristic description exactly as given.
- For nominal: use the numeric value only, no units.
- For tolPlus and tolMinus: split the tolerance into separate + and - values.
  Example: tolerance "±0.20" becomes tolPlus="+0.20" and tolMinus="-0.20"
  Example: tolerance "±0.12" becomes tolPlus="+0.12" and tolMinus="-0.12"
  Example: tolerance "±1.0°" becomes tolPlus="+1.0" and tolMinus="-1.0"
- For certs: list only certifications actually needed based on material and finish fields.
  Always include material cert if material is specified.
  Add finish cert only if a plating, coating, or anodize is specified.

Return ONLY raw JSON in this exact format:
{{
  "s1": {{
    "partNumber": "",
    "partName": "",
    "revision": "",
    "drawingNumber": "",
    "material": "",
    "finish": "",
    "date": "",
    "quantity": "1"
  }},
  "rows": [
    {{
      "n": 1,
      "desc": "Plate width",
      "nominal": "95.25",
      "tolPlus": "+0.45",
      "tolMinus": "-0.45",
      "notes": ""
    }}
  ],
  "certs": ["Material Certification"]
}}"""
            }
        ])

        form_data = extract_json(raw3)
        if not form_data.get("rows"):
            return jsonify({"error": "Form generation failed"}), 422

        return jsonify({
            "extracted": extracted,
            "balloons":  balloons,
            "formData":  form_data,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"\n✅ Running at http://localhost:{port}")
    print(f"   API key: {'YES' if API_KEY else 'NO - check environment variables'}")
    print(f"   Claude model: {CLAUDE_MODEL}\n")
    app.run(host="0.0.0.0", port=port)
