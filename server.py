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
                        "text": """You are a metrologist and GD&T expert performing a First Article Inspection per AS9102.

Analyze this engineering drawing and extract every measurable characteristic an inspector would verify on the physical part. Use precise metrology and GD&T terminology for all descriptions.

═══ COORDINATE DIMENSIONS — extract all of these ═══
- Overall length / Overall width / Overall height
- Sheet thickness / Wall thickness / Plate thickness
- Flange height / Flange width / Flange length
- Leg height / Leg width
- Step height / Step depth / Step width
- Slot length / Slot width / Slot depth
- Boss height / Boss diameter
- Hole diameter (state quantity: "Hole diameter 4x Ø12.7")
- Bore diameter / Counterbore diameter / Countersink diameter
- Bend radius (inside) / Bend radius (outside)
- Corner radius / Fillet radius
- Chamfer size / Chamfer angle
- Hole center to cut edge (horizontal) — distance from hole centerline to nearest machined or cut edge, horizontal direction
- Hole center to cut edge (vertical) — same, vertical direction
- Hole center to bend (horizontal) — distance from hole centerline to bend line, horizontal
- Hole center to bend (vertical) — same, vertical direction
- Hole center to hole center (horizontal) — center-to-center spacing between holes, horizontal
- Hole center to hole center (vertical) — same, vertical direction
- Hole center to hole center (diagonal) — if dimensioned diagonally
- Edge to edge distance — between two cut or formed edges
- Bend angle / Included angle / Half angle
- True radius / Arc radius

═══ GD&T FEATURE CONTROL FRAMES — extract all that appear ═══
For each feature control frame (FCF) on the drawing, extract:
- True Position — e.g. "True position Ø0.25 |A|B|C| (4x holes)"
- Flatness — e.g. "Flatness 0.10 (top surface)"
- Straightness — e.g. "Straightness 0.05"
- Circularity / Roundness
- Cylindricity
- Perpendicularity — e.g. "Perpendicularity Ø0.25 |A|"
- Parallelism — e.g. "Parallelism 0.10 |A|"
- Angularity — e.g. "Angularity 0.5° |A|"
- Profile of a line / Profile of a surface
- Runout (circular) / Total runout
- Concentricity / Coaxiality
- Symmetry
For GD&T: set tolerance to the geometric tolerance value (e.g. "0.25") and include the datum reference in the description.

═══ SURFACE & MATERIAL ═══
- Surface roughness Ra / Rz if called out (e.g. "Surface roughness Ra 3.2 μm")
- Material specification — if a material cert is required (e.g. "Material: AL5052-H32")
- Surface finish / Coating / Plating — if a process cert or visual inspection is required

═══ DEDUPLICATION — CRITICAL ═══
- If the same dimension appears in multiple views, create ONE entry
- If horizontal and vertical hole spacing are equal, create ONE entry: "Hole center to hole center"
- If horizontal and vertical hole-to-edge are equal, create ONE entry
- Never duplicate the same type + nominal combination
- Quantity prefixes like "4X" or "6X" belong in the description, not as separate entries

═══ TOLERANCE RULES ═══
- Use the specific tolerance shown on that dimension (e.g. "±0.12")
- If no specific tolerance, read the general tolerance table in the title block and apply the correct tier
- For GD&T, use the geometric tolerance value shown in the feature control frame (e.g. "0.25" for Ø0.25 position)
- Never write "per title block" — always resolve to the actual numeric value
- Format plus/minus as "±0.20"

═══ DESCRIPTION FORMAT ═══
Use full metrology language that an inspector would understand:
✓ "Overall length"
✓ "Overall width"  
✓ "Sheet thickness"
✓ "Hole diameter 6x (Ø12.7)"
✓ "Hole center to cut edge — left side"
✓ "Hole center to cut edge — bottom"
✓ "Hole center to hole center — horizontal"
✓ "Hole center to bend — horizontal"
✓ "Inside bend radius 4x (R6.35)"
✓ "Flange height"
✓ "Slot width"
✓ "Slot length"
✓ "Slot corner radius 2x (R1.5)"
✓ "True position Ø0.25 |A|B|C| (4x holes)"
✓ "Flatness 0.10 — datum A surface"
✓ "Perpendicularity Ø0.13 |A| (hole axis)"
✓ "Surface roughness Ra 3.2 μm"
✓ "Material: AL5052-H32"
✓ "Finish: black anodize"

matchingText — the EXACT text as printed on the drawing:
- Ø34.93 → "Ø34.93"
- 6XØ12.7 → "6XØ12.7"
- 2XR3.5 → "2XR3.5"
- 103° → "103°"
- ⊕|Ø0.25|A|B|C| → "⊕|Ø0.25|A|B|C|"
- ≤3.2 Ra → "≤3.2 Ra"

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
      "description": "Hole diameter 6x (Ø12.7)",
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
                "content": f"""You are completing an AS9102 First Article Inspection dimensional results form.

partInfo:
{json.dumps(extracted.get("partInfo", {}))}

generalTolerances:
{json.dumps(extracted.get("tolerances", {}))}

characteristics:
{json.dumps(extracted.get("characteristics", []))}

Rules:
- Create exactly one row per characteristic. Do not skip any.
- For n: use the characteristic id number.
- For desc: copy the characteristic description exactly as given — preserve all metrology and GD&T language.
- For nominal: numeric value only, no units, no symbols. For GD&T characteristics use the geometric tolerance value.
- For tolPlus / tolMinus: split the tolerance symmetrically.
  "±0.20" → tolPlus="+0.20", tolMinus="-0.20"
  "±0.12" → tolPlus="+0.12", tolMinus="-0.12"
  "±1.0°" → tolPlus="+1.0", tolMinus="-1.0"
  For GD&T (flatness, position, etc.) where tolerance is a single value like "0.25":
  tolPlus="0.25", tolMinus="0.00" (geometric tolerances are unilateral)
  For surface roughness: tolPlus="", tolMinus="" and put "max" in notes.
  For material/finish: tolPlus="", tolMinus="", notes="Cert required".
- For notes: add any relevant inspection method or datum reference.
  Examples: "CMM", "per datum A", "optical comparator", "Ra profilometer", "visual + cert"
- For certs: list every certification the inspector must obtain.
  Always include material cert if material is specified.
  Add finish cert for any plating, coating, anodize, or powder coat.
  Add process cert for heat treat, passivation, or other processes.

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
      "desc": "Overall length",
      "nominal": "95.25",
      "tolPlus": "+0.20",
      "tolMinus": "-0.20",
      "notes": ""
    }}
  ],
  "certs": ["Material Certification — AL5052-H32"]
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
