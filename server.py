import os
import json
import base64
from flask import Flask, request, jsonify, send_from_directory
import requests

app = Flask(__name__, static_folder="public")
API_KEY = (
    os.environ.get("ANTHROPIC_API_KEY", "")
    .replace("\n", "")
    .replace("\r", "")
    .replace(" ", "")
    .strip()
)
CLAUDE_MODEL = "claude-sonnet-4-5"

def call_claude(messages, max_tokens=4000):
    if not API_KEY:
        raise Exception("Missing ANTHROPIC_API_KEY environment variable.")
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CLAUDE_MODEL,
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
    return "".join(block.get("text", "") for block in data.get("content", []))

def extract_json(raw):
    s = raw.find("{")
    e = raw.rfind("}")
    if s == -1 or e == -1:
        raise Exception("No JSON found: " + raw[:300])
    return json.loads(raw[s:e + 1])

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
        b64 = base64.b64encode(f.read()).decode()

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
                        "text": """You are an FAI inspection engineer analyzing an engineering drawing.

Extract only MEASURABLE inspection characteristics — items an inspector will physically measure or verify on the part.

INCLUDE:
- Linear dimensions (lengths, widths, heights, depths)
- Diameters (each unique diameter as one characteristic)
- Radii (each unique radius value as one characteristic)
- Angles
- Thicknesses
- Slot widths and lengths
- Bend dimensions and bend radii
- Hole diameters and hole pattern spacing
- Surface roughness / Ra values
- Material specification (if it must be verified by cert)
- Finish specification (if it must be verified by cert or visual)

DO NOT include:
- Counts of features (e.g. "4x holes" is not a characteristic — the hole diameter IS)
- Title block text (part number, revision, company name, drawn by, etc.)
- Grid zone labels (A, B, C, 1, 2, 3)
- Border or frame elements
- Notes that are not measurable (e.g. "no sharp edges" only if it cannot be measured)
- Duplicate entries for the same nominal value and type

Each characteristic must have all 8 fields filled in:
- id: sequential number starting at 1
- type: one of [diameter, radius, length, width, height, depth, angle, thickness, slot, bend, hole_spacing, surface_roughness, material, finish]
- description: clear human-readable name (e.g. "Outer diameter", "Flange thickness", "Hole diameter")
- nominal: the numeric value as a string (e.g. "34.93")
- unit: mm, degrees, or text for material/finish
- tolerance: specific value if shown (e.g. "±0.20"), or "per title block" if covered by general tolerance
- view: which view this dimension appears in (e.g. "front", "top", "section A-A", "general note")
- priority: "key" if it is a critical or key characteristic, otherwise "standard"

Return ONLY raw JSON. No markdown. No explanation. Use this exact format:
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
      "description": "Outer diameter",
      "nominal": "34.93",
      "unit": "mm",
      "tolerance": "±0.20",
      "view": "front",
      "priority": "standard"
    }
  ]
}"""
                    },
                ],
            }
        ])

        extracted = extract_json(raw1)
        if not extracted.get("characteristics"):
            return jsonify({"error": "No characteristics found"}), 422

        try:
            chars = ", ".join(
                f"{c.get('id')}:{c.get('description')}({c.get('nominal')})"
                for c in extracted["characteristics"]
            )
            raw2 = call_claude([
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
                            "text": f"""Give approximate position of each characteristic as percent of image.
x = left-right 0-100.
y = top-bottom 0-100.
Characteristics:
{chars}
Return ONLY raw JSON in this exact format:
{{"b":[{{"id":1,"x":45,"y":30}}]}}"""
                        },
                    ],
                }
            ], max_tokens=2000)
            balloons = extract_json(raw2).get("b", [])
        except Exception:
            balloons = [
                {
                    "id": c.get("id"),
                    "x": 5 + (i % 8) * 11,
                    "y": 6 + (i // 8) * 14,
                }
                for i, c in enumerate(extracted["characteristics"])
            ]

        raw3 = call_claude([
            {
                "role": "user",
                "content": f"""Fill an AS9102 FAI form.
partInfo:
{json.dumps(extracted.get("partInfo", {}))}
characteristics:
{json.dumps(extracted.get("characteristics", []))}
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
      "desc": "what measured",
      "nominal": "val",
      "tolPlus": "+val",
      "tolMinus": "-val",
      "notes": ""
    }}
  ],
  "certs": ["Material Certification"]
}}
One row per characteristic."""
            }
        ])

        form_data = extract_json(raw3)
        if not form_data.get("rows"):
            return jsonify({"error": "Form generation failed"}), 422

        return jsonify({
            "extracted": extracted,
            "balloons": balloons,
            "formData": form_data,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"\n✅ Running at http://localhost:{port}")
    print(f"   API key: {'YES' if API_KEY else 'NO - check environment variables'}")
    print(f"   Claude model: {CLAUDE_MODEL}\n")
    app.run(host="0.0.0.0", port=port)
