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


def call_claude(messages, max_tokens=4000):
    if not API_KEY:
        raise Exception("Missing ANTHROPIC_API_KEY environment variable.")
    resp = http_requests.post(
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


def clamp(val, lo, hi):
    try:
        return max(lo, min(float(val), hi))
    except (TypeError, ValueError):
        return None


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
- Duplicate characteristics with the same type and nominal value

TOLERANCE RULES:
- If the drawing shows a specific tolerance, use that exact value (e.g. "±0.12")
- If no specific tolerance is shown, apply the correct tier from the title block general tolerance table
- Do NOT write "per title block" — always resolve to the actual numeric tolerance
- Format as "±0.20" not "+0.20/-0.20"

HOLE PATTERNS:
- Multiple holes with the same diameter = ONE characteristic, e.g. "Hole diameter (6x)"
- Create separate characteristics for center spacing and edge distances

DESCRIPTIONS:
Use plain inspector language: "Plate width", "Plate height", "Sheet thickness", "Hole diameter", "Hole center spacing", "Flange height", "Bend radius", "Slot width", "Outer diameter"

matchingText RULES:
For every characteristic, set matchingText to the exact text string as it appears printed on the drawing.
This is used to locate the dimension callout on the drawing image.

Examples:
- Diameter 34.93 shown as Ø34.93  → matchingText: "Ø34.93"
- Diameter 26 shown as Ø26        → matchingText: "Ø26"
- Length 45.4                     → matchingText: "45.4"
- Two radii shown as 2XR3.5       → matchingText: "2XR3.5"
- Angle 103°                      → matchingText: "103°"
- Surface finish ≤3.2 Ra          → matchingText: "≤3.2 Ra"
- Six holes shown as 6XØ12.7      → matchingText: "6XØ12.7"
- Material Q235                   → matchingText: "Q235"

Return ONLY raw JSON. No markdown. No explanation. Exact format:
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
        ])

        extracted = extract_json(raw1)
        if not extracted.get("characteristics"):
            return jsonify({"error": "No characteristics found"}), 422

        # ── Step 2: Two-pass balloon placement ────────────────────────────
        #
        # Pass A: Ask Claude to describe the location of every dimension
        #         text label in plain English first. This forces it to
        #         actually read the drawing rather than estimate.
        #
        # Pass B: Convert those plain English descriptions into x/y coords.
        #
        # This two-pass approach is significantly more accurate than asking
        # for coordinates directly, because Claude reasons better about
        # position when first describing it in words.

        char_lookup = "\n".join(
            f"- ID {c.get('id')}: \"{c.get('matchingText', c.get('nominal', ''))}\" ({c.get('description', '')})"
            for c in extracted["characteristics"]
        )

        # Pass A — describe locations in plain English
        try:
            raw_describe = call_claude([
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
                            "text": f"""Look carefully at this engineering drawing. 

For each text label listed below, find exactly where it appears printed on the drawing and describe its location precisely. Use compass directions and drawing region names.

Labels to find:
{char_lookup}

Describe each location like:
"ID 1: top of drawing, above the top view outline, roughly centered horizontally"
"ID 2: left side of front view, below the horizontal centerline, on a horizontal dimension line"
"ID 3: center of front view, inside the bolt circle, pointing to hole"

Be specific about which view (top view, front view, side/section view, isometric view) and where within that view.
Do NOT mention the title block — if a label only appears there, say "not found on drawing geometry".

List all {len(extracted["characteristics"])} IDs."""
                        },
                    ],
                }
            ], max_tokens=1500)

            # Pass B — convert descriptions to coordinates
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
                            "text": f"""You have this engineering drawing in front of you. 

A colleague described the locations of dimension labels like this:
{raw_describe}

Now convert each described location into precise x/y coordinates.

COORDINATE SYSTEM:
- x = 0 at left edge of image, x = 100 at right edge
- y = 0 at top edge of image, y = 100 at bottom edge
- The drawing frame itself starts around x=8, y=8 and ends around x=92, y=88
- Title block is bottom-right: x > 55 and y > 65 — NO balloons there

PLACEMENT RULES:
1. Place the coordinate where the text label is, not where the feature is
2. Offset 2% to the right of the text so the balloon sits beside it
3. All values must be between 8 and 88
4. If a label is in the title block or not on the drawing geometry, use a fallback near the relevant feature
5. Spread values out — no two balloons should have both x and y within 3% of each other
6. Return exactly {len(extracted["characteristics"])} entries

Return ONLY raw JSON:
{{"b":[{{"id":1,"matchingText":"95.25","x":40,"y":22}},{{"id":2,"matchingText":"73.03","x":28,"y":52}}]}}"""
                        },
                    ],
                }
            ], max_tokens=1500)

            raw_balloons = extract_json(raw2).get("b", [])

        except Exception as e:
            print(f"Two-pass balloon error: {e}")
            raw_balloons = []

        # Validate, deduplicate, clamp all coordinates
        balloons = []
        seen_ids = set()

        for b in raw_balloons:
            bid = b.get("id")
            if bid is None or bid in seen_ids:
                continue
            seen_ids.add(bid)

            bx = clamp(b.get("x"), 8, 88)
            by = clamp(b.get("y"), 8, 88)
            mt = b.get("matchingText", "")

            if bx is None or by is None:
                idx = next(
                    (i for i, c in enumerate(extracted["characteristics"]) if c.get("id") == bid),
                    len(balloons)
                )
                bx = clamp(8 + (idx % 7) * 12, 8, 88)
                by = clamp(10 + (idx // 7) * 15, 8, 88)

            # Hard block: title block zone
            if bx > 55 and by > 65:
                by = 63

            balloons.append({"id": bid, "matchingText": mt, "x": bx, "y": by})

        # Fill any IDs that were missed
        returned_ids = {b["id"] for b in balloons}
        for i, c in enumerate(extracted["characteristics"]):
            cid = c.get("id")
            if cid not in returned_ids:
                bx = clamp(8 + (i % 7) * 12, 8, 88)
                by = clamp(10 + (i // 7) * 15, 8, 88)
                balloons.append({
                    "id": cid,
                    "matchingText": c.get("matchingText", ""),
                    "x": bx,
                    "y": by,
                })

        balloons.sort(key=lambda b: b["id"])

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
