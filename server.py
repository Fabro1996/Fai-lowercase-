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
BALLOON_MODEL = "claude-fable-5"


def call_claude(messages, max_tokens=4000, model=None):
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
            "model": model or CLAUDE_MODEL,
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
        ])

        extracted = extract_json(raw1)
        if not extracted.get("characteristics"):
            return jsonify({"error": "No characteristics found"}), 422

        # ── Step 2: Two-pass balloon placement ────────────────────────────
        #
        # Pass A: Ask Claude to describe where each text label is located
        #         in plain English. Forces it to actually read the drawing.
        #
        # Pass B: Convert those descriptions to x/y coordinates.

        char_lookup = "\n".join(
            f"- ID {c.get('id')}: find \"{c.get('matchingText', c.get('nominal', ''))}\" ({c.get('description', '')})"
            for c in extracted["characteristics"]
        )

        try:
            # Pass A — describe locations in words
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

The drawing has these regions:
- TOP VIEW: upper-left area, shows the flat plate outline from above with overall length/width dimensions
- FRONT VIEW: middle-left area, shows holes, radii, and hole spacing dimensions
- SIDE/SECTION VIEW: middle-right area, shows the bent profile, slot dimensions, and thickness
- ISOMETRIC VIEW: upper-right area — decorative only, ignore for balloon placement
- TITLE BLOCK: bottom-right corner — ignore, no balloons go here
- TECHNICAL NOTES: bottom-left area, red text — ignore, no balloons go here

For each label below, find exactly where that text appears as a DIMENSION CALLOUT on the drawing geometry (not in the title block, not in the notes). Describe its location using the region names above and where within that region.

{char_lookup}

Format each answer as:
ID X: [region] — [specific location description]

Example:
ID 1: TOP VIEW — above the top edge of the plate outline, centered horizontally, this is the overall length dimension line
ID 2: FRONT VIEW — left side, on a horizontal dimension line below the holes, showing hole pattern width
ID 7: FRONT VIEW — center, pointing into the bolt circle with a leader line, diameter callout

Be precise. All {len(extracted["characteristics"])} IDs must be described."""
                        },
                    ],
                }
            ], max_tokens=1500, model=BALLOON_MODEL)

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
                            "text": f"""You are looking at an engineering drawing. Convert these location descriptions into x/y percentage coordinates.

Location descriptions:
{raw_describe}

COORDINATE SYSTEM:
- x=0 left edge, x=100 right edge of the full image
- y=0 top edge, y=100 bottom edge of the full image
- Drawing frame: approximately x=8 to x=92, y=8 to y=88
- TOP VIEW occupies roughly: x=15 to x=55, y=10 to y=35
- FRONT VIEW occupies roughly: x=15 to x=55, y=35 to y=65
- SIDE VIEW occupies roughly: x=50 to x=72, y=35 to y=65
- ISOMETRIC VIEW: x=58 to x=92, y=10 to y=40 — no balloons here
- TITLE BLOCK: x=55 to x=92, y=65 to y=88 — no balloons here
- TECH NOTES: x=8 to x=50, y=68 to y=85 — no balloons here

RULES:
1. All x and y values must be between 8 and 85
2. Place the coordinate at the dimension TEXT location, offset 2% right so balloon sits beside it
3. No balloon in title block zone (x>55 AND y>65)
4. No balloon in tech notes zone (x<50 AND y>67)
5. No balloon in isometric zone (x>58 AND y<40) — unless a dimension is actually there
6. Spread balloons: if two would overlap (within 4% of each other on both axes), shift one by 4%
7. Return exactly {len(extracted["characteristics"])} entries

Return ONLY raw JSON:
{{"b":[{{"id":1,"matchingText":"95.25","x":38,"y":18}},{{"id":2,"matchingText":"73.03","x":25,"y":55}}]}}"""
                        },
                    ],
                }
            ], max_tokens=1500, model=BALLOON_MODEL)

            raw_balloons = extract_json(raw2).get("b", [])

        except Exception as e:
            print(f"Two-pass balloon error: {e}")
            raw_balloons = []

        # Validate, deduplicate, clamp
        balloons = []
        seen_ids = set()

        for b in raw_balloons:
            bid = b.get("id")
            if bid is None or bid in seen_ids:
                continue
            seen_ids.add(bid)

            bx = clamp(b.get("x"), 8, 85)
            by = clamp(b.get("y"), 8, 85)
            mt = b.get("matchingText", "")

            if bx is None or by is None:
                idx = next(
                    (i for i, c in enumerate(extracted["characteristics"]) if c.get("id") == bid),
                    len(balloons)
                )
                bx = clamp(10 + (idx % 6) * 13, 8, 85)
                by = clamp(12 + (idx // 6) * 16, 8, 85)

            # Block title block zone
            if bx > 55 and by > 65:
                by = 63

            # Block tech notes zone
            if bx < 50 and by > 67:
                by = 65

            balloons.append({"id": bid, "matchingText": mt, "x": bx, "y": by})

        # Fill any missed IDs
        returned_ids = {b["id"] for b in balloons}
        for i, c in enumerate(extracted["characteristics"]):
            cid = c.get("id")
            if cid not in returned_ids:
                bx = clamp(10 + (i % 6) * 13, 8, 85)
                by = clamp(12 + (i // 6) * 16, 8, 85)
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
