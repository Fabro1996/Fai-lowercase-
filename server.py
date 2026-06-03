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


def clamp(val, lo, hi):
    try:
        return max(lo, min(float(val), hi))
    except (TypeError, ValueError):
        return None


# ── OCR INTEGRATION POINT ─────────────────────────────────────────────────────
# Future: replace this stub with a real OCR engine (e.g. Google Vision,
# Tesseract, AWS Textract). The function should accept the raw image bytes
# and return a list of detected text blocks with bounding boxes:
#
# [
#   {
#     "text": "Ø34.93",
#     "x": 42.1,   # center x as % of image width
#     "y": 31.5,   # center y as % of image height
#     "w": 4.2,    # box width as % of image width
#     "h": 1.8     # box height as % of image height
#   },
#   ...
# ]
#
# Once OCR is wired in, balloon placement becomes:
#   1. Run ocr_detect_text(image_bytes) → bounding boxes
#   2. For each characteristic, match matchingText against bounding boxes
#   3. Use the matched bounding box center as the balloon x/y
#   4. Fall back to Claude coordinate estimation only when no match is found
#
def ocr_detect_text(image_bytes):
    # STUB — returns empty list until OCR engine is integrated
    return []


def find_balloon_position_via_ocr(matching_text, ocr_blocks):
    """
    Given a matchingText string and a list of OCR bounding boxes,
    return (x, y) if a match is found, or None if not.

    Matching strategy (to be refined when OCR is live):
      1. Exact string match
      2. Numeric value match (strip Ø, R, ≤, spaces, degree symbol)
      3. Partial match for compound callouts like "2X R3.5"
    """
    # STUB — no OCR blocks available yet
    if not ocr_blocks or not matching_text:
        return None

    clean = (
        matching_text
        .replace("Ø", "")
        .replace("R", "")
        .replace("≤", "")
        .replace("°", "")
        .replace(" ", "")
        .strip()
    )

    for block in ocr_blocks:
        block_clean = (
            block.get("text", "")
            .replace("Ø", "")
            .replace("R", "")
            .replace("≤", "")
            .replace("°", "")
            .replace(" ", "")
            .strip()
        )
        if clean and (clean in block_clean or block_clean in clean):
            return block.get("x"), block.get("y")

    return None
# ── END OCR INTEGRATION POINT ─────────────────────────────────────────────────


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
This will be used to locate the dimension callout via OCR for precise balloon placement.

Examples:
- A diameter of 34.93 shown as Ø34.93 → matchingText: "Ø34.93"
- A diameter of 26 shown as Ø26 → matchingText: "Ø26"
- A length of 45.4 shown as 45.4 → matchingText: "45.4"
- Two radii of 3.5 shown as 2XR3.5 → matchingText: "2XR3.5"
- An angle of 103 degrees shown as 103° → matchingText: "103°"
- Surface roughness shown as ≤3.2 Ra → matchingText: "≤3.2 Ra"
- A hole diameter of 12.7 shown as 6XØ12.7 → matchingText: "6XØ12.7"
- Material Q235 → matchingText: "Q235"

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

        # ── Step 2: Balloon placement ─────────────────────────────────────
        #
        # CURRENT METHOD: Claude vision estimates x/y from the image.
        # FUTURE METHOD:  OCR detects bounding boxes, matchingText lookup
        #                 finds the exact callout, balloon is placed precisely.
        #
        # When OCR is integrated, replace the call_claude block below with:
        #   ocr_blocks = ocr_detect_text(image_bytes)
        #   for each characteristic:
        #     pos = find_balloon_position_via_ocr(c["matchingText"], ocr_blocks)
        #     if pos: use pos
        #     else: fall back to Claude estimate
        #
        # OCR integration point — run text detection on the drawing
        ocr_blocks = ocr_detect_text(image_bytes)

        try:
            char_list = "\n".join(
                f"ID {c.get('id')}: {c.get('description')} = {c.get('nominal')} {c.get('unit', '')} | matchingText: \"{c.get('matchingText', '')}\""
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
                            "text": f"""You are placing balloon markers on an engineering drawing.

For each characteristic below, find the exact dimension callout text on the drawing and return its location.

Characteristics (with the exact text to find):
{char_list}

RULES:
1. x and y are percentages of the full image. x=0 left, x=100 right. y=0 top, y=100 bottom.
2. ALL values must be between 5 and 95. Never go below 5 or above 95.
3. Use the matchingText field to find the exact callout on the drawing.
4. Place the coordinate 2-3% beside the text, not on top of it.
5. Do NOT place balloons in the title block (bottom-right corner).
6. Do NOT place balloons outside the drawing frame.
7. You must return exactly {len(extracted["characteristics"])} entries.

Return ONLY raw JSON:
{{"b":[{{"id":1,"matchingText":"Ø34.93","x":45,"y":30}}]}}"""
                        },
                    ],
                }
            ], max_tokens=2000)

            raw_balloons = extract_json(raw2).get("b", [])

            # Build a lookup of OCR-matched positions (populated when OCR is live)
            # OCR match point — override Claude estimates with precise OCR positions
            ocr_positions = {}
            for c in extracted["characteristics"]:
                cid = c.get("id")
                mt = c.get("matchingText", "")
                pos = find_balloon_position_via_ocr(mt, ocr_blocks)
                if pos and pos[0] is not None and pos[1] is not None:
                    ocr_positions[cid] = {
                        "x": clamp(pos[0], 5, 95),
                        "y": clamp(pos[1], 5, 95),
                    }

            # Merge: OCR position wins, Claude estimate is fallback
            balloons = []
            seen_ids = set()

            for b in raw_balloons:
                bid = b.get("id")
                if bid is None or bid in seen_ids:
                    continue
                seen_ids.add(bid)

                if bid in ocr_positions:
                    # OCR match found — use precise position
                    bx = ocr_positions[bid]["x"]
                    by = ocr_positions[bid]["y"]
                else:
                    # Claude vision estimate — clamp to safe area
                    bx = clamp(b.get("x"), 5, 95)
                    by = clamp(b.get("y"), 5, 95)

                    if bx is None or by is None:
                        idx = next(
                            (i for i, c in enumerate(extracted["characteristics"]) if c.get("id") == bid),
                            len(balloons)
                        )
                        bx = clamp(5 + (idx % 8) * 11, 5, 95)
                        by = clamp(6 + (idx // 8) * 14, 5, 95)

                    # Push out of title block zone
                    if bx > 60 and by > 72:
                        by = 70

                # Carry matchingText forward for future OCR debugging
                mt = b.get("matchingText", "")

                balloons.append({"id": bid, "matchingText": mt, "x": bx, "y": by})

            # Fill any IDs the model missed
            returned_ids = {b["id"] for b in balloons}
            for i, c in enumerate(extracted["characteristics"]):
                cid = c.get("id")
                if cid not in returned_ids:
                    if cid in ocr_positions:
                        bx = ocr_positions[cid]["x"]
                        by = ocr_positions[cid]["y"]
                    else:
                        bx = clamp(5 + (i % 8) * 11, 5, 95)
                        by = clamp(6 + (i // 8) * 14, 5, 95)
                    balloons.append({
                        "id": cid,
                        "matchingText": c.get("matchingText", ""),
                        "x": bx,
                        "y": by,
                    })

            balloons.sort(key=lambda b: b["id"])

        except Exception:
            balloons = [
                {
                    "id": c.get("id"),
                    "matchingText": c.get("matchingText", ""),
                    "x": clamp(5 + (i % 8) * 11, 5, 95),
                    "y": clamp(6 + (i // 8) * 14, 5, 95),
                }
                for i, c in enumerate(extracted["characteristics"])
            ]

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
