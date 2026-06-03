import os
import json
import base64
import re
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import requests as http_requests

# EasyOCR is imported lazily so the server starts fast.
# The reader is cached after first use.
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader


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


# ── OCR ENGINE ────────────────────────────────────────────────────────────────

def ocr_detect_text(image_bytes):
    """
    Run EasyOCR on the drawing image.
    Returns a list of detected text blocks:
    [
      {"text": "Ø34.93", "x": 42.1, "y": 31.5, "w": 4.2, "h": 1.8}
    ]
    x, y, w, h are percentages of the image dimensions.
    x and y are the center of the bounding box.
    """
    try:
        # Load image via Pillow to get dimensions
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = pil_img.size

        # EasyOCR works on a numpy array
        img_array = np.array(pil_img)

        reader = get_ocr_reader()
        # detail=1 returns bounding boxes; paragraph=False keeps individual words
        results = reader.readtext(img_array, detail=1, paragraph=False)

        blocks = []
        for (bbox, text, confidence) in results:
            if confidence < 0.2:
                continue
            # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (four corners)
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            # Convert to percentage of image size
            cx = ((x1 + x2) / 2) / img_w * 100
            cy = ((y1 + y2) / 2) / img_h * 100
            bw = (x2 - x1) / img_w * 100
            bh = (y2 - y1) / img_h * 100

            blocks.append({
                "text": text.strip(),
                "x": round(cx, 2),
                "y": round(cy, 2),
                "w": round(bw, 2),
                "h": round(bh, 2),
            })

        return blocks

    except Exception as e:
        # OCR failure is non-fatal — fall back to Claude estimates
        print(f"OCR error: {e}")
        return []


def normalize_for_matching(text):
    """
    Strip symbols and whitespace so "Ø34.93" matches "34.93",
    "2XR3.5" matches "R3.5" or "3.5", etc.
    """
    if not text:
        return ""
    t = text.upper()
    # Remove common drawing prefixes/symbols
    t = re.sub(r'[ØøOΩ°≤≥~±×X\s]', '', t)
    t = re.sub(r'^[RMC]+', '', t)   # strip leading R, M, C (radius, metric, coarse)
    t = t.replace('RA', '').replace('MM', '')
    return t.strip()


def find_balloon_position_via_ocr(matching_text, ocr_blocks):
    """
    Try to match matchingText to one of the OCR bounding boxes.
    Returns (x, y) as percentages if found, or None if not.

    Matching strategy (most-specific to least-specific):
      1. Exact string match (case-insensitive)
      2. Normalized numeric match (strip symbols, compare core number)
      3. Substring match in either direction
    """
    if not ocr_blocks or not matching_text:
        return None

    mt_clean = normalize_for_matching(matching_text)
    mt_lower = matching_text.lower().strip()

    best = None

    for block in ocr_blocks:
        raw = block.get("text", "")
        if not raw:
            continue

        raw_lower = raw.lower().strip()
        raw_clean = normalize_for_matching(raw)

        # 1. Exact match
        if raw_lower == mt_lower:
            return block["x"], block["y"]

        # 2. Normalized numeric core match
        if mt_clean and raw_clean and (mt_clean == raw_clean):
            best = (block["x"], block["y"])
            continue

        # 3. Substring match — matching_text contains OCR text or vice versa
        if mt_clean and raw_clean:
            if mt_clean in raw_clean or raw_clean in mt_clean:
                if best is None:
                    best = (block["x"], block["y"])

    return best


def offset_balloon(x, y, offset=2.5):
    """
    Nudge the balloon slightly away from the dimension text center
    so it sits beside the callout rather than on top of it.
    Prefers nudging right and up; falls back to left if near right edge.
    """
    nx = x + offset if x < 90 else x - offset
    ny = max(5, y - offset)
    return clamp(nx, 5, 95), clamp(ny, 5, 95)


# ── ROUTES ────────────────────────────────────────────────────────────────────

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
This is used by the OCR engine to locate the dimension callout precisely.

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

        # ── Step 2: OCR — detect all text blocks in the drawing ───────────
        print("Running OCR on drawing...")
        ocr_blocks = ocr_detect_text(image_bytes)
        print(f"OCR detected {len(ocr_blocks)} text blocks")

        # Build OCR position lookup keyed by characteristic id
        ocr_positions = {}
        for c in extracted["characteristics"]:
            cid = c.get("id")
            mt  = c.get("matchingText", "")
            pos = find_balloon_position_via_ocr(mt, ocr_blocks)
            if pos and pos[0] is not None and pos[1] is not None:
                ox, oy = offset_balloon(pos[0], pos[1])
                ocr_positions[cid] = {"x": ox, "y": oy}
                print(f"  OCR match: id={cid} matchingText='{mt}' → x={ox} y={oy}")
            else:
                print(f"  OCR miss:  id={cid} matchingText='{mt}' — will use Claude estimate")

        # ── Step 3: Claude vision estimates (fallback for OCR misses) ─────
        try:
            char_list = "\n".join(
                f"ID {c.get('id')}: {c.get('description')} = {c.get('nominal')} {c.get('unit', '')} | matchingText: \"{c.get('matchingText', '')}\""
                for c in extracted["characteristics"]
            )
            # Only ask Claude to estimate positions for IDs that OCR missed
            missing_ids = [
                c for c in extracted["characteristics"]
                if c.get("id") not in ocr_positions
            ]

            claude_positions = {}
            if missing_ids:
                missing_list = "\n".join(
                    f"ID {c.get('id')}: {c.get('description')} = {c.get('nominal')} {c.get('unit', '')} | matchingText: \"{c.get('matchingText', '')}\""
                    for c in missing_ids
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

These characteristics could not be located by OCR. Estimate their positions visually:
{missing_list}

RULES:
1. x and y are percentages of the full image. x=0 left, x=100 right. y=0 top, y=100 bottom.
2. ALL values must be between 5 and 95.
3. Find the dimension callout text on the drawing and place the coordinate 2-3% beside it.
4. Do NOT place balloons in the title block (bottom-right corner).
5. Do NOT place balloons outside the drawing frame.
6. Return exactly {len(missing_ids)} entries — one per ID listed above.

Return ONLY raw JSON:
{{"b":[{{"id":1,"matchingText":"Ø34.93","x":45,"y":30}}]}}"""
                            },
                        ],
                    }
                ], max_tokens=2000)

                raw_claude = extract_json(raw2).get("b", [])
                for b in raw_claude:
                    bid = b.get("id")
                    bx  = clamp(b.get("x"), 5, 95)
                    by  = clamp(b.get("y"), 5, 95)
                    if bid is not None and bx is not None and by is not None:
                        # Push out of title block zone
                        if bx > 60 and by > 72:
                            by = 70
                        claude_positions[bid] = {"x": bx, "y": by}

        except Exception as e:
            print(f"Claude balloon estimation error: {e}")
            claude_positions = {}

        # ── Step 4: Merge OCR + Claude positions into final balloon list ──
        balloons = []
        seen_ids = set()

        for c in extracted["characteristics"]:
            cid = c.get("id")
            mt  = c.get("matchingText", "")

            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            if cid in ocr_positions:
                # Best: OCR found the exact text on the drawing
                pos = ocr_positions[cid]
            elif cid in claude_positions:
                # Fallback: Claude vision estimate
                pos = claude_positions[cid]
            else:
                # Last resort: grid placement
                idx = len(balloons)
                pos = {
                    "x": clamp(5 + (idx % 8) * 11, 5, 95),
                    "y": clamp(6 + (idx // 8) * 14, 5, 95),
                }

            balloons.append({
                "id":          cid,
                "matchingText": mt,
                "x":           pos["x"],
                "y":           pos["y"],
            })

        balloons.sort(key=lambda b: b["id"])

        # ── Step 5: FAI form ──────────────────────────────────────────────
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
