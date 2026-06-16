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

        # ── Step 2: Return extracted data ────────────────────────────────
        # Balloon placement and FAI form generation happen after the user
        # places balloons manually. The /api/identify-balloons endpoint
        # handles both — Claude reads the drawing at each tapped position
        # and returns the characteristic data + form rows in one shot.
        return jsonify({
            "extracted": extracted,
            "balloons":  [],
            "formData":  None,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/identify-balloons", methods=["POST"])
def identify_balloons():
    """
    Called after the user finishes placing balloons.
    Receives the drawing image + list of {id, x, y} positions.
    Claude looks at each position and identifies the nearest dimension,
    returning a complete FAI form row for each balloon.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f = request.files["file"]
        mime = f.mimetype or "image/jpeg"
        b64 = base64.b64encode(f.read()).decode()

        raw_positions = request.form.get("positions")
        if not raw_positions:
            return jsonify({"error": "No balloon positions provided"}), 400
        positions = json.loads(raw_positions)

        raw_part_info = request.form.get("partInfo")
        part_info = json.loads(raw_part_info) if raw_part_info else {}

        raw_tolerances = request.form.get("tolerances")
        tolerances = json.loads(raw_tolerances) if raw_tolerances else {}

        # Build position list for the prompt
        pos_list = "\n".join(
            f"Balloon #{p['id']}: x={p['x']}%, y={p['y']}% of image"
            for p in positions
        )

        raw = call_claude([
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
                        "text": f"""You are a senior metrologist identifying dimensions on an engineering drawing. Your job is to determine EXACTLY what each balloon position is measuring by reading the dimension lines, arrowheads, and witness lines at that location — not just the nearest number.

COORDINATE SYSTEM: x=0% left edge, x=100% right edge, y=0% top edge, y=100% bottom edge.

General tolerances from title block:
{json.dumps(tolerances)}

Balloon positions to identify:
{pos_list}

═══ HOW TO IDENTIFY EACH DIMENSION CORRECTLY ═══

For every balloon position, follow this process:

STEP 1 — Find the dimension line at that location.
A dimension line has arrowheads at both ends pointing to what is being measured.
Look at what the two arrowheads are pointing TO — not just the number.

STEP 2 — Determine what the arrowheads point to:
  A) Both arrowheads point to EDGES or SURFACES of the part
     → This is an overall or feature dimension (length, width, height, thickness, slot width, flange height, etc.)

  B) One arrowhead points to a HOLE CENTER (centerline crossing), other points to an EDGE
     → This is "Hole center to cut edge" — specify which edge (left, right, top, bottom)

  C) Both arrowheads point to HOLE CENTERS (centerlines crossing)
     → This is "Hole center to hole center" — specify horizontal or vertical

  D) One arrowhead points to a HOLE CENTER, other points to a BEND LINE
     → This is "Hole center to bend"

  E) Arrow points to a RADIUS with a leader line
     → This is a radius — specify inside bend radius, corner radius, or fillet radius

  F) Arrow points to a CIRCLE or HOLE with a leader line and Ø symbol
     → This is a diameter

  G) Feature control frame (box with GD&T symbol)
     → Extract the GD&T type, tolerance value, and datum references

STEP 3 — Read the numeric value shown between or near the dimension line.

STEP 4 — Determine tolerance: use the value shown on that specific dimension first.
If no tolerance shown, apply the correct tier from the general tolerance table.

═══ CRITICAL DISTINCTION — HOLE CENTER vs EDGE ═══
- If a dimension line has one end on a CENTERLINE CROSS (hole center), it is a hole center dimension
- Centerlines appear as alternating long-short dashed lines through hole centers
- If BOTH ends are on centerlines → hole center to hole center
- If ONE end is on a centerline and ONE end is on a solid edge line → hole center to edge

═══ DESCRIPTION FORMAT ═══
- "Overall length" / "Overall width" / "Overall height"
- "Sheet thickness" / "Wall thickness" / "Plate thickness"
- "Hole diameter Nx (ØX.XX)"
- "Hole center to cut edge — left" / "— right" / "— top" / "— bottom"
- "Hole center to hole center — horizontal" / "— vertical"
- "Hole center to bend — horizontal" / "— vertical"
- "Inside bend radius Nx (RX.XX)"
- "Corner radius Nx (RX.XX)"
- "Slot width" / "Slot length" / "Slot depth"
- "Slot corner radius Nx (RX.XX)"
- "Flange height" / "Flange width" / "Leg height"
- "Chamfer X × X°"
- "Bend angle"
- "True position Ø X.XX |datums|"
- "Flatness X.XX"
- "Perpendicularity Ø X.XX |datum|"
- "Parallelism X.XX |datum|"
- "Runout X.XX |datum|"
- "Surface roughness Ra X.X μm"
- "Material: [spec]"
- "Finish: [spec]"

═══ TOLERANCE RULES ═══
- Specific tolerance on dimension → use that value
- No specific tolerance → apply correct tier from general tolerance table
- Format: tolPlus="+0.20", tolMinus="-0.20"
- GD&T: tolPlus="0.25", tolMinus="0.00"
- Material/finish: tolPlus="", tolMinus=""
- Cannot identify: desc="Verify manually", nominal="", tolPlus="", tolMinus=""

═══ INSPECTION METHOD (notes field) ═══
Caliper · Micrometer · CMM · Height gauge · Optical comparator · Pin gauge · Go/no-go gauge · Ra profilometer · Visual · Cert required

Return ONLY raw JSON — no markdown:
{{
  "rows": [
    {{
      "n": 1,
      "desc": "Overall length",
      "nominal": "95.25",
      "tolPlus": "+0.20",
      "tolMinus": "-0.20",
      "notes": "Caliper"
    }}
  ],
  "certs": ["Material Certification — AL5052-H32"]
}}

Total rows required: {len(positions)}. Every balloon must have exactly one row."""
                    },
                ],
            }
        ], max_tokens=4000)

        form_data = extract_json(raw)
        if not form_data.get("rows"):
            return jsonify({"error": "Could not identify dimensions at balloon positions"}), 422

        # Build s1 from partInfo
        form_data["s1"] = {
            "partNumber":   part_info.get("partNumber", ""),
            "partName":     part_info.get("partName", ""),
            "revision":     part_info.get("revision", ""),
            "drawingNumber": part_info.get("drawingNumber", ""),
            "material":     part_info.get("material", ""),
            "finish":       part_info.get("finish", ""),
            "date":         part_info.get("date", ""),
            "quantity":     "1",
        }

        return jsonify({"formData": form_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"\n✅ Running at http://localhost:{port}")
    print(f"   API key: {'YES' if API_KEY else 'NO - check environment variables'}")
    print(f"   Claude model: {CLAUDE_MODEL}\n")
    app.run(host="0.0.0.0", port=port)
