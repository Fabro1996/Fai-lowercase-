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

Your job is to extract only characteristics that would appear on a real AS9102 First Article Inspection report — items an inspector physically measures or verifies on the actual part.

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
- Any text that is metadata, not a dimension
- Duplicate characteristics with the same type and nominal value

TOLERANCE RULES:
- If the drawing shows a specific tolerance for the dimension, use that exact value (e.g. "±0.12")
- If no specific tolerance is shown, look at the title block general tolerance table and apply the correct tier based on the nominal value range
- Do NOT write "per title block" — always resolve to the actual numeric tolerance
- Format tolerance as "±0.20" not "+0.20/-0.20"

HOLE PATTERNS:
- If multiple holes share the same diameter, create ONE characteristic for the diameter with description like "Hole diameter (6x)"
- Create separate characteristics for the hole center spacing and edge distances shown by dimension lines

DESCRIPTIONS:
Use plain inspector language: "Plate width", "Plate height", "Sheet thickness", "Hole diameter", "Hole center spacing", "Flange height", "Bend radius", "Slot width", "Slot length", "Outer diameter", "Bore diameter"

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

       # ── Step 2: Balloon placement ─────────────────────────────────────
       try:
           char_list = "\n".join(
               f"ID {c.get('id')}: {c.get('description')} = {c.get('nominal')} {c.get('unit', '')} (tolerance: {c.get('tolerance', '')})"
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

These characteristics were extracted from the drawing:
{char_list}

CRITICAL RULES — read carefully before responding:

1. x and y are percentages of the full image size. x=0 is the left edge, x=100 is the right edge. y=0 is the top edge, y=100 is the bottom edge.

2. ALL coordinates must be between 5 and 95. Never return a value below 5 or above 95 for either x or y.

3. Locate the actual dimension text on the drawing for each characteristic. Look for the numeric value as a callout with dimension lines.

4. Place the balloon 2-4% away from the dimension text — beside it, not on top of it.

5. The title block is in the bottom-right corner of the drawing. Do NOT place any balloons there. If a dimension only appears in the title block, place the balloon near the relevant geometry instead.

6. The drawing border, zone labels (A B C D, 1 2 3 4 5 6), and any area outside the drawing frame are OFF LIMITS. All balloons must be inside the drawing frame.

7. If you are uncertain where a dimension is, place the balloon in the approximate area of the drawing where that type of feature would be — never outside the drawing.

8. You must return exactly {len(extracted["characteristics"])} entries — one for every ID listed above.

Return ONLY raw JSON. No markdown. No explanation:
{{"b":[{{"id":1,"x":45,"y":30}},{{"id":2,"x":62,"y":55}}]}}"""
                       },
                   ],
               }
           ], max_tokens=2000)

           raw_balloons = extract_json(raw2).get("b", [])

           # Safety clamp — enforce all coordinates are within 5–95
           balloons = []
           seen_ids = set()
           for b in raw_balloons:
               bid = b.get("id")
               if bid is None or bid in seen_ids:
                   continue
               seen_ids.add(bid)

               bx = clamp(b.get("x"), 5, 95)
               by = clamp(b.get("y"), 5, 95)

               # If clamp returned None (missing or non-numeric), use fallback
               if bx is None or by is None:
                   idx = next((i for i, c in enumerate(extracted["characteristics"]) if c.get("id") == bid), len(balloons))
                   bx = 5 + (idx % 8) * 11
                   by = 6 + (idx // 8) * 14
                   bx = clamp(bx, 5, 95)
                   by = clamp(by, 5, 95)

               # Push balloons out of title block zone (bottom-right)
               if bx > 60 and by > 72:
                   by = 70

               balloons.append({"id": bid, "x": bx, "y": by})

           # Add fallback for any IDs the model missed entirely
           returned_ids = {b["id"] for b in balloons}
           for i, c in enumerate(extracted["characteristics"]):
               cid = c.get("id")
               if cid not in returned_ids:
                   fx = clamp(5 + (i % 8) * 11, 5, 95)
                   fy = clamp(6 + (i // 8) * 14, 5, 95)
                   balloons.append({"id": cid, "x": fx, "y": fy})

           balloons.sort(key=lambda b: b["id"])

       except Exception:
           balloons = [
               {
                   "id": c.get("id"),
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
