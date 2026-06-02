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
                           "text": f"""You are analyzing an engineering drawing to find the exact screen position of dimension callouts.

I have extracted these characteristics from this drawing:
{char_list}

Your task:
1. Visually scan the drawing for each dimension text label (e.g. "95.25", "Ø12.7", "R6.35", "1.6", "47.6").
2. Find where that exact number appears as a dimension callout on the drawing — not in the title block, not in the tolerance table, but on the actual geometry.
3. Return the x/y position of that dimension text as a percentage of the total image size.
  - x=0 is the left edge, x=100 is the right edge
  - y=0 is the top edge, y=100 is the bottom edge
4. Place the coordinate slightly to the right or above the dimension text so a balloon circle will sit beside it, not on top of it. Offset by roughly 2-3% from the text center.
5. If a dimension appears with a multiplier (e.g. "4X R6.35" or "6XØ12.7"), return the position of that single callout label, not each individual feature.
6. If you cannot locate a specific dimension on the drawing, make your best estimate based on where that type of feature would logically appear.
7. Stay within the drawing area — do not place coordinates in the title block (bottom-right corner) or tolerance table.

Return ONLY raw JSON. No markdown. No explanation:
{{"b":[{{"id":1,"x":45,"y":30}},{{"id":2,"x":62,"y":55}}]}}

Every characteristic ID must have an entry. Total entries required: {len(extracted["characteristics"])}"""
                       },
                   ],
               }
           ], max_tokens=2000)

           raw_balloons = extract_json(raw2).get("b", [])

           # Validate, deduplicate, clamp to drawing area
           balloons = []
           seen_ids = set()
           for b in raw_balloons:
               bid = b.get("id")
               bx = b.get("x", 50)
               by = b.get("y", 50)
               if bid is None or bid in seen_ids:
                   continue
               seen_ids.add(bid)
               # Clamp to safe drawing area
               bx = max(2, min(bx, 98))
               by = max(2, min(by, 98))
               # If coordinate lands in title block (bottom-right), shift up
               if bx > 60 and by > 72:
                   by = 70
               balloons.append({"id": bid, "x": bx, "y": by})

           # Fill in any IDs the model missed
           returned_ids = {b["id"] for b in balloons}
           for i, c in enumerate(extracted["characteristics"]):
               cid = c.get("id")
               if cid not in returned_ids:
                   balloons.append({
                       "id": cid,
                       "x": 5 + (i % 8) * 11,
                       "y": 6 + (i // 8) * 14,
                   })

           balloons.sort(key=lambda b: b["id"])

       except Exception:
           balloons = [
               {
                   "id": c.get("id"),
                   "x": 5 + (i % 8) * 11,
                   "y": 6 + (i // 8) * 14,
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
