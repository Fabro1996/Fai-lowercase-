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
