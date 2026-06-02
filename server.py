import os, json, base64
from flask import Flask, request, jsonify, send_from_directory
import requests

app = Flask(__name__, static_folder="public")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def call_claude(messages, max_tokens=4000):
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={"model": "claude-sonnet-4-20250514", "max_tokens": max_tokens, "messages": messages},
        timeout=120,
    )
    data = resp.json()
    if not resp.ok:
        raise Exception(f"API {resp.status_code}: {data}")
    return "".join(b.get("text", "") for b in data.get("content", []))

def extract_json(raw):
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e == -1:
        raise Exception("No JSON found: " + raw[:200])
    return json.loads(raw[s:e+1])

@app.route("/")
def index():
    return send_from_directory("public", "index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    mime = f.mimetype or "image/jpeg"
    b64 = base64.b64encode(f.read()).decode()

    raw1 = call_claude([{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
        {"type": "text", "text": """Analyze this engineering drawing for First Article Inspection. Extract ALL characteristics.
Return ONLY raw JSON (no markdown):
{"partInfo":{"partNumber":"","partName":"","revision":"","material":"","finish":"","drawingNumber":"","date":"","drawnBy":"","checkedBy":"","approvedBy":"","company":"","units":"mm"},"generalNotes":[],"tolerances":{"linear":"","angular":"","hole":"","surfaceRoughness":""},"characteristics":[{"id":1,"type":"diameter","description":"Outer diameter","nominal":"34.93","unit":"mm","tolerance":"per title block","view":"front","priority":"standard"}]}
Replace the example with ALL real characteristics. Extract every dimension, diameter, radius, angle, material, finish, surface roughness, and note."""}
    ]}])
    extracted = extract_json(raw1)
    if not extracted.get("characteristics"):
        return jsonify({"error": "No characteristics found"}), 422

    try:
        raw2 = call_claude([{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
            {"type": "text", "text": f"Give approximate position of each characteristic as % of image (x=left-right 0-100, y=top-bottom 0-100).\nCharacteristics: {', '.join(f\"{c['id']}:{c['description']}({c['nominal']})\" for c in extracted['characteristics'])}\nReturn ONLY raw JSON: {{\"b\":[{{\"id\":1,\"x\":45,\"y\":30}}]}}"}
        ]}], max_tokens=2000)
        balloons = extract_json(raw2).get("b", [])
    except:
        balloons = [{"id": c["id"], "x": 5 + (i % 8) * 11, "y": 6 + (i // 8) * 14}
                    for i, c in enumerate(extracted["characteristics"])]

    raw3 = call_claude([{"role": "user", "content":
        f"""Fill an AS9102 FAI form.
partInfo: {json.dumps(extracted['partInfo'])}
characteristics: {json.dumps(extracted['characteristics'])}
Return ONLY raw JSON:
{{"s1":{{"partNumber":"","partName":"","revision":"","drawingNumber":"","material":"","finish":"","date":"","quantity":"1"}},"rows":[{{"n":1,"desc":"what measured","nominal":"val","tolPlus":"+val","tolMinus":"-val","notes":""}}],"certs":["Material Certification"]}}
One row per characteristic."""}])
    form_data = extract_json(raw3)
    if not form_data.get("rows"):
        return jsonify({"error": "Form generation failed"}), 422

    return jsonify({"extracted": extracted, "balloons": balloons, "formData": form_data})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"\n✅ Running at http://localhost:{port}")
    print(f"   API key: {'YES' if API_KEY else 'NO - check environment variables'}\n")
    app.run(host="0.0.0.0", port=port)
