import os

from flask import Flask, render_template, request, jsonify
import rintag

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tag", methods=["POST"])
def tag_text():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = rintag.tag(text)
    
    # Serialize the TagResult into a list of dicts for JSON
    serialized = []
    for tok in result:
        serialized.append({
            "token": tok.token,
            "tag": tok.tag,
            "confidence": tok.confidence,
            "alternatives": tok.alternatives[:5],  # Just return top 5 in demo
            "features": tok.features
        })
        
    return jsonify({"tokens": serialized})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
