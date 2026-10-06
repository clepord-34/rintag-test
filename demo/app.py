import os

from flask import Flask, render_template, request, jsonify
import rintag

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tag", methods=["POST"])
def tag_text():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    results = rintag.tag_detailed(text, top_k=5)
    return jsonify({"tokens": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
