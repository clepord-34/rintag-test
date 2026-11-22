import os
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import gspread
import rintag

try:
    from dotenv import load_dotenv
    # Search for .env in project root, demo dir, or cwd
    for p in [Path(__file__).resolve().parents[1] / ".env", Path(__file__).resolve().parent / ".env", Path(".env")]:
        if p.exists():
            load_dotenv(dotenv_path=p, override=True)
            break
except ImportError:
    pass

def _load_env_fallback():
    """Pure-Python fallback to read .env if python-dotenv is missing."""
    for p in [Path(__file__).resolve().parents[1] / ".env", Path(__file__).resolve().parent / ".env", Path(".env")]:
        if p.exists():
            try:
                content = p.read_text(encoding="utf-8")
                if "FEEDBACK_SHEET_URL" not in os.environ:
                    for line in content.splitlines():
                        if line.strip().startswith("FEEDBACK_SHEET_URL="):
                            os.environ["FEEDBACK_SHEET_URL"] = line.split("=", 1)[1].strip().strip("'\"")
                            break
                if "GOOGLE_CREDENTIALS" not in os.environ:
                    parts = content.split("GOOGLE_CREDENTIALS=", 1)
                    if len(parts) > 1:
                        val = parts[1].strip()
                        if val.startswith("'"):
                            val = val[1:].split("'", 1)[0]
                        elif val.startswith('"'):
                            val = val[1:].split('"', 1)[0]
                        else:
                            val = val.splitlines()[0].strip()
                        os.environ["GOOGLE_CREDENTIALS"] = val
            except Exception:
                pass
            break

_load_env_fallback()

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ── Google Sheets connection (lazy) ──
_sheet = None
_sheet_error = None

FEEDBACK_HEADERS = [
    "sentence_id", "timestamp", "full_sentence", "token_index",
    "token", "model_tag", "user-corrected_tag", "confidence",
    "is_oov", "was_corrected", "validator_tag"
]


def _get_sheet():
    """Lazily connect to the feedback Google Sheet."""
    global _sheet, _sheet_error
    if _sheet is not None:
        return _sheet

    creds_val = os.environ.get("GOOGLE_CREDENTIALS", "").strip()
    sheet_url = os.environ.get("FEEDBACK_SHEET_URL", "").strip()
    if not creds_val or not sheet_url:
        _sheet_error = "Missing GOOGLE_CREDENTIALS or FEEDBACK_SHEET_URL in .env file or environment variables."
        return None
    try:
        # Support both direct JSON string and local file path
        if os.path.isfile(creds_val):
            gc = gspread.service_account(filename=creds_val)
        else:
            creds_info = json.loads(creds_val)
            gc = gspread.service_account_from_dict(creds_info)
        _sheet = gc.open_by_url(sheet_url).sheet1
        # Ensure header row exists
        if not _sheet.row_values(1):
            _sheet.append_row(FEEDBACK_HEADERS)
        _sheet_error = None
        return _sheet
    except Exception as e:
        _sheet_error = str(e)
        app.logger.error(f"Failed to connect to Google Sheets: {e}")
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/tag", methods=["POST"])
def tag_text():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = rintag.tag(text)
    serialized = []
    for tok in result:
        serialized.append({
            "token": tok.token,
            "tag": tok.tag,
            "confidence": tok.confidence,
            "alternatives": tok.alternatives[:5],
            "features": tok.features,
            "is_oov": tok.is_oov,
        })
    return jsonify({"tokens": serialized})


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Write complete sentence tokens with user corrections to Google Sheets."""
    data = request.get_json() or {}
    tokens = data.get("tokens", [])
    sentence = data.get("original_text", "").strip()
    if not tokens or not sentence:
        return jsonify({"error": "No sentence or tokens provided"}), 400

    sheet = _get_sheet()
    if sheet is None:
        return jsonify({
            "error": f"Feedback storage error: {_sheet_error or 'Not configured'}"
        }), 503

    sentence_id = f"demo_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    rows = []
    for idx, t in enumerate(tokens):
        was_corrected = bool(t.get("was_corrected", False))
        user_tag = t.get("corrected_tag", "") if was_corrected else ""
        rows.append([
            sentence_id,
            timestamp,
            sentence,
            idx,
            t.get("token", ""),
            t.get("tag", ""),
            user_tag,
            t.get("confidence", 0.0),
            t.get("is_oov", False),
            was_corrected,
            "",  # validator_tag is maintained separately from user feedback
        ])

    sheet.append_rows(rows, value_input_option="RAW")
    return jsonify({
        "status": "ok",
        "sentence_id": sentence_id,
        "tokens_saved": len(rows),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
