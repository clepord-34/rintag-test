# Demonstration & Deployment Guide

This document describes how to run and deploy the RinTag interactive web demo and uncertainty showcase.

---

## 1. Web Application (`demo/app.py`)

The demo web application provides an interactive interface for tagging Rinconada Bikol sentences, viewing token confidences and alternative predictions, and submitting corrections directly to an attached Google Sheet.

### Local Execution

```bash
# 1. Navigate to repository root and install dependencies
pip install -r demo/requirements.txt
pip install -e .

# 2. Configure environment variables (optional for local testing)
cp .env.example .env
# Edit .env with your Google Sheets credentials if feedback saving is required

# 3. Launch Flask development server
python demo/app.py
```
Visit `http://localhost:5000` in your web browser.

---

## 2. Environment Variables & Google Sheets Integration

The demo supports automated collection of user corrections into Google Sheets:

| Variable | Description |
|---|---|
| `FEEDBACK_SHEET_URL` | Google Sheet URL where community tag corrections are appended. |
| `GOOGLE_CREDENTIALS` | Service account JSON string or path to service account credentials file. |
| `PORT` | Web server listening port (defaults to `5000`). |

If Google credentials are not configured, the application functions normally for POS tagging; feedback submission returns a graceful configuration notice.

---

## 3. Production Deployment (Render)

The repository includes `render.yaml` for zero-configuration deployment on [Render](https://render.com):

```yaml
services:
  - type: web
    name: rintag-demo
    runtime: python
    buildCommand: pip install -r demo/requirements.txt
    startCommand: gunicorn --chdir demo app:app
    plan: free
    envVars:
      - key: GOOGLE_CREDENTIALS
        sync: false
      - key: FEEDBACK_SHEET_URL
        sync: false
```

To deploy:
1. Connect your GitHub repository to Render.
2. Select **New Blueprint Instance** and link to `render.yaml`.
3. Add the `GOOGLE_CREDENTIALS` and `FEEDBACK_SHEET_URL` secret environment variables in the Render Dashboard.

---

## 4. Maintainer Uncertainty Sampling Showcase

Located in `demo/hitl_showcase/`:

```bash
# Run batch uncertainty sampling across ~18K unseen corpus sentences
python demo/hitl_showcase/hitl_feedback_demo.py --top-n 25

# Demonstrate passive background collection in an end-user workflow
python demo/hitl_showcase/demo_passive_feedback.py
```
Output files are generated in `demo/hitl_showcase/output/` (gitignored). See [demo/hitl_showcase/README.md](file:///c:/Users/clepord/Desktop/RinTag%20Thesis/RinTag/demo/hitl_showcase/README.md) for flag details.
