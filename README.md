# GreenIndex -> Chlorophyll Deficiency Detection System

A complete end-to-end system extending the formerly known GreenLeafVI project to perform white balancing, segmentation, vegetation index extraction using RGB data, and chlorophyll deficiency visualization (heathmap generation).

## Setup & Installation

1. Prepare your environment (create a virtual environment and activate it):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Start the GreenIndex FastAPI server locally:

```bash
uvicorn app.main:app --reload
```

Then, you can POST images to the `/analyze` endpoint to receive a structured RGB evaluation, health status, and heatmap of the leaf tissue.
Example:
```bash
curl -X POST "http://0.0.0.0:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/dummy_test_leaf.png;type=image/png"
```

