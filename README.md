# Multi-cloud Alzheimer's analytics

Person C owns the integration seam, mock-first dashboard, and demo flow.

## Run the C demo

```powershell
python -m pip install -r requirements.txt
python main.py --mock
streamlit run dashboard/app.py
```

The dashboard uses seeded mock data until both upstream contracts exist:

- `outputs/model_metrics.json` — produced by Person A.
- `outputs/cloud_recommendation.json` — produced by `main.py` after calling
  Person B's `cloud_engine/decision_engine.py`.

To run the live A → B → dashboard path:

```powershell
python main.py --launch-dashboard
```

See [DEMO.md](DEMO.md) for the talk track and [slides/SLIDES.md](slides/SLIDES.md)
for the slide outline.
