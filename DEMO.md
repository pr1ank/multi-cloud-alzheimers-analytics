# Person C demo handoff

## Communication job

By the end of the demo, the audience should understand that one integration
layer can combine model outputs, compare cloud trade-offs, and surface a final
prediction without coupling the dashboard to a specific model provider.

## Run it

From the repository root:

```powershell
python -m pip install -r requirements.txt
python main.py --mock
streamlit run dashboard/app.py
```

The dashboard starts in mock mode and does not require the Kaggle dataset,
Person A's saved model, or Person B's decision engine. After both upstream
tracks produce their agreed JSON files, run:

```powershell
python main.py --launch-dashboard
```

This reads `outputs/model_metrics.json`, calls
`cloud_engine/decision_engine.py`, writes
`outputs/cloud_recommendation.json`, and launches the same dashboard. Workload
inputs can be changed with `--data-size-gb`, `--latency-requirement-ms`, and
`--budget-usd`.

## Three-minute talk track

1. **Frame the problem (20 sec).** “The model produces a diagnosis signal, but
   deployment still has a cloud cost, latency, and provider-selection problem.”

2. **Show the contract (30 sec).** Point to the model metrics JSON and explain
   that C consumes the stable fields rather than reaching into training code.

3. **Show the combiner (30 sec).** Run the dummy prediction example from
   `integration/ensemble.py`: provider probability arrays are normalized and
   combined by the provider weights.

4. **Show the dashboard (60 sec).** Walk left-to-right: Table 5-style
   accuracy/training-time/latency comparison, cost stack, recommended provider,
   and final class confidence.

5. **Switch to live wiring (30 sec).** Add A/B outputs under `outputs/`, run
   `python main.py --launch-dashboard`, and point out that the UI code stays
   unchanged.

6. **Close (10 sec).** “The deliverable is the seam: A can improve the model,
   B can change the ranking policy, and C keeps the end-to-end experience
   stable.”

## Presenter notes

- The seeded chart is explicitly labeled mock-first. Replace the two seeded
  Table 5-style rows with the paper's exact cited figures once the paper source
  is available to the team.
- The final prediction is a deterministic dummy four-class probability vector;
  it is a UI/integration demonstration, not a clinical prediction.
- `combine()` returns a weighted average and normalizes weights internally, so
  Person B can pass normalized score weights or raw positive scores.
