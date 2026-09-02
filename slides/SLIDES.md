# Alzheimer's Multicloud MVP — demo slides

## 1. One model, three cloud choices

The MVP joins model quality with deployment cost and latency.

## 2. The integration seam is a JSON contract

`model_metrics.json` is produced by training and consumed by the decision
engine and dashboard. `cloud_recommendation.json` carries the ranked choice,
cost view, and ensemble weights into the UI.

## 3. Weighted prediction combines provider signals

`Final = α(AWS) + β(Azure) + γ(GCP)`

`integration/ensemble.py` accepts same-shaped dummy arrays now and model
probabilities later. Weights are normalized internally.

## 4. Trade-offs are visible in one view

Show the dashboard's Table 5-style comparison for accuracy, training time, and
latency, then show the provider cost breakdown.

## 5. Recommendation becomes an actionable prediction

Highlight the recommended provider, its rationale, the contributing provider
weights, and the final four-class prediction confidence.

## 6. The live handoff is one command

```text
python main.py --launch-dashboard
```

Person A and B can change their implementations behind the two JSON contracts;
the C dashboard and demo flow remain stable.

### Source note

The repository prompt supplies the model contract and cloud profile values.
The dashboard's Table 5-style rows are marked as mock data until the team adds
the paper's exact Table 5 figures and citation.
