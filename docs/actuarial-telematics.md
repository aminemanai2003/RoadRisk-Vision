# Actuarial telematics exposure example

RoadRisk Vision produces candidate explanatory features, not actuarial outcomes.
The `exposure-report` command aggregates completed local runs:

```powershell
roadrisk exposure-report runs/trip-one runs/trip-two `
  --low-ttc-s 2.0 --output reports/portfolio_exposure
```

Outputs are versioned JSON, one-row CSV and a plain-language Markdown report.
They contain trip count, exposure hours, valid kilometres, risk-event counts by
type/severity, rates per 100 hours, eligible rates per 100 km and a separately
defined low-TTC count.

Per-100-km results include only trips with at least 80% GPS coverage and 1 km of
valid distance. Low TTC means the event's measured minimum TTC is at or below
the chosen threshold; it is not automatically a near miss, accident or claim.

## Genuine actuarial continuation

An actuary could join these governed telematics features to independently held
policy exposure and claims data, then test predictive value, selection effects,
credibility, fairness, stability and data quality. That later analysis needs
appropriate consent, jurisdictional review and leakage controls. RoadRisk Vision
does not join claims, estimate claim frequency/severity, set premiums or make
underwriting decisions.

`examples/actuarial_exposure_report.py` shows the equivalent Python API using
synthetic/local run paths.
