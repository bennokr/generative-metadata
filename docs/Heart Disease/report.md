# Data Report — Heart Disease

**Source**: [UCI dataset 45](https://archive.ics.uci.edu/dataset/45)

- Metadata file: [metadata.json](metadata.json)
- JSON-LD (schema.org/Dataset): [dataset.json](dataset.json)
- SemMap JSON-LD: [dataset.semmap.json](dataset.semmap.json)
- SemMap HTML: [dataset.semmap.html](dataset.semmap.html)
- Rows: 297
- Columns: 14
- Discrete: 7  |  Continuous: 7

## Dataset metadata

- Name: Heart Disease (UCI id 45)

### Description

4 databases: Cleveland, Hungary, Switzerland, and the VA Long Beach

- Links:
  - URL: https://archive.ics.uci.edu/dataset/45
## Variables and summary

| variable   | description                                           | inferred   | declared    |   count | unique   | top   | freq   | mean               | std                | min   | 25%   | 50%   | 75%   | max   |
|:-----------|:------------------------------------------------------|:-----------|:------------|--------:|:---------|:------|:-------|:-------------------|:-------------------|:------|:------|:------|:------|:------|
| age        |                                                       | continuous | Integer     |     303 |          |       |        | 54.43894389438944  | 9.038662442446746  | 29.0  | 48.0  | 56.0  | 61.0  | 77.0  |
| sex        |                                                       | discrete   | Categorical |     303 | 2        | 1     | 206    |                    |                    |       |       |       |       |       |
| cp         |                                                       | discrete   | Categorical |     303 | 4        | 4     | 144    |                    |                    |       |       |       |       |       |
| trestbps   | resting blood pressure (on admission to the hospital) | continuous | Integer     |     303 |          |       |        | 131.68976897689768 | 17.59974772958769  | 94.0  | 120.0 | 130.0 | 140.0 | 200.0 |
| chol       | serum cholestoral                                     | continuous | Integer     |     303 |          |       |        | 246.69306930693068 | 51.77691754263704  | 126.0 | 211.0 | 241.0 | 275.0 | 564.0 |
| fbs        | fasting blood sugar > 120 mg/dl                       | discrete   | Categorical |     303 | 2        | 0     | 258    |                    |                    |       |       |       |       |       |
| restecg    |                                                       | discrete   | Categorical |     303 | 3        | 0     | 151    |                    |                    |       |       |       |       |       |
| thalach    | maximum heart rate achieved                           | continuous | Integer     |     303 |          |       |        | 149.6072607260726  | 22.875003276980376 | 71.0  | 133.5 | 153.0 | 166.0 | 202.0 |
| exang      | exercise induced angina                               | discrete   | Categorical |     303 | 2        | 0     | 204    |                    |                    |       |       |       |       |       |
| oldpeak    | ST depression induced by exercise relative to rest    | continuous | Integer     |     303 |          |       |        | 1.0396039603960396 | 1.1610750220686348 | 0.0   | 0.0   | 0.8   | 1.6   | 6.2   |
| slope      |                                                       | discrete   | Categorical |     303 | 3        | 1     | 142    |                    |                    |       |       |       |       |       |
| ca         | number of major vessels (0-3) colored by flourosopy   | continuous | Integer     |     299 |          |       |        | 0.6722408026755853 | 0.9374383177242163 | 0.0   | 0.0   | 0.0   | 1.0   | 3.0   |
| thal       |                                                       | continuous | Categorical |     301 |          |       |        | 4.73421926910299   | 1.939705769378644  | 3.0   | 3.0   | 3.0   | 7.0   | 7.0   |
| num        | diagnosis of heart disease                            | discrete   | Integer     |     303 | 5        | 0     | 164    |                    |                    |       |       |       |       |       |

## Models

| name     | backend   |   rows |   seed |   disc_jsd_mean |   disc_jsd_median |   cont_ks_mean |   cont_w1_mean |
|:---------|:----------|-------:|-------:|----------------:|------------------:|---------------:|---------------:|
| clg_mi2  | pybnesian |    237 |     42 |        0.100273 |         0.0995307 |       0.234388 |        4.41091 |
| semi_mi5 | pybnesian |    237 |     42 |        0.100273 |         0.0995307 |       0.234388 |        4.41091 |

### Model: clg_mi2 (pybnesian)

- Seed: 42
- Rows: 237
- Params: `{"max_indegree": 2, "operators": ["arcs"], "score": "bic", "type": "clg"}`
- Metrics: disc_jsd_mean=0.1003, disc_jsd_median=0.0995, cont_ks_mean=0.2344, cont_w1_mean=4.4109
- Synthetic CSV: [models/clg_mi2/synthetic.csv](models/clg_mi2/synthetic.csv)
- Per-variable metrics: [models/clg_mi2/per_variable_metrics.csv](models/clg_mi2/per_variable_metrics.csv)
- Metrics JSON: [models/clg_mi2/metrics.json](models/clg_mi2/metrics.json)
- UMAP: [umap.png](models/clg_mi2/umap.png)
### Model: semi_mi5 (pybnesian)

- Seed: 42
- Rows: 237
- Params: `{"max_indegree": 5, "operators": ["arcs"], "score": "bic", "type": "semiparametric"}`
- Metrics: disc_jsd_mean=0.1003, disc_jsd_median=0.0995, cont_ks_mean=0.2344, cont_w1_mean=4.4109
- Synthetic CSV: [models/semi_mi5/synthetic.csv](models/semi_mi5/synthetic.csv)
- Per-variable metrics: [models/semi_mi5/per_variable_metrics.csv](models/semi_mi5/per_variable_metrics.csv)
- Metrics JSON: [models/semi_mi5/metrics.json](models/semi_mi5/metrics.json)
- UMAP: [umap.png](models/semi_mi5/umap.png)
MetaSyn GMF: [metasyn_gmf.json](metasyn_gmf.json)

MetaSyn serialization

- Synthetic sample (SemMap Parquet): [synthetic_metasyn.semmap.parquet](synthetic_metasyn.semmap.parquet)

## Fidelity (MetaSyn)

| model   |   disc_jsd_mean |   disc_jsd_median |   cont_ks_mean |   cont_w1_mean |
|:--------|----------------:|------------------:|---------------:|---------------:|
| MetaSyn |          0.1053 |            0.1034 |         0.2903 |         2.5822 |

## UMAP overview (same projection)

| Real (sample) | MetaSyn (synthetic) | pybnesian: clg_mi2 | pybnesian: semi_mi5 |
| --- | --- | --- | --- |
| <img src='umap_real.png' width='280'/> | <img src='umap_metasyn.png' width='280'/> | <img src='umap.png' width='280'/> | <img src='umap.png' width='280'/> |

