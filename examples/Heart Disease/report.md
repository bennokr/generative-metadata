# Data Report — Heart Disease

- Metadata file: [metadata.json](metadata.json)
- JSON-LD (schema.org/Dataset): [dataset.jsonld](dataset.jsonld)
- Rows: 297
- Columns: 14
- Discrete: 7  |  Continuous: 7

## Dataset metadata


### Description

4 databases: Cleveland, Hungary, Switzerland, and the VA Long Beach

- Creators: Andras Janosi, William Steinbrunn, Matthias Pfisterer, Robert Detrano

### Variables

| variable   | measurement   | unit   |
|:-----------|:--------------|:-------|
| age        | continuous    |        |
| sex        | discrete      |        |
| cp         | discrete      |        |
| trestbps   | continuous    |        |
| chol       | continuous    |        |
| fbs        | discrete      |        |
| restecg    | discrete      |        |
| thalach    | continuous    |        |
| exang      | discrete      |        |
| oldpeak    | continuous    |        |
| slope      | discrete      |        |
| ca         | continuous    |        |
| thal       | continuous    |        |
| num        | discrete      |        |

## Baseline summary

| variable   |   count | unique   | top   | freq   | mean               | std                | min   | 25%   | 50%   | 75%   | max   |
|:-----------|--------:|:---------|:------|:-------|:-------------------|:-------------------|:------|:------|:------|:------|:------|
| age        |     303 |          |       |        | 54.43894389438944  | 9.038662442446743  | 29.0  | 48.0  | 56.0  | 61.0  | 77.0  |
| sex        |     303 | 2        | 1     | 206    |                    |                    |       |       |       |       |       |
| cp         |     303 | 4        | 4     | 144    |                    |                    |       |       |       |       |       |
| trestbps   |     303 |          |       |        | 131.68976897689768 | 17.599747729587687 | 94.0  | 120.0 | 130.0 | 140.0 | 200.0 |
| chol       |     303 |          |       |        | 246.69306930693068 | 51.776917542637015 | 126.0 | 211.0 | 241.0 | 275.0 | 564.0 |
| fbs        |     303 | 2        | 0     | 258    |                    |                    |       |       |       |       |       |
| restecg    |     303 | 3        | 0     | 151    |                    |                    |       |       |       |       |       |
| thalach    |     303 |          |       |        | 149.6072607260726  | 22.875003276980376 | 71.0  | 133.5 | 153.0 | 166.0 | 202.0 |
| exang      |     303 | 2        | 0     | 204    |                    |                    |       |       |       |       |       |
| oldpeak    |     303 |          |       |        | 1.0396039603960396 | 1.161075022068634  | 0.0   | 0.0   | 0.8   | 1.6   | 6.2   |
| slope      |     303 | 3        | 1     | 142    |                    |                    |       |       |       |       |       |
| ca         |     299 |          |       |        | 0.6722408026755853 | 0.9374383177242157 | 0.0   | 0.0   | 0.0   | 1.0   | 3.0   |
| thal       |     301 |          |       |        | 4.73421926910299   | 1.9397057693786417 | 3.0   | 3.0   | 3.0   | 7.0   | 7.0   |
| num        |     303 | 5        | 0     | 164    |                    |                    |       |       |       |       |       |

## Learned BN structures and configurations

### Arc blacklist

- Sensitive variables: age, sex
- Rule: forbid parent arcs from sensitive to non-sensitive
- Forbidden arc count: 24

### clg_mi2

| param        | value    |
|:-------------|:---------|
| bn_type      | clg      |
| score        | bic      |
| operators    | ['arcs'] |
| max_indegree | 2        |
| seed         | 42       |

![BN graph](bn_clg_mi2.png)

Serialization

- Structure (GraphML): [structure_clg_mi2.graphml](structure_clg_mi2.graphml)
- Full model (pickle): [model_clg_mi2.pickle](model_clg_mi2.pickle)

### semi_mi5

| param        | value          |
|:-------------|:---------------|
| bn_type      | semiparametric |
| score        | bic            |
| operators    | ['arcs']       |
| max_indegree | 5              |
| seed         | 42             |

![BN graph](bn_semi_mi5.png)

Serialization

- Structure (GraphML): [structure_semi_mi5.graphml](structure_semi_mi5.graphml)
- Full model (pickle): [model_semi_mi5.pickle](model_semi_mi5.pickle)

MetaSyn GMF: [metasyn_gmf.json](metasyn_gmf.json)

## Fidelity (BN vs MetaSyn)

| model       | mean_loglik   | std_loglik   | sum_loglik   |   disc_jsd_mean |   disc_jsd_median |   cont_ks_mean |   cont_w1_mean |
|:------------|:--------------|:-------------|:-------------|----------------:|------------------:|---------------:|---------------:|
| BN:clg_mi2  | -27.2247      | 2.8397       | -1633.4811   |          0.0982 |            0.0998 |         0.2269 |         3.1875 |
| BN:semi_mi5 | -27.1422      | 2.7747       | -1628.5349   |          0.0982 |            0.0998 |         0.2272 |         3.1898 |
| MetaSyn     |               |              |              |          0.1003 |            0.1128 |         0.297  |         2.9962 |

### Per-variable distances (lower is closer)

| variable   | type       | ('clg_mi2', 'JSD')   | ('clg_mi2', 'KS')   | ('clg_mi2', 'W1')   | ('semi_mi5', 'JSD')   | ('semi_mi5', 'KS')   | ('semi_mi5', 'W1')   | ('MetaSyn', 'JSD')   | ('MetaSyn', 'KS')   | ('MetaSyn', 'W1')   |
|:-----------|:-----------|:---------------------|:--------------------|:--------------------|:----------------------|:---------------------|:---------------------|:---------------------|:--------------------|:--------------------|
| age        | continuous |                      | 0.0743              | 1.2354              |                       | 0.0767               | 1.2511               |                      | 0.0997              | 1.2428              |
| ca         | continuous | 0.0998               |                     |                     | 0.0998                |                      |                      | 0.1128               |                     |                     |
| chol       | continuous | 0.1164               |                     |                     | 0.1164                |                      |                      | 0.1134               |                     |                     |
| cp         | discrete   |                      | 0.191               | 4.25                |                       | 0.191                | 4.25                 |                      | 0.174               | 3.3243              |
| exang      | discrete   |                      | 0.1607              | 9.7137              |                       | 0.1607               | 9.7137               |                      | 0.1237              | 8.7702              |
| fbs        | discrete   | 0.1128               |                     |                     | 0.1128                |                      |                      | 0.1128               |                     |                     |
| num        | discrete   | 0.1089               |                     |                     | 0.1089                |                      |                      | 0.0884               |                     |                     |
| oldpeak    | continuous |                      | 0.2157              | 5.6019              |                       | 0.2157               | 5.6019               |                      | 0.2147              | 6.2437              |
| restecg    | discrete   | 0.0986               |                     |                     | 0.0986                |                      |                      | 0.1225               |                     |                     |
| sex        | discrete   |                      | 0.187               | 0.2177              |                       | 0.187                | 0.2177               |                      | 0.2833              | 0.1743              |
| slope      | discrete   | 0.0694               |                     |                     | 0.0694                |                      |                      | 0.088                |                     |                     |
| thal       | continuous |                      | 0.3383              | 0.3443              |                       | 0.3383               | 0.3443               |                      | 0.5833              | 0.2998              |
| thalach    | continuous |                      | 0.421               | 0.9498              |                       | 0.421                | 0.9498               |                      | 0.6                 | 0.9185              |
| trestbps   | continuous | 0.0812               |                     |                     | 0.0812                |                      |                      | 0.0644               |                     |                     |

## UMAP overview (same projection)

| Real (sample) | MetaSyn (synthetic) | BN: clg_mi2 | BN: semi_mi5 |
| --- | --- | --- | --- |
| <img src='umap_real.png' width='280'/> | <img src='umap_metasyn.png' width='280'/> | <img src='umap_bn_clg_mi2.png' width='280'/> | <img src='umap_bn_semi_mi5.png' width='280'/> |

