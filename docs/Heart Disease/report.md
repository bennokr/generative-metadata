# Data Report — Heart Disease

- Metadata file: [metadata.json](metadata.json)
- JSON-LD (schema.org/Dataset): [dataset.jsonld](dataset.jsonld)
- Rows: 297
- Columns: 14
- Discrete: 7  |  Continuous: 7
- UCI ML page: https://archive.ics.uci.edu/dataset/45

## Dataset metadata


### Description

4 databases: Cleveland, Hungary, Switzerland, and the VA Long Beach

- Creators: Andras Janosi, William Steinbrunn, Matthias Pfisterer, Robert Detrano
- Citation: International application of a new probability algorithm for the diagnosis of coronary artery disease.. R. Detrano, A. Jánosi, W. Steinbrunn, M. Pfisterer, J. Schmid, S. Sandhu, K. Guppy, S. Lee, V. Froelicher. American Journal of Cardiology. 1989
## Variables and summary

| variable   | measurement   | unit   |   count | unique   | top   | freq   | mean               | std                | min   | 25%   | 50%   | 75%   | max   |
|:-----------|:--------------|:-------|--------:|:---------|:------|:-------|:-------------------|:-------------------|:------|:------|:------|:------|:------|
| age        | continuous    |        |     303 |          |       |        | 54.43894389438944  | 9.038662442446743  | 29.0  | 48.0  | 56.0  | 61.0  | 77.0  |
| sex        | discrete      |        |     303 | 2        | 1     | 206    |                    |                    |       |       |       |       |       |
| cp         | discrete      |        |     303 | 4        | 4     | 144    |                    |                    |       |       |       |       |       |
| trestbps   | continuous    |        |     303 |          |       |        | 131.68976897689768 | 17.599747729587687 | 94.0  | 120.0 | 130.0 | 140.0 | 200.0 |
| chol       | continuous    |        |     303 |          |       |        | 246.69306930693068 | 51.776917542637015 | 126.0 | 211.0 | 241.0 | 275.0 | 564.0 |
| fbs        | discrete      |        |     303 | 2        | 0     | 258    |                    |                    |       |       |       |       |       |
| restecg    | discrete      |        |     303 | 3        | 0     | 151    |                    |                    |       |       |       |       |       |
| thalach    | continuous    |        |     303 |          |       |        | 149.6072607260726  | 22.875003276980376 | 71.0  | 133.5 | 153.0 | 166.0 | 202.0 |
| exang      | discrete      |        |     303 | 2        | 0     | 204    |                    |                    |       |       |       |       |       |
| oldpeak    | continuous    |        |     303 |          |       |        | 1.0396039603960396 | 1.161075022068634  | 0.0   | 0.0   | 0.8   | 1.6   | 6.2   |
| slope      | discrete      |        |     303 | 3        | 1     | 142    |                    |                    |       |       |       |       |       |
| ca         | continuous    |        |     299 |          |       |        | 0.6722408026755853 | 0.9374383177242157 | 0.0   | 0.0   | 0.0   | 1.0   | 3.0   |
| thal       | continuous    |        |     301 |          |       |        | 4.73421926910299   | 1.9397057693786417 | 3.0   | 3.0   | 3.0   | 7.0   | 7.0   |
| num        | discrete      |        |     303 | 5        | 0     | 164    |                    |                    |       |       |       |       |       |

## Learned BN structures and configurations

### Arc blacklist

- Sensitive variables: age, sex
- Rule: forbid incoming arcs into sensitive from non-sensitive
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
| BN:clg_mi2  | -27.3163      | 2.9854       | -1638.9769   |          0.0979 |            0.1075 |         0.2266 |         3.6304 |
| BN:semi_mi5 | -27.3163      | 2.9854       | -1638.9769   |          0.0979 |            0.1075 |         0.2266 |         3.6304 |
| MetaSyn     |               |              |              |          0.0962 |            0.0995 |         0.2894 |         2.7986 |

### Per-variable distances (lower is closer)

| variable   | type       | ('JSD', 'clg_mi2')   | ('JSD', 'semi_mi5')   | ('JSD', 'MetaSyn')   | ('KS', 'clg_mi2')   | ('KS', 'semi_mi5')   | ('KS', 'MetaSyn')   | ('W1', 'clg_mi2')   | ('W1', 'semi_mi5')   | ('W1', 'MetaSyn')   |
|:-----------|:-----------|:---------------------|:----------------------|:---------------------|:--------------------|:---------------------|:--------------------|:--------------------|:---------------------|:--------------------|
| age        | continuous |                      |                       |                      | 0.0803              | 0.0803               | 0.0923              | 1.2421              | 1.2421               | 1.3608              |
| ca         | continuous | 0.0952               | 0.0952                | 0.0789               |                     |                      |                     |                     |                      |                     |
| chol       | continuous | 0.1094               | 0.1094                | 0.1202               |                     |                      |                     |                     |                      |                     |
| cp         | discrete   |                      |                       |                      | 0.191               | 0.191                | 0.1637              | 3.5802              | 3.5802               | 3.5881              |
| exang      | discrete   |                      |                       |                      | 0.1727              | 0.1727               | 0.1043              | 12.6292             | 12.6292              | 7.5447              |
| fbs        | discrete   | 0.1128               | 0.1128                | 0.1094               |                     |                      |                     |                     |                      |                     |
| num        | discrete   | 0.1089               | 0.1089                | 0.1039               |                     |                      |                     |                     |                      |                     |
| oldpeak    | continuous |                      |                       |                      | 0.2067              | 0.2067               | 0.1987              | 6.3633              | 6.3633               | 5.6675              |
| restecg    | discrete   | 0.1075               | 0.1075                | 0.0995               |                     |                      |                     |                     |                      |                     |
| sex        | discrete   |                      |                       |                      | 0.175               | 0.175                | 0.2833              | 0.2662              | 0.2662               | 0.216               |
| slope      | discrete   | 0.0561               | 0.0561                | 0.0694               |                     |                      |                     |                     |                      |                     |
| thal       | continuous |                      |                       |                      | 0.3523              | 0.3523               | 0.5833              | 0.3478              | 0.3478               | 0.2856              |
| thalach    | continuous |                      |                       |                      | 0.408               | 0.408                | 0.6                 | 0.984               | 0.984                | 0.9275              |
| trestbps   | continuous | 0.0952               | 0.0952                | 0.0923               |                     |                      |                     |                     |                      |                     |

## UMAP overview (same projection)

| Real (sample) | MetaSyn (synthetic) | BN: clg_mi2 | BN: semi_mi5 |
| --- | --- | --- | --- |
| <img src='umap_real.png' width='280'/> | <img src='umap_metasyn.png' width='280'/> | <img src='umap_bn_clg_mi2.png' width='280'/> | <img src='umap_bn_semi_mi5.png' width='280'/> |

