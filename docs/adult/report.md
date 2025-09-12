# Data Report — adult

- Metadata file: [metadata.json](metadata.json)
- JSON-LD (schema.org/Dataset): [dataset.jsonld](dataset.jsonld)
- Rows: 48842
- Columns: 15
- Discrete: 10  |  Continuous: 5
- OpenML page: https://www.openml.org/search?type=data&id=45068

## Dataset metadata


### Description

Prediction task is to determine whether a person makes over 50K a year. Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

- Creators: Yoontae Hwang, Youngbin Lee, Yongjae Lee
- Date: 2023-01-27T11:26:07
- Citation: Yoontae Hwang, Youngbin Lee, Yongjae Lee,Semi-Supervised Learning for Tabular Datasets with Continuous and Categorical Variables, Archive, 2023
- Links:
  - URL: https://api.openml.org/data/v1/download/22112026/adult.arff
## Variables and summary

| variable       | measurement   | unit   |   count | unique   | top                | freq   | mean               | std                | min     | 25%      | 50%      | 75%      | max       |
|:---------------|:--------------|:-------|--------:|:---------|:-------------------|:-------|:-------------------|:-------------------|:--------|:---------|:---------|:---------|:----------|
| age            | continuous    |        |   48842 |          |                    |        | 38.64358543876172  | 13.71050993444316  | 17.0    | 28.0     | 37.0     | 48.0     | 90.0      |
| fnlwgt         | continuous    |        |   48842 |          |                    |        | 189664.13459727284 | 105604.02542315786 | 12285.0 | 117550.5 | 178144.5 | 237642.0 | 1490400.0 |
| education-num  | discrete      |        |   48842 | 16       | 9                  | 15784  |                    |                    |         |          |          |          |           |
| capital-gain   | continuous    |        |   48842 |          |                    |        | 1079.0676262233324 | 7452.01905765375   | 0.0     | 0.0      | 0.0      | 0.0      | 99999.0   |
| capital-loss   | continuous    |        |   48842 |          |                    |        | 87.50231358257237  | 403.00455212445047 | 0.0     | 0.0      | 0.0      | 0.0      | 4356.0    |
| hours-per-week | continuous    |        |   48842 |          |                    |        | 40.422382375824085 | 12.391444024255906 | 1.0     | 40.0     | 40.0     | 45.0     | 99.0      |
| workclass      | discrete      |        |   48842 | 9        | Private            | 33906  |                    |                    |         |          |          |          |           |
| education      | discrete      |        |   48842 | 16       | HS-grad            | 15784  |                    |                    |         |          |          |          |           |
| marital-status | discrete      |        |   48842 | 7        | Married-civ-spouse | 22379  |                    |                    |         |          |          |          |           |
| occupation     | discrete      |        |   48842 | 15       | Prof-specialty     | 6172   |                    |                    |         |          |          |          |           |
| relationship   | discrete      |        |   48842 | 6        | Husband            | 19716  |                    |                    |         |          |          |          |           |
| race           | discrete      |        |   48842 | 5        | White              | 41762  |                    |                    |         |          |          |          |           |
| sex            | discrete      |        |   48842 | 2        | Male               | 32650  |                    |                    |         |          |          |          |           |
| native-country | discrete      |        |   48842 | 42       | United-States      | 43832  |                    |                    |         |          |          |          |           |
| class          | discrete      |        |   48842 | 2        | <=50K              | 37155  |                    |                    |         |          |          |          |           |

## Learned BN structures and configurations

### Arc blacklist

- Sensitive variables: age, sex, race
- Rule: forbid incoming arcs into sensitive from non-sensitive
- Forbidden arc count: 36

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
| BN:clg_mi2  | -46.2197      | 17.6846      | -450734.5953 |          0.0393 |            0.0311 |         0.3024 |        3787.54 |
| BN:semi_mi5 | -46.2052      | 17.685       | -450592.8675 |          0.0393 |            0.0311 |         0.3057 |        4035.03 |
| MetaSyn     |               |              |              |          0.0401 |            0.0409 |         0.4581 |        5056.04 |

### Per-variable distances (lower is closer)

| variable       | type       | ('JSD', 'clg_mi2')   | ('JSD', 'semi_mi5')   | ('JSD', 'MetaSyn')   | ('KS', 'clg_mi2')   | ('KS', 'semi_mi5')   | ('KS', 'MetaSyn')   | ('W1', 'clg_mi2')   | ('W1', 'semi_mi5')   | ('W1', 'MetaSyn')   |
|:---------------|:-----------|:---------------------|:----------------------|:---------------------|:--------------------|:---------------------|:--------------------|:--------------------|:---------------------|:--------------------|
| age            | continuous |                      |                       |                      | 0.0833              | 0.0833               | 0.0489              | 2.0701              | 2.0701               | 0.5968              |
| capital-gain   | continuous |                      |                       |                      | 0.0943              | 0.1048               | 0.1353              | 16363.4901          | 17554.3394           | 23496.0059          |
| capital-loss   | continuous | 0.0434               | 0.0434                | 0.0431               |                     |                      |                     |                     |                      |                     |
| class          | discrete   |                      |                       |                      | 0.5206              | 0.5366               | 0.9106              | 2268.4204           | 2314.9021            | 1636.1787           |
| education      | discrete   |                      |                       |                      | 0.563               | 0.563                | 0.952               | 300.3598            | 300.3598             | 143.8699            |
| education-num  | discrete   |                      |                       |                      | 0.2506              | 0.2406               | 0.2436              | 3.3566              | 3.4827               | 3.5677              |
| fnlwgt         | continuous | 0.037                | 0.037                 | 0.0471               |                     |                      |                     |                     |                      |                     |
| hours-per-week | continuous | 0.0434               | 0.0434                | 0.0486               |                     |                      |                     |                     |                      |                     |
| marital-status | discrete   | 0.0134               | 0.0134                | 0.0309               |                     |                      |                     |                     |                      |                     |
| native-country | discrete   | 0.0531               | 0.0531                | 0.0463               |                     |                      |                     |                     |                      |                     |
| occupation     | discrete   | 0.0184               | 0.0184                | 0.0387               |                     |                      |                     |                     |                      |                     |
| race           | discrete   | 0.0246               | 0.0246                | 0.0188               |                     |                      |                     |                     |                      |                     |
| relationship   | discrete   | 0.0251               | 0.0251                | 0.0156               |                     |                      |                     |                     |                      |                     |
| sex            | discrete   | 0.1104               | 0.1104                | 0.0978               |                     |                      |                     |                     |                      |                     |
| workclass      | discrete   | 0.0238               | 0.0238                | 0.0136               |                     |                      |                     |                     |                      |                     |

## UMAP overview (same projection)

| Real (sample) | MetaSyn (synthetic) | BN: clg_mi2 | BN: semi_mi5 |
| --- | --- | --- | --- |
| <img src='umap_real.png' width='280'/> | <img src='umap_metasyn.png' width='280'/> | <img src='umap_bn_clg_mi2.png' width='280'/> | <img src='umap_bn_semi_mi5.png' width='280'/> |

