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

| variable       | inferred   |   count | unique   | top                | freq   | mean               | std                | min     | 25%      | 50%      | 75%      | max       |
|:---------------|:-----------|--------:|:---------|:-------------------|:-------|:-------------------|:-------------------|:--------|:---------|:---------|:---------|:----------|
| age            | continuous |   48842 |          |                    |        | 38.64358543876172  | 13.71050993444316  | 17.0    | 28.0     | 37.0     | 48.0     | 90.0      |
| fnlwgt         | continuous |   48842 |          |                    |        | 189664.13459727284 | 105604.02542315786 | 12285.0 | 117550.5 | 178144.5 | 237642.0 | 1490400.0 |
| education-num  | discrete   |   48842 | 16       | 9                  | 15784  |                    |                    |         |          |          |          |           |
| capital-gain   | continuous |   48842 |          |                    |        | 1079.0676262233324 | 7452.01905765375   | 0.0     | 0.0      | 0.0      | 0.0      | 99999.0   |
| capital-loss   | continuous |   48842 |          |                    |        | 87.50231358257237  | 403.00455212445047 | 0.0     | 0.0      | 0.0      | 0.0      | 4356.0    |
| hours-per-week | continuous |   48842 |          |                    |        | 40.422382375824085 | 12.391444024255906 | 1.0     | 40.0     | 40.0     | 45.0     | 99.0      |
| workclass      | discrete   |   48842 | 9        | Private            | 33906  |                    |                    |         |          |          |          |           |
| education      | discrete   |   48842 | 16       | HS-grad            | 15784  |                    |                    |         |          |          |          |           |
| marital-status | discrete   |   48842 | 7        | Married-civ-spouse | 22379  |                    |                    |         |          |          |          |           |
| occupation     | discrete   |   48842 | 15       | Prof-specialty     | 6172   |                    |                    |         |          |          |          |           |
| relationship   | discrete   |   48842 | 6        | Husband            | 19716  |                    |                    |         |          |          |          |           |
| race           | discrete   |   48842 | 5        | White              | 41762  |                    |                    |         |          |          |          |           |
| sex            | discrete   |   48842 | 2        | Male               | 32650  |                    |                    |         |          |          |          |           |
| native-country | discrete   |   48842 | 42       | United-States      | 43832  |                    |                    |         |          |          |          |           |
| class          | discrete   |   48842 | 2        | <=50K              | 37155  |                    |                    |         |          |          |          |           |

## Learned BN structures and configurations

### Arc blacklist

- Root variables: age, sex, race
- Forbidden arc count: 45

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
| BN:clg_mi2  | -46.2333      | 17.6838      | -450867.3177 |          0.0349 |            0.0311 |         0.298  |        3711.92 |
| BN:semi_mi5 | -46.2188      | 17.6841      | -450725.5899 |          0.0363 |            0.0318 |         0.2984 |        4011.98 |
| MetaSyn     |               |              |              |          0.0399 |            0.036  |         0.4581 |        5126.04 |

### Per-variable distances (lower is closer)

| variable       | type       | ('JSD', 'clg_mi2')   | ('JSD', 'semi_mi5')   | ('JSD', 'MetaSyn')   | ('KS', 'clg_mi2')   | ('KS', 'semi_mi5')   | ('KS', 'MetaSyn')   | ('W1', 'clg_mi2')   | ('W1', 'semi_mi5')   | ('W1', 'MetaSyn')   |
|:---------------|:-----------|:---------------------|:----------------------|:---------------------|:--------------------|:---------------------|:--------------------|:--------------------|:---------------------|:--------------------|
| age            | continuous |                      |                       |                      | 0.0647              | 0.0647               | 0.0649              | 1.9326              | 1.9326               | 1.3021              |
| capital-gain   | continuous |                      |                       |                      | 0.0872              | 0.0982               | 0.1075              | 16008.75            | 17469.3816           | 23843.0704          |
| capital-loss   | continuous | 0.0311               | 0.0434                | 0.0556               |                     |                      |                     |                     |                      |                     |
| class          | discrete   |                      |                       |                      | 0.5276              | 0.5216               | 0.9106              | 2244.6671           | 2283.6989            | 1636.6148           |
| education      | discrete   |                      |                       |                      | 0.563               | 0.566                | 0.952               | 300.9421            | 301.3958             | 145.5235            |
| education-num  | discrete   |                      |                       |                      | 0.2476              | 0.2416               | 0.2556              | 3.2947              | 3.4741               | 3.6759              |
| fnlwgt         | continuous | 0.0385               | 0.0385                | 0.0251               |                     |                      |                     |                     |                      |                     |
| hours-per-week | continuous | 0.0311               | 0.0434                | 0.0399               |                     |                      |                     |                     |                      |                     |
| marital-status | discrete   | 0.0114               | 0.0114                | 0.0322               |                     |                      |                     |                     |                      |                     |
| native-country | discrete   | 0.0516               | 0.0516                | 0.0441               |                     |                      |                     |                     |                      |                     |
| occupation     | discrete   | 0.0209               | 0.0209                | 0.0398               |                     |                      |                     |                     |                      |                     |
| race           | discrete   | 0.0154               | 0.0154                | 0.0237               |                     |                      |                     |                     |                      |                     |
| relationship   | discrete   | 0.0251               | 0.0251                | 0.0191               |                     |                      |                     |                     |                      |                     |
| sex            | discrete   | 0.0929               | 0.0929                | 0.0998               |                     |                      |                     |                     |                      |                     |
| workclass      | discrete   | 0.031                | 0.0207                | 0.0197               |                     |                      |                     |                     |                      |                     |

## UMAP overview (same projection)

| Real (sample) | MetaSyn (synthetic) | BN: clg_mi2 | BN: semi_mi5 |
| --- | --- | --- | --- |
| <img src='umap_real.png' width='280'/> | <img src='umap_metasyn.png' width='280'/> | <img src='umap_bn_clg_mi2.png' width='280'/> | <img src='umap_bn_semi_mi5.png' width='280'/> |

