# Data Report — adult

**Source**: [OpenML dataset 45068](https://www.openml.org/search?type=data&id=45068)

- Metadata file: [metadata.json](metadata.json)
- JSON-LD (schema.org/Dataset): [dataset.json](dataset.json)
- Rows: 48842
- Columns: 15
- Discrete: 10  |  Continuous: 5

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

## Learned structures and configurations

MetaSyn GMF: [metasyn_gmf.json](metasyn_gmf.json)

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

## Fidelity (BN vs MetaSyn)

| model       | mean_loglik   | std_loglik   | sum_loglik   |   disc_jsd_mean |   disc_jsd_median |   cont_ks_mean |   cont_w1_mean |
|:------------|:--------------|:-------------|:-------------|----------------:|------------------:|---------------:|---------------:|
| BN:clg_mi2  | -46.2333      | 17.6838      | -450867.3177 |          0.0363 |            0.0318 |         0.301  |        3727.1  |
| BN:semi_mi5 | -46.2188      | 17.6841      | -450725.5899 |          0.0363 |            0.0318 |         0.2984 |        4011.98 |
| MetaSyn     |               |              |              |          0.0428 |            0.0364 |         0.4443 |        5180.55 |

### Per-variable distances (lower is closer)

<table class="dataframe table per-var-dist">
  <thead>
    <tr>
      <th colspan="2" halign="left"></th>
      <th colspan="3" halign="left">JSD</th>
      <th colspan="3" halign="left">KS</th>
      <th colspan="3" halign="left">W1</th>
    </tr>
    <tr>
      <th>variable</th>
      <th>type</th>
      <th>clg_mi2</th>
      <th>semi_mi5</th>
      <th>MetaSyn</th>
      <th>clg_mi2</th>
      <th>semi_mi5</th>
      <th>MetaSyn</th>
      <th>clg_mi2</th>
      <th>semi_mi5</th>
      <th>MetaSyn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>age</td>
      <td>continuous</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.0647</td>
      <td>0.0647</td>
      <td>0.0456</td>
      <td>1.9326</td>
      <td>1.9326</td>
      <td>0.6628</td>
    </tr>
    <tr>
      <td>capital-gain</td>
      <td>continuous</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.0872</td>
      <td>0.0982</td>
      <td>0.0678</td>
      <td>16008.7500</td>
      <td>17469.3816</td>
      <td>24073.0913</td>
    </tr>
    <tr>
      <td>capital-loss</td>
      <td>continuous</td>
      <td>0.0434</td>
      <td>0.0434</td>
      <td>0.0588</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>class</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.5376</td>
      <td>0.5216</td>
      <td>0.9106</td>
      <td>2320.0599</td>
      <td>2283.6989</td>
      <td>1679.7544</td>
    </tr>
    <tr>
      <td>education</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.5660</td>
      <td>0.5660</td>
      <td>0.9520</td>
      <td>301.3958</td>
      <td>301.3958</td>
      <td>145.8651</td>
    </tr>
    <tr>
      <td>education-num</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.2496</td>
      <td>0.2416</td>
      <td>0.2456</td>
      <td>3.3540</td>
      <td>3.4741</td>
      <td>3.3620</td>
    </tr>
    <tr>
      <td>fnlwgt</td>
      <td>continuous</td>
      <td>0.0385</td>
      <td>0.0385</td>
      <td>0.0376</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>hours-per-week</td>
      <td>continuous</td>
      <td>0.0434</td>
      <td>0.0434</td>
      <td>0.0352</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>marital-status</td>
      <td>discrete</td>
      <td>0.0114</td>
      <td>0.0114</td>
      <td>0.0328</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>native-country</td>
      <td>discrete</td>
      <td>0.0516</td>
      <td>0.0516</td>
      <td>0.0600</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>occupation</td>
      <td>discrete</td>
      <td>0.0209</td>
      <td>0.0209</td>
      <td>0.0484</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>race</td>
      <td>discrete</td>
      <td>0.0154</td>
      <td>0.0154</td>
      <td>0.0259</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relationship</td>
      <td>discrete</td>
      <td>0.0251</td>
      <td>0.0251</td>
      <td>0.0033</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sex</td>
      <td>discrete</td>
      <td>0.0929</td>
      <td>0.0929</td>
      <td>0.1033</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>workclass</td>
      <td>discrete</td>
      <td>0.0207</td>
      <td>0.0207</td>
      <td>0.0228</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

## UMAP overview (same projection)

| Real (sample) | MetaSyn (synthetic) | BN: clg_mi2 | BN: semi_mi5 |
| --- | --- | --- | --- |
| <img src='umap_real.png' width='280'/> | <img src='umap_metasyn.png' width='280'/> | <img src='umap_bn_clg_mi2.png' width='280'/> | <img src='umap_bn_semi_mi5.png' width='280'/> |

