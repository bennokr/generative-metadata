# Data Report — Heart Disease

**Source**: [UCI dataset 45](https://archive.ics.uci.edu/dataset/45)

- Metadata file: [metadata.json](metadata.json)
- JSON-LD (schema.org/Dataset): [dataset.json](dataset.json)
- Rows: 297
- Columns: 14
- Discrete: 7  |  Continuous: 7

## Dataset metadata


### Description

4 databases: Cleveland, Hungary, Switzerland, and the VA Long Beach

- Creators: Andras Janosi, William Steinbrunn, Matthias Pfisterer, Robert Detrano
- Citation: International application of a new probability algorithm for the diagnosis of coronary artery disease.. R. Detrano, A. Jánosi, W. Steinbrunn, M. Pfisterer, J. Schmid, S. Sandhu, K. Guppy, S. Lee, V. Froelicher. American Journal of Cardiology. 1989
- Links:
  - URL: https://archive.ics.uci.edu/dataset/45
## Variables and summary

| variable   | description                                           | inferred   | declared    |   count | unique   | top   | freq   | mean               | std                | min   | 25%   | 50%   | 75%   | max   |
|:-----------|:------------------------------------------------------|:-----------|:------------|--------:|:---------|:------|:-------|:-------------------|:-------------------|:------|:------|:------|:------|:------|
| age        |                                                       | continuous | Integer     |     303 |          |       |        | 54.43894389438944  | 9.038662442446743  | 29.0  | 48.0  | 56.0  | 61.0  | 77.0  |
| sex        |                                                       | discrete   | Categorical |     303 | 2        | 1     | 206    |                    |                    |       |       |       |       |       |
| cp         |                                                       | discrete   | Categorical |     303 | 4        | 4     | 144    |                    |                    |       |       |       |       |       |
| trestbps   | resting blood pressure (on admission to the hospital) | continuous | Integer     |     303 |          |       |        | 131.68976897689768 | 17.599747729587687 | 94.0  | 120.0 | 130.0 | 140.0 | 200.0 |
| chol       | serum cholestoral                                     | continuous | Integer     |     303 |          |       |        | 246.69306930693068 | 51.776917542637015 | 126.0 | 211.0 | 241.0 | 275.0 | 564.0 |
| fbs        | fasting blood sugar > 120 mg/dl                       | discrete   | Categorical |     303 | 2        | 0     | 258    |                    |                    |       |       |       |       |       |
| restecg    |                                                       | discrete   | Categorical |     303 | 3        | 0     | 151    |                    |                    |       |       |       |       |       |
| thalach    | maximum heart rate achieved                           | continuous | Integer     |     303 |          |       |        | 149.6072607260726  | 22.875003276980376 | 71.0  | 133.5 | 153.0 | 166.0 | 202.0 |
| exang      | exercise induced angina                               | discrete   | Categorical |     303 | 2        | 0     | 204    |                    |                    |       |       |       |       |       |
| oldpeak    | ST depression induced by exercise relative to rest    | continuous | Integer     |     303 |          |       |        | 1.0396039603960396 | 1.161075022068634  | 0.0   | 0.0   | 0.8   | 1.6   | 6.2   |
| slope      |                                                       | discrete   | Categorical |     303 | 3        | 1     | 142    |                    |                    |       |       |       |       |       |
| ca         | number of major vessels (0-3) colored by flourosopy   | continuous | Integer     |     299 |          |       |        | 0.6722408026755853 | 0.9374383177242157 | 0.0   | 0.0   | 0.0   | 1.0   | 3.0   |
| thal       |                                                       | continuous | Categorical |     301 |          |       |        | 4.73421926910299   | 1.9397057693786417 | 3.0   | 3.0   | 3.0   | 7.0   | 7.0   |
| num        | diagnosis of heart disease                            | discrete   | Integer     |     303 | 5        | 0     | 164    |                    |                    |       |       |       |       |       |

## Learned structures and configurations

MetaSyn GMF: [metasyn_gmf.json](metasyn_gmf.json)

### Arc blacklist

- Root variables: age, sex
- Forbidden arc count: 28

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
| BN:clg_mi2  | -27.3163      | 2.9854       | -1638.9769   |          0.0979 |            0.1075 |         0.2266 |         3.6304 |
| BN:semi_mi5 | -27.3163      | 2.9854       | -1638.9769   |          0.0979 |            0.1075 |         0.2266 |         3.6304 |
| MetaSyn     |               |              |              |          0.0952 |            0.0868 |         0.2907 |         2.8371 |

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
      <td>0.0803</td>
      <td>0.0803</td>
      <td>0.0903</td>
      <td>1.2421</td>
      <td>1.2421</td>
      <td>1.3481</td>
    </tr>
    <tr>
      <td>ca</td>
      <td>continuous</td>
      <td>0.0952</td>
      <td>0.0952</td>
      <td>0.1128</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>chol</td>
      <td>continuous</td>
      <td>0.1094</td>
      <td>0.1094</td>
      <td>0.1150</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>cp</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.1910</td>
      <td>0.1910</td>
      <td>0.1317</td>
      <td>3.5802</td>
      <td>3.5802</td>
      <td>2.8285</td>
    </tr>
    <tr>
      <td>exang</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.1727</td>
      <td>0.1727</td>
      <td>0.1257</td>
      <td>12.6292</td>
      <td>12.6292</td>
      <td>8.8700</td>
    </tr>
    <tr>
      <td>fbs</td>
      <td>discrete</td>
      <td>0.1128</td>
      <td>0.1128</td>
      <td>0.0792</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>num</td>
      <td>discrete</td>
      <td>0.1089</td>
      <td>0.1089</td>
      <td>0.0768</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>oldpeak</td>
      <td>continuous</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.2067</td>
      <td>0.2067</td>
      <td>0.2207</td>
      <td>6.3633</td>
      <td>6.3633</td>
      <td>5.3542</td>
    </tr>
    <tr>
      <td>restecg</td>
      <td>discrete</td>
      <td>0.1075</td>
      <td>0.1075</td>
      <td>0.1137</td>
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
      <td></td>
      <td></td>
      <td></td>
      <td>0.1750</td>
      <td>0.1750</td>
      <td>0.2833</td>
      <td>0.2662</td>
      <td>0.2662</td>
      <td>0.2267</td>
    </tr>
    <tr>
      <td>slope</td>
      <td>discrete</td>
      <td>0.0561</td>
      <td>0.0561</td>
      <td>0.0820</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>thal</td>
      <td>continuous</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.3523</td>
      <td>0.3523</td>
      <td>0.5833</td>
      <td>0.3478</td>
      <td>0.3478</td>
      <td>0.3075</td>
    </tr>
    <tr>
      <td>thalach</td>
      <td>continuous</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.4080</td>
      <td>0.4080</td>
      <td>0.6000</td>
      <td>0.9840</td>
      <td>0.9840</td>
      <td>0.9245</td>
    </tr>
    <tr>
      <td>trestbps</td>
      <td>continuous</td>
      <td>0.0952</td>
      <td>0.0952</td>
      <td>0.0868</td>
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

