# Data Report — bank-marketing

**Source**: [OpenML dataset 46910](https://www.openml.org/search?type=data&id=46910)

- Metadata file: [metadata.json](metadata.json)
- JSON-LD (schema.org/Dataset): [dataset.json](dataset.json)
- Rows: 45211
- Columns: 14
- Discrete: 9  |  Continuous: 5

## Dataset metadata


### Description

This dataset was curated for [TabArena](https://tabarena.ai/) by the TabArena team
as part of the [TabArena Tabular ML IID Study](https://tabarena.ai/data-tabular-ml-iid-study).
For more details on the study, see our [paper](https://tabarena.ai/paper-tabular-ml-iid-study).

**Dataset Focus**: This dataset shall be used for evaluating predictive machine
learning models for independent and identically distributed tabular data. The
intended task is classification.

---
#### Dataset Metadata
- **Licence:** CC BY 4.0
- **Original Data Source:** https://doi.org/10.24432/C5K306
- **Reference (please cite)**: Moro, Sergio, Paulo Cortez, and Paulo Rita. 'A data-driven approach to predict the success of bank telemarketing.' Decision Support Systems 62 (2014): 22-31. https://doi.org/10.1016/j.dss.2014.03.001
- **Dataset Year:** 2012
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We removed the "duration" feature following its original description to obtain a "realistic predictive model".
- We further remove the "month" and "day_of_week" features, as they also relate to the last contact -- which is not available in a real-world scenario.

- Creators: See original data source.
- Date: 2025-04-30T19:54:48
- Citation: Moro, Sergio, Paulo Cortez, and Paulo Rita. 'A data-driven approach to predict the success of bank telemarketing.' Decision Support Systems 62 (2014): 22-31. https://doi.org/10.1016/j.dss.2014.03.001
- Links:
  - URL: https://api.openml.org/data/v1/download/22125221/bank-marketing.arff
  - sameAs: https://doi.org/10.1016/j.dss.2014.03.001
## Variables and summary

| variable             | inferred   |   count | unique   | top         | freq   | mean               | std                | min     | 25%   | 50%   | 75%    | max      |
|:---------------------|:-----------|--------:|:---------|:------------|:-------|:-------------------|:-------------------|:--------|:------|:------|:-------|:---------|
| age                  | continuous |   45211 |          |             |        | 40.93621021432837  | 10.618762040975485 | 18.0    | 33.0  | 39.0  | 48.0   | 95.0     |
| job                  | discrete   |   45211 | 12       | blue-collar | 9732   |                    |                    |         |       |       |        |          |
| marital              | discrete   |   45211 | 3        | married     | 27214  |                    |                    |         |       |       |        |          |
| education            | discrete   |   45211 | 4        | secondary   | 23202  |                    |                    |         |       |       |        |          |
| default              | discrete   |   45211 | 2        | no          | 44396  |                    |                    |         |       |       |        |          |
| balance              | continuous |   45211 |          |             |        | 1362.2720576850766 | 3044.765829168626  | -8019.0 | 72.0  | 448.0 | 1428.0 | 102127.0 |
| housing              | discrete   |   45211 | 2        | yes         | 25130  |                    |                    |         |       |       |        |          |
| loan                 | discrete   |   45211 | 2        | no          | 37967  |                    |                    |         |       |       |        |          |
| contact              | discrete   |   45211 | 3        | cellular    | 29285  |                    |                    |         |       |       |        |          |
| campaign             | continuous |   45211 |          |             |        | 2.763840658246887  | 3.0980208832796765 | 1.0     | 1.0   | 2.0   | 3.0    | 63.0     |
| pdays                | continuous |   45211 |          |             |        | 40.19782796222158  | 100.12874599062957 | -1.0    | -1.0  | -1.0  | -1.0   | 871.0    |
| previous             | continuous |   45211 |          |             |        | 0.5803233726305546 | 2.30344104493196   | 0.0     | 0.0   | 0.0   | 0.0    | 275.0    |
| poutcome             | discrete   |   45211 | 4        | unknown     | 36959  |                    |                    |         |       |       |        |          |
| SubscribeTermDeposit | discrete   |   45211 | 2        | no          | 39922  |                    |                    |         |       |       |        |          |

## Learned structures and configurations

MetaSyn GMF: [metasyn_gmf.json](metasyn_gmf.json)

### Arc blacklist

- Root variables: age
- Forbidden arc count: 14

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
| BN:clg_mi2  | -25.1747      | 198.1935     | -227654.9385 |          0.0153 |            0.0103 |         0.2951 |        284.918 |
| BN:semi_mi5 | -70.1978      | 4601.4407    | -634728.935  |          0.0204 |            0.0162 |         0.3006 |        289.255 |
| MetaSyn     |               |              |              |          0.0206 |            0.0114 |         0.4702 |        331.213 |

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
      <td>SubscribeTermDeposit</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.0967</td>
      <td>0.1257</td>
      <td>0.0750</td>
      <td>1.7204</td>
      <td>1.7485</td>
      <td>0.9360</td>
    </tr>
    <tr>
      <td>age</td>
      <td>continuous</td>
      <td>0.0446</td>
      <td>0.0446</td>
      <td>0.0525</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>balance</td>
      <td>continuous</td>
      <td>0.0261</td>
      <td>0.0261</td>
      <td>0.0114</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>campaign</td>
      <td>continuous</td>
      <td>0.0162</td>
      <td>0.0162</td>
      <td>0.0306</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>contact</td>
      <td>discrete</td>
      <td>0.0070</td>
      <td>0.0058</td>
      <td>0.0027</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>default</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.2571</td>
      <td>0.2479</td>
      <td>0.2556</td>
      <td>1413.3576</td>
      <td>1435.0688</td>
      <td>1612.1766</td>
    </tr>
    <tr>
      <td>education</td>
      <td>discrete</td>
      <td>0.0103</td>
      <td>0.0009</td>
      <td>0.0086</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>housing</td>
      <td>discrete</td>
      <td>0.0094</td>
      <td>0.0094</td>
      <td>0.0090</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>job</td>
      <td>discrete</td>
      <td>0.0103</td>
      <td>0.0139</td>
      <td>0.0263</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>loan</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.2770</td>
      <td>0.2630</td>
      <td>0.3850</td>
      <td>1.4294</td>
      <td>1.4182</td>
      <td>0.6508</td>
    </tr>
    <tr>
      <td>marital</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.4296</td>
      <td>0.4476</td>
      <td>0.8176</td>
      <td>7.7782</td>
      <td>7.7073</td>
      <td>41.6624</td>
    </tr>
    <tr>
      <td>pdays</td>
      <td>continuous</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.4150</td>
      <td>0.4186</td>
      <td>0.8176</td>
      <td>0.3044</td>
      <td>0.3323</td>
      <td>0.6406</td>
    </tr>
    <tr>
      <td>poutcome</td>
      <td>discrete</td>
      <td>0.0126</td>
      <td>0.0357</td>
      <td>0.0377</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>previous</td>
      <td>continuous</td>
      <td>0.0013</td>
      <td>0.0306</td>
      <td>0.0066</td>
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

