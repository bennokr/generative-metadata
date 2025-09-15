# Data Report — credit-g

**Source**: [OpenML dataset 46918](https://www.openml.org/search?type=data&id=46918)

- Metadata file: [metadata.json](metadata.json)
- JSON-LD (schema.org/Dataset): [dataset.json](dataset.json)
- Rows: 1000
- Columns: 21
- Discrete: 18  |  Continuous: 3

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
- **Original Data Source:** https://doi.org/10.24432/C5NC77
- **Reference (please cite)**: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.
- **Dataset Year:** 1994
- **Dataset Description:** see the reference and the original data source for details.

#### Curation comments by the TabArena team (for code see the [page of the study](https://tabarena.ai/data-tabular-ml-iid-study)):
- We revered the original ordinal encoding. 
- Anomaly: the original task used a cost matrix for evaluation.

- Creators: See original data source.
- Date: 2025-04-30T20:15:41
- Citation: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.
- Links:
  - URL: https://api.openml.org/data/v1/download/22125229/credit-g.arff
  - sameAs: https://doi.org/10.24432/C5NC77.
## Variables and summary

| variable                 | inferred   |   count | unique   | top                                 | freq   | mean     | std                | min   | 25%    | 50%    | 75%     | max     |
|:-------------------------|:-----------|--------:|:---------|:------------------------------------|:-------|:---------|:-------------------|:------|:-------|:-------|:--------|:--------|
| checking_status          | discrete   |    1000 | 4        | no checking account                 | 394    |          |                    |       |        |        |         |         |
| duration_months          | continuous |    1000 |          |                                     |        | 20.903   | 12.058814452756378 | 4.0   | 12.0   | 18.0   | 24.0    | 72.0    |
| credit_history           | discrete   |    1000 | 5        | existing credits paid duly till now | 530    |          |                    |       |        |        |         |         |
| credit_purpose           | discrete   |    1000 | 10       | radio/television                    | 280    |          |                    |       |        |        |         |         |
| credit_amount            | continuous |    1000 |          |                                     |        | 3271.258 | 2822.736875960441  | 250.0 | 1365.5 | 2319.5 | 3972.25 | 18424.0 |
| savings_status           | discrete   |    1000 | 5        | < 100 DM                            | 603    |          |                    |       |        |        |         |         |
| employment_since         | discrete   |    1000 | 5        | 1 <= ... < 4 years                  | 339    |          |                    |       |        |        |         |         |
| installment_rate_percent | discrete   |    1000 | 4        | 4                                   | 476    |          |                    |       |        |        |         |         |
| personal_status_sex      | discrete   |    1000 | 4        | male: single                        | 548    |          |                    |       |        |        |         |         |
| other_debtors            | discrete   |    1000 | 3        | none                                | 907    |          |                    |       |        |        |         |         |
| residence_since          | discrete   |    1000 | 4        | 4                                   | 413    |          |                    |       |        |        |         |         |
| property                 | discrete   |    1000 | 4        | car or other (not savings)          | 332    |          |                    |       |        |        |         |         |
| age_years                | continuous |    1000 |          |                                     |        | 35.546   | 11.375468574317505 | 19.0  | 27.0   | 33.0   | 42.0    | 75.0    |
| other_installment_plans  | discrete   |    1000 | 3        | none                                | 814    |          |                    |       |        |        |         |         |
| housing                  | discrete   |    1000 | 3        | own                                 | 713    |          |                    |       |        |        |         |         |
| existing_credits_count   | discrete   |    1000 | 4        | 1                                   | 633    |          |                    |       |        |        |         |         |
| job                      | discrete   |    1000 | 4        | skilled employee / official         | 630    |          |                    |       |        |        |         |         |
| people_liable            | discrete   |    1000 | 2        | 1                                   | 845    |          |                    |       |        |        |         |         |
| telephone                | discrete   |    1000 | 2        | none                                | 596    |          |                    |       |        |        |         |         |
| foreign_worker           | discrete   |    1000 | 2        | yes                                 | 963    |          |                    |       |        |        |         |         |
| good_or_bad_customer     | discrete   |    1000 | 2        | good                                | 700    |          |                    |       |        |        |         |         |

## Learned structures and configurations

MetaSyn GMF: [metasyn_gmf.json](metasyn_gmf.json)

### Arc blacklist

- Forbidden arc count: 0

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
| BN:clg_mi2  | -33.3695      | 4.5068       | -6507.0569   |          0.0575 |            0.055  |         0.1467 |        280.426 |
| BN:semi_mi5 | -34.0024      | 11.9792      | -6630.4653   |          0.0487 |            0.0474 |         0.1333 |        275.397 |
| MetaSyn     |               |              |              |          0.0517 |            0.0545 |         0.0917 |        118.862 |

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
      <td>age_years</td>
      <td>continuous</td>
      <td>0.1096</td>
      <td>0.0938</td>
      <td>0.0934</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>checking_status</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.130</td>
      <td>0.102</td>
      <td>0.144</td>
      <td>2.3982</td>
      <td>2.2424</td>
      <td>1.9838</td>
    </tr>
    <tr>
      <td>credit_amount</td>
      <td>continuous</td>
      <td>0.0763</td>
      <td>0.0780</td>
      <td>0.0548</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>credit_history</td>
      <td>discrete</td>
      <td>0.1162</td>
      <td>0.0939</td>
      <td>0.1040</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>credit_purpose</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.165</td>
      <td>0.161</td>
      <td>0.055</td>
      <td>836.7660</td>
      <td>821.6475</td>
      <td>352.9769</td>
    </tr>
    <tr>
      <td>duration_months</td>
      <td>continuous</td>
      <td>0.0798</td>
      <td>0.0792</td>
      <td>0.0779</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>employment_since</td>
      <td>discrete</td>
      <td>0.0743</td>
      <td>0.0416</td>
      <td>0.0670</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>existing_credits_count</td>
      <td>discrete</td>
      <td>0.0396</td>
      <td>0.0254</td>
      <td>0.0542</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>foreign_worker</td>
      <td>discrete</td>
      <td>0.0474</td>
      <td>0.0618</td>
      <td>0.0500</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>good_or_bad_customer</td>
      <td>discrete</td>
      <td>0.0556</td>
      <td>0.0444</td>
      <td>0.0680</td>
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
      <td>0.0455</td>
      <td>0.0579</td>
      <td>0.0574</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>installment_rate_percent</td>
      <td>discrete</td>
      <td>0.0593</td>
      <td>0.0405</td>
      <td>0.0335</td>
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
      <td></td>
      <td></td>
      <td></td>
      <td>0.145</td>
      <td>0.137</td>
      <td>0.076</td>
      <td>2.1151</td>
      <td>2.3000</td>
      <td>1.6250</td>
    </tr>
    <tr>
      <td>other_debtors</td>
      <td>discrete</td>
      <td>0.0543</td>
      <td>0.0581</td>
      <td>0.0564</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>other_installment_plans</td>
      <td>discrete</td>
      <td>0.0378</td>
      <td>0.0113</td>
      <td>0.0245</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>people_liable</td>
      <td>discrete</td>
      <td>0.0664</td>
      <td>0.0783</td>
      <td>0.0904</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>personal_status_sex</td>
      <td>discrete</td>
      <td>0.0431</td>
      <td>0.0340</td>
      <td>0.0253</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>property</td>
      <td>discrete</td>
      <td>0.0215</td>
      <td>0.0012</td>
      <td>0.0226</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>residence_since</td>
      <td>discrete</td>
      <td>0.0096</td>
      <td>0.0233</td>
      <td>0.0052</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>savings_status</td>
      <td>discrete</td>
      <td>0.0288</td>
      <td>0.0041</td>
      <td>0.0171</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>telephone</td>
      <td>discrete</td>
      <td>0.0704</td>
      <td>0.0504</td>
      <td>0.0290</td>
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

