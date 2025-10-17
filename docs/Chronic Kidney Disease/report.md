# Data Report — Chronic Kidney Disease

**Source**: [UCI dataset 336](https://archive.ics.uci.edu/dataset/336)

- SemMap JSON-LD: [dataset.semmap.json](dataset.semmap.json)
- SemMap HTML: [dataset.semmap.html](dataset.semmap.html)
- Rows: 158
- Columns: 25
- Discrete: 14  |  Continuous: 11

## Variables and summary

| variable   | inferred   | dist                                                  |
|:-----------|:-----------|:------------------------------------------------------|
| age        | continuous | 49.5633 ± 15.5122 [6, 39.25, 50.5, 60, 83]            |
| bp         | discrete   | 80: 63 (39.87%)                                       |
|            |            | 60: 40 (25.32%)                                       |
|            |            | 70: 37 (23.42%)                                       |
|            |            | 90: 9 (5.70%)                                         |
|            |            | 100: 7 (4.43%)                                        |
|            |            | 50: 1 (0.63%)                                         |
|            |            | 110: 1 (0.63%)                                        |
| sg         | continuous | 1.0199 ± 0.0055 [1.005, 1.02, 1.02, 1.025, 1.025]     |
| al         | discrete   | 0: 116 (73.42%)                                       |
|            |            | 4: 15 (9.49%)                                         |
|            |            | 3: 15 (9.49%)                                         |
|            |            | 2: 9 (5.70%)                                          |
|            |            | 1: 3 (1.90%)                                          |
| su         | discrete   | 0: 140 (88.61%)                                       |
|            |            | 2: 6 (3.80%)                                          |
|            |            | 1: 6 (3.80%)                                          |
|            |            | 3: 3 (1.90%)                                          |
|            |            | 4: 2 (1.27%)                                          |
|            |            | 5: 1 (0.63%)                                          |
| rbc        | discrete   | 140 (88.61%)                                          |
| pc         | discrete   | 129 (81.65%)                                          |
| pcc        | discrete   | 144 (91.14%)                                          |
| ba         | discrete   | 146 (92.41%)                                          |
| bgr        | continuous | 131.3418 ± 64.9398 [70, 97, 115.5, 131.75, 490]       |
| bu         | continuous | 52.5759 ± 47.3954 [10, 26, 39.5, 49.75, 309]          |
| sc         | continuous | 2.1886 ± 3.0776 [0.4, 0.7, 1.1, 1.6, 15.2]            |
| sod        | continuous | 138.8481 ± 7.4894 [111, 135, 139, 144, 150]           |
| pot        | continuous | 4.6367 ± 3.4764 [2.5, 3.7, 4.5, 4.9, 47]              |
| hemo       | continuous | 13.6873 ± 2.8822 [3.1, 12.6, 14.25, 15.775, 17.8]     |
| pcv        | continuous | 41.9177 ± 9.1052 [9, 37.5, 44, 48, 54]                |
| wbcc       | continuous | 8475.9494 ± 3126.8802 [3800, 6525, 7800, 9775, 26400] |
| rbcc       | continuous | 4.8918 ± 1.0194 [2.1, 4.5, 4.95, 5.6, 8]              |
| htn        | discrete   | 34 (21.52%)                                           |
| dm         | discrete   | 28 (17.72%)                                           |
| cad        | discrete   | 11 (6.96%)                                            |
| appet      | discrete   | 139 (87.97%)                                          |
| pe         | discrete   | 20 (12.66%)                                           |
| ane        | discrete   | 16 (10.13%)                                           |
| class      | discrete   | 115 (72.78%)                                          |

## Fidelity summary

| model      | backend   |   disc_jsd_mean |   disc_jsd_median |   cont_ks_mean |   cont_w1_mean |
|:-----------|:----------|----------------:|------------------:|---------------:|---------------:|
| clg_mi2    | pybnesian |          0.0544 |            0.0502 |         0.2405 |        49.3847 |
| semi_mi5   | pybnesian |          0.0419 |            0.0411 |         0.2446 |        59.6372 |
| ctgan_fast | synthcity |          0.1475 |            0.1492 |         0.6607 |       880.088  |
| tvae_quick | synthcity |          0.1676 |            0.1862 |         0.2515 |        74.6213 |

## Models

<table>
<tr><th>UMAP</th><th>Details</th><th>Structure</th></tr>
<tr><td><img src='umap_real.png' width='280'/></td><td><h3>Real data</h3></td><td></td></tr>
<tr><td>
<img src='models/clg_mi2/umap.png' width='280'/></td><td>

<h3>Model: clg_mi2 (pybnesian)</h3>
<ul>
<li>Seed: 42, rows: 126</li>
<li>Params: <code>{"max_indegree": 2, "operators": ["arcs"], "score": "bic", "type": "clg"}</code></li>
<li><a href="models/clg_mi2/synthetic.csv">Synthetic CSV</a></li>
<li><a href="models/clg_mi2/per_variable_metrics.csv">Per-variable metrics</a></li>
<li><a href="models/clg_mi2/metrics.json">Metrics JSON</a></li>
</ul></td><td>

<a href='models/clg_mi2/structure.png'><img src='models/clg_mi2/structure.png' width='280'/></a>
</td></tr>


<tr><td>
<img src='models/semi_mi5/umap.png' width='280'/></td><td>

<h3>Model: semi_mi5 (pybnesian)</h3>
<ul>
<li>Seed: 42, rows: 126</li>
<li>Params: <code>{"max_indegree": 5, "operators": ["arcs"], "score": "bic", "type": "semiparametric"}</code></li>
<li><a href="models/semi_mi5/synthetic.csv">Synthetic CSV</a></li>
<li><a href="models/semi_mi5/per_variable_metrics.csv">Per-variable metrics</a></li>
<li><a href="models/semi_mi5/metrics.json">Metrics JSON</a></li>
</ul></td><td>

<a href='models/semi_mi5/structure.png'><img src='models/semi_mi5/structure.png' width='280'/></a>
</td></tr>


<tr><td>
<img src='models/ctgan_fast/umap.png' width='280'/></td><td>

<h3>Model: ctgan_fast (synthcity)</h3>
<ul>
<li>Seed: 42, rows: 1000</li>
<li>Params: <code>{"batch_size": 256, "n_iter": 5}</code></li>
<li><a href="models/ctgan_fast/synthetic.csv">Synthetic CSV</a></li>
<li><a href="models/ctgan_fast/per_variable_metrics.csv">Per-variable metrics</a></li>
<li><a href="models/ctgan_fast/metrics.json">Metrics JSON</a></li>
</ul></td><td>

</td></tr>


<tr><td>
<img src='models/tvae_quick/umap.png' width='280'/></td><td>

<h3>Model: tvae_quick (synthcity)</h3>
<ul>
<li>Seed: 42, rows: 1000</li>
<li>Params: <code>{"batch_size": 256}</code></li>
<li><a href="models/tvae_quick/synthetic.csv">Synthetic CSV</a></li>
<li><a href="models/tvae_quick/per_variable_metrics.csv">Per-variable metrics</a></li>
<li><a href="models/tvae_quick/metrics.json">Metrics JSON</a></li>
</ul></td><td>

</td></tr>


<table>

