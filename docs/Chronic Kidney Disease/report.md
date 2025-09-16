# Data Report — Chronic Kidney Disease

**Source**: [UCI dataset 336](https://archive.ics.uci.edu/dataset/336)

- Metadata file: [metadata.json](metadata.json)
- JSON-LD (schema.org/Dataset): [dataset.json](dataset.json)
- SemMap JSON-LD: [dataset.semmap.json](dataset.semmap.json)
- SemMap HTML: [dataset.semmap.html](dataset.semmap.html)
- Rows: 158
- Columns: 25
- Discrete: 11  |  Continuous: 14

## Dataset metadata

- Name: Chronic Kidney Disease (UCI id 336)

### Description

Clinical records for early detection of CKD (subset of variables mapped).

- Creators: L. Rubini, P. Soundarapandian, P. Eswaran
- Links:
  - URL: https://archive.ics.uci.edu/dataset/336

## Metadata (rich)

[Standalone SemMap metadata view](dataset.semmap.html)

<style>
.semmap-metadata { display: grid; gap: 1.5rem; margin: 1.5rem 0; }
.semmap-metadata .item { border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.25rem 1.5rem; background: linear-gradient(180deg, rgba(248, 250, 252, 0.6), #ffffff); box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05); }
.semmap-metadata .item-title { margin: 0 0 .75rem; font-size: 1.1rem; color: #0f172a; }
.semmap-metadata .prop { margin: .3rem 0; color: #334155; }
.semmap-metadata .name { font-weight: 600; margin-right: .35rem; color: #0f172a; }
.semmap-metadata .prop-table { border-collapse: collapse; width: 100%; margin: .35rem 0 1rem; font-size: .95rem; }
.semmap-metadata .prop-table th, .semmap-metadata .prop-table td { border: 1px solid #e2e8f0; padding: .45rem .6rem; text-align: left; }
.semmap-metadata .prop-table th { background: #f8fafc; color: #475569; text-transform: uppercase; letter-spacing: .04em; font-size: .7rem; }
</style>
<div class="semmap-metadata">
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chronic Kidney Disease — SemMap metadata</title>
  <style>
    body { font-family: system-ui, Arial, sans-serif; line-height: 1.5; margin: 2rem; }
    .item { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
    .item-title { margin: 0 0 .5rem; font-size: 1.1rem; }
    .prop { margin: .2rem 0; }
    .name { font-weight: 600; margin-right: .25rem; }
    .id-link { margin: .2rem 0; }
    table.prop-table { border-collapse: collapse; width: 100%; margin: .25rem 0 1rem; }
    table.prop-table th, table.prop-table td { border: 1px solid #ddd; padding: .35rem .5rem; text-align: left; vertical-align: top; }
    table.prop-table th { background: #f3f3f3; }
  </style>
</head>
<body>
<h1>Chronic Kidney Disease — SemMap metadata</h1>
<div class="item" vocab="https://w3id.org/semmap/context/v1/" typeof="Thing"><div property="dataset" typeof="disco:LogicalDataSet dsv:Dataset dcat:Dataset"><div class="prop"><span class="name">dct:title</span><span property="dct:title">Chronic Kidney Disease (UCI id 336)</span></div>
<div class="prop"><span class="name">dct:description</span><span property="dct:description">Clinical records for early detection of CKD (subset of variables mapped).</span></div></div>
<div class="prop"><span class="name">disco:variable</span></div><table class="prop-table" data-prop="disco:variable"><thead><tr><th>disco:representation</th><th>skos:notation</th><th>skos:prefLabel</th><th>dct:source</th><th>dct:description</th><th>dct:title</th></tr></thead><tr property="disco:variable" typeof="disco:LogicalDataSet dsv:Dataset dcat:Dataset"><td></td><td></td><td></td><td></td><td><span property="dct:description">Clinical records for early detection of CKD (subset of variables mapped).</span></td><td><span property="dct:title">Chronic Kidney Disease (UCI id 336)</span></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation"><h2 class="item-title">disco:Representation</h2>
<div class="prop"><span class="name">dsv:valueType</span><span property="dsv:valueType">xsd:integer</span></div>
<div class="prop"><span class="name">qudt:hasUnit</span><span property="qudt:hasUnit">unit:MilliM_HG</span></div>
<div class="prop"><span class="name">semmap:pintUnit</span><span property="semmap:pintUnit">mmHg</span></div></div></td><td><span property="skos:notation">bp</span></td><td><span property="skos:prefLabel">Diastolic blood pressure</span></td><td><a rel="dct:source" href="https://loinc.org/8462-4">https://loinc.org/8462-4</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation"><h2 class="item-title">disco:Representation</h2>
<div class="prop"><span class="name">dsv:valueType</span><span property="dsv:valueType">xsd:decimal</span></div></div></td><td><span property="skos:notation">sg</span></td><td><span property="skos:prefLabel">Urine specific gravity</span></td><td><a rel="dct:source" href="https://loinc.org/2965-2">https://loinc.org/2965-2</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation skos:ConceptScheme"></div></td><td><span property="skos:notation">al</span></td><td><span property="skos:prefLabel">Urine albumin (dipstick)</span></td><td><a rel="dct:source" href="https://loinc.org/50949-7">https://loinc.org/50949-7</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation skos:ConceptScheme"></div></td><td><span property="skos:notation">su</span></td><td><span property="skos:prefLabel">Urine glucose (dipstick)</span></td><td><a rel="dct:source" href="https://loinc.org/25428-4">https://loinc.org/25428-4</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation skos:ConceptScheme"></div></td><td><span property="skos:notation">rbc</span></td><td><span property="skos:prefLabel">RBCs in urine (presence)</span></td><td><a rel="dct:source" href="https://loinc.org/32776-7">https://loinc.org/32776-7</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation skos:ConceptScheme"></div></td><td><span property="skos:notation">pc</span></td><td><span property="skos:prefLabel">Urine leukocytes (presence)</span></td><td><a rel="dct:source" href="https://loinc.org/20455-2">https://loinc.org/20455-2</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation skos:ConceptScheme"></div></td><td><span property="skos:notation">pcc</span></td><td><span property="skos:prefLabel">Pus cell clumps (urine)</span></td><td><a rel="dct:source" href="https://loinc.org/67848-2">https://loinc.org/67848-2</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation skos:ConceptScheme"></div></td><td><span property="skos:notation">ba</span></td><td><span property="skos:prefLabel">Bacteria in urine (presence)</span></td><td><a rel="dct:source" href="https://loinc.org/25145-4">https://loinc.org/25145-4</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation"><h2 class="item-title">disco:Representation</h2>
<div class="prop"><span class="name">dsv:valueType</span><span property="dsv:valueType">xsd:integer</span></div>
<div class="prop"><span class="name">qudt:hasUnit</span><span property="qudt:hasUnit">unit:MilliGM-PER-DeciL</span></div>
<div class="prop"><span class="name">semmap:pintUnit</span><span property="semmap:pintUnit">mg/dL</span></div></div></td><td><span property="skos:notation">bgr</span></td><td><span property="skos:prefLabel">Blood glucose (random)</span></td><td></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation"><h2 class="item-title">disco:Representation</h2>
<div class="prop"><span class="name">dsv:valueType</span><span property="dsv:valueType">xsd:integer</span></div>
<div class="prop"><span class="name">qudt:hasUnit</span><span property="qudt:hasUnit">unit:MilliGM-PER-DeciL</span></div>
<div class="prop"><span class="name">semmap:pintUnit</span><span property="semmap:pintUnit">mg/dL</span></div></div></td><td><span property="skos:notation">bu</span></td><td><span property="skos:prefLabel">Blood urea</span></td><td></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation"><h2 class="item-title">disco:Representation</h2>
<div class="prop"><span class="name">dsv:valueType</span><span property="dsv:valueType">xsd:decimal</span></div>
<div class="prop"><span class="name">semmap:pintUnit</span><span property="semmap:pintUnit">mg/dL</span></div></div></td><td><span property="skos:notation">sc</span></td><td><span property="skos:prefLabel">Serum creatinine</span></td><td><a rel="dct:source" href="https://loinc.org/2160-0">https://loinc.org/2160-0</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation"><h2 class="item-title">disco:Representation</h2>
<div class="prop"><span class="name">dsv:valueType</span><span property="dsv:valueType">xsd:integer</span></div>
<div class="prop"><span class="name">semmap:pintUnit</span><span property="semmap:pintUnit">mmol/L</span></div></div></td><td><span property="skos:notation">sod</span></td><td><span property="skos:prefLabel">Serum sodium</span></td><td><a rel="dct:source" href="https://loinc.org/2951-2">https://loinc.org/2951-2</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation"><h2 class="item-title">disco:Representation</h2>
<div class="prop"><span class="name">dsv:valueType</span><span property="dsv:valueType">xsd:integer</span></div>
<div class="prop"><span class="name">semmap:pintUnit</span><span property="semmap:pintUnit">mmol/L</span></div></div></td><td><span property="skos:notation">pot</span></td><td><span property="skos:prefLabel">Serum potassium</span></td><td><a rel="dct:source" href="https://loinc.org/2823-3">https://loinc.org/2823-3</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation"><h2 class="item-title">disco:Representation</h2>
<div class="prop"><span class="name">dsv:valueType</span><span property="dsv:valueType">xsd:decimal</span></div>
<div class="prop"><span class="name">semmap:pintUnit</span><span property="semmap:pintUnit">g/dL</span></div></div></td><td><span property="skos:notation">hemo</span></td><td><span property="skos:prefLabel">Hemoglobin</span></td><td><a rel="dct:source" href="https://loinc.org/718-7">https://loinc.org/718-7</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation"><h2 class="item-title">disco:Representation</h2>
<div class="prop"><span class="name">dsv:valueType</span><span property="dsv:valueType">xsd:integer</span></div>
<div class="prop"><span class="name">semmap:pintUnit</span><span property="semmap:pintUnit">%</span></div></div></td><td><span property="skos:notation">pcv</span></td><td><span property="skos:prefLabel">Packed cell volume (hematocrit)</span></td><td><a rel="dct:source" href="https://loinc.org/20570-8">https://loinc.org/20570-8</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:LogicalDataSet dsv:Dataset dcat:Dataset"><td></td><td></td><td></td><td></td><td><span property="dct:description">Clinical records for early detection of CKD (subset of variables mapped).</span></td><td><span property="dct:title">Chronic Kidney Disease (UCI id 336)</span></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation"><h2 class="item-title">disco:Representation</h2>
<div class="prop"><span class="name">dsv:valueType</span><span property="dsv:valueType">xsd:decimal</span></div></div></td><td><span property="skos:notation">rbcc</span></td><td><span property="skos:prefLabel">RBC count (blood)</span></td><td><a rel="dct:source" href="https://loinc.org/789-8">https://loinc.org/789-8</a></td><td></td><td></td></tr><tr property="disco:variable" typeof="disco:LogicalDataSet dsv:Dataset dcat:Dataset"><td></td><td></td><td></td><td></td><td><span property="dct:description">Clinical records for early detection of CKD (subset of variables mapped).</span></td><td><span property="dct:title">Chronic Kidney Disease (UCI id 336)</span></td></tr><tr property="disco:variable" typeof="disco:LogicalDataSet dsv:Dataset dcat:Dataset"><td></td><td></td><td></td><td></td><td><span property="dct:description">Clinical records for early detection of CKD (subset of variables mapped).</span></td><td><span property="dct:title">Chronic Kidney Disease (UCI id 336)</span></td></tr><tr property="disco:variable" typeof="disco:LogicalDataSet dsv:Dataset dcat:Dataset"><td></td><td></td><td></td><td></td><td><span property="dct:description">Clinical records for early detection of CKD (subset of variables mapped).</span></td><td><span property="dct:title">Chronic Kidney Disease (UCI id 336)</span></td></tr><tr property="disco:variable" typeof="disco:LogicalDataSet dsv:Dataset dcat:Dataset"><td></td><td></td><td></td><td></td><td><span property="dct:description">Clinical records for early detection of CKD (subset of variables mapped).</span></td><td><span property="dct:title">Chronic Kidney Disease (UCI id 336)</span></td></tr><tr property="disco:variable" typeof="disco:LogicalDataSet dsv:Dataset dcat:Dataset"><td></td><td></td><td></td><td></td><td><span property="dct:description">Clinical records for early detection of CKD (subset of variables mapped).</span></td><td><span property="dct:title">Chronic Kidney Disease (UCI id 336)</span></td></tr><tr property="disco:variable" typeof="disco:LogicalDataSet dsv:Dataset dcat:Dataset"><td></td><td></td><td></td><td></td><td><span property="dct:description">Clinical records for early detection of CKD (subset of variables mapped).</span></td><td><span property="dct:title">Chronic Kidney Disease (UCI id 336)</span></td></tr><tr property="disco:variable" typeof="disco:Variable dsv:Column"><td><div property="disco:representation" typeof="disco:Representation skos:ConceptScheme"></div></td><td><span property="skos:notation">class</span></td><td><span property="skos:prefLabel">Outcome label (CKD)</span></td><td><a rel="dct:source" href="http://snomed.info/id/709044004">http://snomed.info/id/709044004</a></td><td></td><td></td></tr></table></div>
</body>
</html>

</div>

## Variables and summary

| variable   | description             | inferred   | declared    |   count | unique   | top        | freq   | mean                | std                  | min    | 25%    | 50%                | 75%    | max     |
|:-----------|:------------------------|:-----------|:------------|--------:|:---------|:-----------|:-------|:--------------------|:---------------------|:-------|:-------|:-------------------|:-------|:--------|
| age        |                         | continuous | Integer     |     391 |          |            |        | 51.48337595907928   | 17.16971408926224    | 2.0    | 42.0   | 55.0               | 64.5   | 90.0    |
| bp         | blood pressure          | continuous | Integer     |     388 |          |            |        | 76.46907216494846   | 13.683637493525255   | 50.0   | 70.0   | 80.0               | 80.0   | 180.0   |
| sg         | specific gravity        | continuous | Categorical |     353 |          |            |        | 1.0174079320113314  | 0.005716616974376362 | 1.005  | 1.01   | 1.02               | 1.02   | 1.025   |
| al         | albumin                 | continuous | Categorical |     354 |          |            |        | 1.0169491525423728  | 1.3526789127628434   | 0.0    | 0.0    | 0.0                | 2.0    | 5.0     |
| su         | sugar                   | continuous | Categorical |     351 |          |            |        | 0.45014245014245013 | 1.099191251885409    | 0.0    | 0.0    | 0.0                | 0.0    | 5.0     |
| rbc        | red blood cells         | discrete   | Binary      |     248 | 2        | normal     | 201    |                     |                      |        |        |                    |        |         |
| pc         | pus cell                | discrete   | Binary      |     335 | 2        | normal     | 259    |                     |                      |        |        |                    |        |         |
| pcc        | pus cell clumps         | discrete   | Binary      |     396 | 2        | notpresent | 354    |                     |                      |        |        |                    |        |         |
| ba         | bacteria                | discrete   | Binary      |     396 | 2        | notpresent | 374    |                     |                      |        |        |                    |        |         |
| bgr        | blood glucose random    | continuous | Integer     |     356 |          |            |        | 148.0365168539326   | 79.28171423511776    | 22.0   | 99.0   | 121.0              | 163.0  | 490.0   |
| bu         | blood urea              | continuous | Integer     |     381 |          |            |        | 57.425721784776904  | 50.5030058492225     | 1.5    | 27.0   | 42.0               | 66.0   | 391.0   |
| sc         | serum creatinine        | continuous | Continuous  |     383 |          |            |        | 3.072454308093995   | 5.741126066859781    | 0.4    | 0.9    | 1.3                | 2.8    | 76.0    |
| sod        | sodium                  | continuous | Integer     |     313 |          |            |        | 137.52875399361022  | 10.408752051798789   | 4.5    | 135.0  | 138.0              | 142.0  | 163.0   |
| pot        | potassium               | continuous | Continuous  |     312 |          |            |        | 4.62724358974359    | 3.1939041765566967   | 2.5    | 3.8    | 4.4                | 4.9    | 47.0    |
| hemo       | hemoglobin              | continuous | Continuous  |     348 |          |            |        | 12.526436781609195  | 2.9125866088267647   | 3.1    | 10.3   | 12.649999999999999 | 15.0   | 17.8    |
| pcv        | packed cell volume      | continuous | Integer     |     329 |          |            |        | 38.88449848024316   | 8.990104814740938    | 9.0    | 32.0   | 40.0               | 45.0   | 54.0    |
| wbcc       | white blood cell count  | continuous | Integer     |     294 |          |            |        | 8406.122448979591   | 2944.474190410339    | 2200.0 | 6500.0 | 8000.0             | 9800.0 | 26400.0 |
| rbcc       | red blood cell count    | continuous | Continuous  |     269 |          |            |        | 4.707434944237917   | 1.0253232655721793   | 2.1    | 3.9    | 4.8                | 5.4    | 8.0     |
| htn        | hypertension            | discrete   | Binary      |     398 | 2        | no         | 251    |                     |                      |        |        |                    |        |         |
| dm         | diabetes mellitus       | discrete   | Binary      |     398 | 3        | no         | 260    |                     |                      |        |        |                    |        |         |
| cad        | coronary artery disease | discrete   | Binary      |     398 | 2        | no         | 364    |                     |                      |        |        |                    |        |         |
| appet      | appetite                | discrete   | Binary      |     399 | 2        | good       | 317    |                     |                      |        |        |                    |        |         |
| pe         | pedal edema             | discrete   | Binary      |     399 | 2        | no         | 323    |                     |                      |        |        |                    |        |         |
| ane        | anemia                  | discrete   | Binary      |     399 | 2        | no         | 339    |                     |                      |        |        |                    |        |         |
| class      | ckd or not ckd          | discrete   | Binary      |     400 | 3        | ckd        | 248    |                     |                      |        |        |                    |        |         |

## Learned structures and configurations

MetaSyn GMF: [metasyn_gmf.json](metasyn_gmf.json)

MetaSyn serialization

- Synthetic sample (SemMap Parquet): [synthetic_metasyn.semmap.parquet](synthetic_metasyn.semmap.parquet)

### Arc blacklist

- Root variables: age
- Forbidden arc count: 25

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
| BN:clg_mi2  | -104.6948     | 390.3088     | -3350.2348   |          0.0438 |            0.0322 |         0.2272 |        53.9006 |
| BN:semi_mi5 | -104.1015     | 390.4263     | -3331.2472   |          0.0454 |            0.046  |         0.2211 |        43.8072 |
| MetaSyn     |               |              |              |          0.0462 |            0.0393 |         0.3051 |        49.1622 |

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
      <td>0.1190</td>
      <td>0.1190</td>
      <td>0.1530</td>
      <td>2.4896</td>
      <td>2.4896</td>
      <td>2.7723</td>
    </tr>
    <tr>
      <td>al</td>
      <td>continuous</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.3075</td>
      <td>0.2675</td>
      <td>0.2705</td>
      <td>3.8179</td>
      <td>4.0047</td>
      <td>3.6767</td>
    </tr>
    <tr>
      <td>ane</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.2892</td>
      <td>0.2852</td>
      <td>0.4062</td>
      <td>0.0017</td>
      <td>0.0018</td>
      <td>0.0019</td>
    </tr>
    <tr>
      <td>appet</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.3938</td>
      <td>0.3688</td>
      <td>0.7188</td>
      <td>0.3813</td>
      <td>0.3534</td>
      <td>0.6416</td>
    </tr>
    <tr>
      <td>ba</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.5135</td>
      <td>0.5165</td>
      <td>0.9375</td>
      <td>0.2876</td>
      <td>0.3111</td>
      <td>0.2820</td>
    </tr>
    <tr>
      <td>bgr</td>
      <td>continuous</td>
      <td>0.0322</td>
      <td>0.0439</td>
      <td>0.0477</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>bp</td>
      <td>continuous</td>
      <td>0.0747</td>
      <td>0.0803</td>
      <td>0.0703</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>bu</td>
      <td>continuous</td>
      <td>0.0655</td>
      <td>0.0528</td>
      <td>0.0381</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>cad</td>
      <td>discrete</td>
      <td>0.0910</td>
      <td>0.0927</td>
      <td>0.1033</td>
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
      <td>0.1430</td>
      <td>0.1470</td>
      <td>0.2890</td>
      <td>18.0992</td>
      <td>19.9513</td>
      <td>21.9635</td>
    </tr>
    <tr>
      <td>dm</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.2478</td>
      <td>0.2408</td>
      <td>0.1858</td>
      <td>8.6729</td>
      <td>8.2219</td>
      <td>12.8053</td>
    </tr>
    <tr>
      <td>hemo</td>
      <td>continuous</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.2155</td>
      <td>0.2515</td>
      <td>0.2790</td>
      <td>0.4597</td>
      <td>0.4577</td>
      <td>0.8323</td>
    </tr>
    <tr>
      <td>htn</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.1812</td>
      <td>0.1792</td>
      <td>0.1928</td>
      <td>2.2160</td>
      <td>2.4184</td>
      <td>2.1626</td>
    </tr>
    <tr>
      <td>pc</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.1812</td>
      <td>0.1582</td>
      <td>0.1565</td>
      <td>1.5103</td>
      <td>1.5024</td>
      <td>1.4807</td>
    </tr>
    <tr>
      <td>pcc</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.0985</td>
      <td>0.0945</td>
      <td>0.1380</td>
      <td>0.4116</td>
      <td>0.4239</td>
      <td>0.6638</td>
    </tr>
    <tr>
      <td>pcv</td>
      <td>continuous</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.1852</td>
      <td>0.1962</td>
      <td>0.2102</td>
      <td>1.8503</td>
      <td>1.9152</td>
      <td>2.2199</td>
    </tr>
    <tr>
      <td>pe</td>
      <td>discrete</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.2002</td>
      <td>0.1642</td>
      <td>0.2013</td>
      <td>714.2046</td>
      <td>571.0541</td>
      <td>638.5333</td>
    </tr>
    <tr>
      <td>pot</td>
      <td>continuous</td>
      <td></td>
      <td></td>
      <td></td>
      <td>0.1048</td>
      <td>0.1068</td>
      <td>0.1328</td>
      <td>0.2062</td>
      <td>0.1956</td>
      <td>0.2345</td>
    </tr>
    <tr>
      <td>rbc</td>
      <td>discrete</td>
      <td>0.0111</td>
      <td>0.0135</td>
      <td>0.0028</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>rbcc</td>
      <td>continuous</td>
      <td>0.0138</td>
      <td>0.0060</td>
      <td>0.0166</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sc</td>
      <td>continuous</td>
      <td>0.0856</td>
      <td>0.0918</td>
      <td>0.0840</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sg</td>
      <td>continuous</td>
      <td>0.0486</td>
      <td>0.0460</td>
      <td>0.0607</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sod</td>
      <td>continuous</td>
      <td>0.0224</td>
      <td>0.0188</td>
      <td>0.0336</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>su</td>
      <td>continuous</td>
      <td>0.0253</td>
      <td>0.0466</td>
      <td>0.0393</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>wbcc</td>
      <td>continuous</td>
      <td>0.0116</td>
      <td>0.0073</td>
      <td>0.0116</td>
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

