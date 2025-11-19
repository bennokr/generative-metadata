# SemMap metadata schema

This project represents dataset semantics using a SemMap-flavoured JSON-LD
profile that blends [DCAT](https://www.w3.org/TR/vocab-dcat-3/)/[DSV](https://w3id.org/dsv-ontology),
[CSVW](https://www.w3.org/TR/tabular-data-model/), [PROV](https://www.w3.org/TR/prov-o/),
[QUDT](https://qudt.org/)/[UCUM](https://ucum.org/), and
[SKOS](https://www.w3.org/TR/skos-reference/) mappings. The
canonical definitions live in `semsynth.semmap` and power every stage of the
pipeline, from ingestion to reporting.

## Core objects

- **Metadata**: Root document that carries dataset-level [DCAT](https://www.w3.org/TR/vocab-dcat-3/)
  information
  (title, description, purpose, landing page, citations, identifiers, funding,
  access rights) plus `summaryStatistics` such as dataset completeness and row
  counts. It contains a `DatasetSchema`.
- **DatasetSchema**: Holds an ordered list of `Column` nodes.
- **Column**: Captures [CSVW](https://www.w3.org/TR/tabular-data-model/)/[DSV](https://w3id.org/dsv-ontology)
  fields (`name`, `titles`, descriptions,
  `prov:hadRole`, defaults) and optional `summaryStatistics` with declared
  `statisticalDataType`, completeness, missing-value formats, and numeric
  aggregates. Columns may embed a `ColumnProperty`.
- **ColumnProperty**: Encodes richer semantics including units
  ([UCUM](https://ucum.org/)/[QUDT](https://qudt.org/)), codebooks
  (`hasCodeBook` with [SKOS](https://www.w3.org/TR/skos-reference/) concepts),
  links to variable definitions, provenance (`source`), and
  [SKOS](https://www.w3.org/TR/skos-reference/) mappings
  (`exactMatch`, `closeMatch`, etc.).
- **CodeBook/CodeConcept**: Capture enumerations or codelists, each
  [SKOS](https://www.w3.org/TR/skos-reference/) concept
  supporting mappings to external ontologies. Presence of a codebook implies a
  categorical type for privacy metadata.

Every dataclass inherits `JSONLDMixin`, enabling round-trips via
`to_jsonld()/from_jsonld` and storage inside parquet files through the
pandas `semmap` accessor.

## Creation and ingestion

1. **Templates**: `semsynth/dataproviders/uciml.py` emits [DCAT](https://www.w3.org/TR/vocab-dcat-3/)/[DSV](https://w3id.org/dsv-ontology)
   JSON-LD that matches the SemMap dataclasses. The same layout is loaded by
   `Metadata.from_dcat_dsv`.
2. **Column mapping**: `map_columns/shared.py` parses curated metadata for LLM
   prompts and returns a `Metadata` instance; `map_columns/sssom_to_semmap.py`
   merges [SSSOM](https://w3id.org/sssom/) mapping files into `ColumnProperty`
   [SKOS](https://www.w3.org/TR/skos-reference/) relationships to
   keep column mappings aligned with downstream semantics.
3. **Pipeline loading**: `semsynth.pipeline.Preprocessor` applies curated
   metadata (`DatasetSpec.meta` or JSON-LD files) to incoming dataframes through
   the pandas SemMap accessor, producing a shared `Metadata` object and
   JSON-LD export for later stages.

## Updates during preprocessing

- **Type inference**: Inferred discrete/continuous hints are stored alongside
  metadata and used as fallbacks when semantics are incomplete.
- **Missingness**: When missingness wrapping is enabled, `Metadata` is updated
  via `update_completeness_from_missingness` to refresh dataset/column
  completeness and missing-value annotations derived from the fitted model.
- **Persistence**: The updated metadata is carried through
  `PreprocessingResult.semmap_metadata` and stored with artifacts for each run.

## Use in analytics

- **Privacy metrics**: `Metadata.to_privacy_frame` normalizes SemMap roles to
  the privacy expectations (`qi`, `sensitive`, `id`, `ignore`, `target`) and
  maps `statisticalDataType` or codebooks to coarse types
  (`numeric`/`categorical`). The resulting dataframe is passed to
  `privacy_metrics.summarize_privacy_synthcity`, ensuring curated roles and
  types drive quasi-identifier selection and sensitive feature handling. When
  SemMap metadata is unavailable, the pipeline falls back to the inferred type
  map.
- **Downstream fidelity**: The same `Metadata` instance is serialized to JSON-LD
  and provided to `downstream_fidelity.compare_real_vs_synth`, allowing role and
  codebook semantics to guide modeling and encoding.

## Reporting and exports

`semsynth.reporting.write_report_md` receives the shared `Metadata` and renders
human-readable dataset descriptions, citations, and completeness figures that
match the SemMap export stored alongside backend outputs. This keeps reports,
privacy metrics, and SemMap artifacts consistent.
