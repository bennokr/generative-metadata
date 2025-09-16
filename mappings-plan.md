# Plan: Integrate semmap.py and jsonld_to_html.py with mappings/ to enrich metadata and downloadable artifacts

This plan describes how to integrate:
- The SemMap metadata accessors (semmap.py)
- The JSON-LD-to-HTML renderer (jsonld_to_html.py)
- The dataset-specific mappings in mappings/

…to (a) improve dataset metadata (content and rendering) and (b) serialize synthetic data together with this metadata for download.

## Goals
- Accept dataset-specific mappings from mappings/ and convert them into a clean, standardized JSON-LD profile.
- Apply this JSON-LD metadata to the in-memory DataFrame using semmap.py, so that:
  - Column-level metadata (units, code schemes, descriptions, mappings to ontologies) is attached to Series.
  - Dataset-level metadata (title, description, license) is attached to the DataFrame.
- Merge/align the SemMap JSON-LD with the existing schema.org Dataset JSON-LD we produce (dataset.json) to maximize web discoverability, while preserving rich metadata for download.
- Render the rich SemMap JSON-LD into lightweight HTML for the report page, using jsonld_to_html.py.
- Serialize all synthetic datasets with their SemMap JSON-LD embedded, so users can download a single Parquet that round-trips the metadata.

## Inputs and expected formats
- inputs:
  - DataFrame df and its inferred types (already present in pipeline).
  - Provider and dataset identifiers (openml/uciml).
  - Curated JSON-LD mappings in mappings/ (e.g., mappings/uciml-45.metadata.json, mappings/uciml-336.metadata.json).

- expected standardized JSON-LD (SemMap/DDI/DCAT/DSV blend):
  - Dataset-level metadata in df.attrs["semmap_jsonld"], minimally including:
    - @context: https://w3id.org/semmap/context/v1
    - @type: [disco:LogicalDataSet, dsv:Dataset, dcat:Dataset]
    - dct:title, dct:description, dct:license (if known)
  - Column-level metadata on each Series.attrs["semmap_jsonld"], minimally including:
    - @context
    - @type: [disco:Variable, dsv:Column]
    - skos:notation (column key), skos:prefLabel (human label)
    - disco:representation (numeric or categorical), optional qudt:hasUnit and semmap:pintUnit

SemMap handles these shapes and persists them into Parquet (Arrow schema/field metadata), with optional pint dtype round-trip.

## High-level architecture changes
1) Use curated JSON-LD mappings directly
   - Convert the current mappings under mappings/ into curated JSON-LD files (done in-repo):
     - mappings/uciml-45.metadata.json
     - mappings/uciml-336.metadata.json
   - Each file follows tests/fixtures/heart.metadata.json structure:
     {
       "dataset": { … dataset JSON-LD … },
       "disco:variable": [ { … column JSON-LD … }, … ]
     }
   - We will use these JSON files directly; no runtime translation step is needed.

2) Pipeline integration (bncli.pipeline.process_dataset)
   - After loading df and before inferring/merging schema.org JSON-LD, if a curated mapping exists, load the JSON-LD file directly.
   - Apply it to the DataFrame using df.semmap.apply_json_metadata(curated_jsonld, convert_pint=True):
     - This attaches dataset-level and column-level JSON-LD and optionally converts pint units.
   - Build the schema.org Dataset JSON-LD as today, but enrich its variableMeasured entries with:
     - description, units, value labels when present in SemMap metadata.
     - Where conflicts exist, prefer SemMap-provided values.
   - Continue with the rest of the pipeline: modeling, UMAP, etc.

3) Report rendering
   - Replace/augment the current “Variables and summary” block with a richer HTML block generated from SemMap JSON-LD:
     - Use jsonld_to_html.render_microdata (or RDFa) to render df.semmap.export_json_metadata() into HTML.
     - Embed the generated HTML fragment directly in report.md (as raw HTML), just like we already do for images and custom tables.
     - Keep the existing summary statistics table below or above the SemMap block; the SemMap block is the authoritative metadata presentation.
   - Optionally write a separate dataset.semmap.html file for deep inspection and link it from the report.

4) Synthetic data serialization with metadata
   - For each synthetic dataset (BN configs and MetaSyn):
     - Copy or apply the same SemMap metadata to the synthetic DataFrame:
       - exported = df_no_na.semmap.export_json_metadata()
       - synth_df.semmap.apply_json_metadata(exported, convert_pint=False)
     - Serialize to Parquet with embedded metadata:
       - synth_df.semmap.to_parquet("synthetic_bn_<label>.semmap.parquet", index=False)
       - synth_meta_df.semmap.to_parquet("synthetic_metasyn.semmap.parquet", index=False)
     - Also write JSON export alongside Parquet for transparency: synthetic_bn_<label>.semmap.json
   - Add links in the report (Serialization section) to these downloads.

## Detailed steps

### A) Curated mapping discovery
- Add bncli/mappings.py with:
  - resolve_mapping_json(provider: str, provider_id: Optional[int], dataset_name: Optional[str]) -> Optional[Path]
    - Convention: look for mappings/<provider>-<id>.metadata.json; optionally mappings/<provider>-<name>.metadata.json as fallback.
  - load_mapping_json(path: Path) -> Dict[str, Any]
    - Validate basic shape: {"dataset":{…}, "disco:variable":[…]}
    - Optionally normalize @context if missing.

### B) Apply mapping in pipeline
- In process_dataset:
  - As soon as df is loaded and lightly cleaned (ID/index cols removed), attempt:
    - path = resolve_mapping_json(provider, provider_id, name)
    - if path: curated = load_mapping_json(path); df.semmap.apply_json_metadata(curated, convert_pint=True)
  - Keep current dtype inference and schema.org JSON-LD build. For variableMeasured rows:
    - Read column SemMap metadata when present to enrich entries with description/unit/value labels (e.g., first 5 code labels or link count). Prefer SemMap over other sources.
  - Persist df.semmap.export_json_metadata() into outdir as dataset.semmap.json for transparency.

### C) Enhance rendering
- In reporting.write_report_md:
  - Produce an HTML fragment for SemMap JSON-LD:
    - meta = df.semmap.export_json_metadata()
    - html_fragment = jsonld_to_html.render_microdata(meta, context=meta.get("@context", "https://schema.org/"))
    - Write this fragment under a new section “Metadata (rich)” above the current table.
  - Keep the existing Variables + summary table for stats; the SemMap block provides richer semantics (units, code schemes, links, etc.).
  - Optionally: write the SemMap HTML into a separate file (dataset.semmap.html) using jsonld_to_html.wrap_html and link it.

### D) Synthetic data with SemMap metadata
- For each synthetic DataFrame (BN variants and MetaSyn):
  - metadata_export = df_no_na.semmap.export_json_metadata()
  - synth_df.semmap.apply_json_metadata(metadata_export, convert_pint=False)
  - synth_df.semmap.to_parquet(outdir/"synthetic_bn_<label>.semmap.parquet", index=False)
  - Write JSON: outdir/"synthetic_bn_<label>.semmap.json"
- Do the same for MetaSyn: synthetic_metasyn.semmap.parquet/json.
- Add hyperlinks to these files in the corresponding sections of the report.

### E) CLI additions
- bncli/cli.py report: add flags
  - --mapping auto|PATH (default: auto)
  - --export-semmap/--no-export-semmap (default: true)
  - --serialize-synthetic/--no-serialize-synthetic (default: true)
- Auto mode resolves mappings as described; if none found, silently skip with a log line.

### F) JSON-LD merging strategy
- Keep dataset.json as a schema.org-focused artifact for web search, links, and a compact overview.
- Maintain a separate dataset.semmap.json as the canonical rich metadata.
- Enrich schema.org variableMeasured with selected fields (description, unit) derived from SemMap, but do not embed the entire SemMap model to avoid bloating.
- Precedence: SemMap > provider-derived > inferred.

### G) Download packaging (optional)
- Zip bundle for each model containing:
  - synthetic_bn_<label>.semmap.parquet
  - synthetic_bn_<label>.semmap.json
  - dataset.semmap.json
  - dataset.json (schema.org)
  - report.md + index.html
- Link the zip on the report page.

## Testing
- Unit tests (pytest):
  - tests/test_semmap.py already exercises Parquet round-trip. Extend with:
    - Applying JSON-LD exported from mappings to a small fixture DataFrame, verifying Series/DataFrame attrs are set and pint conversion works when requested.
    - Round-trip for synthetic outputs: write parquet, read back, compare metadata JSON.
  - Golden tests for jsonld_to_html rendering of typical mapping structures (numeric with unit, categorical with code labels & exactMatch IRIs).
- Integration tests:
  - Run the pipeline on uciml id 45 using mappings/uciml-45.md, assert that:
    - outdir/dataset.semmap.json exists and includes expected variable structures.
    - outdir/dataset.json variableMeasured includes SemMap-derived descriptions/units.
    - outdir/synthetic_bn_*.semmap.parquet exist; reading them restores JSON-LD in df.attrs/series.attrs.

## Migration and rollout
- Phase 1: Implement mapping compiler + pipeline hooks; keep existing report sections.
- Phase 2: Add SemMap HTML rendering block; adjust CSS (minimal) if needed.
- Phase 3: Add synthetic serialization with SemMap; add links to report.
- Phase 4: Optional: add bundle .zip packaging.

## Risks and mitigations
- Mapping parser robustness:
  - Define a simple, explicit mapping schema inside fenced code blocks (JSON/YAML). Validate before applying.
- JSON-LD bloat on the page:
  - Keep schema.org dataset.json concise; render SemMap HTML separately.
- Unit conversion errors:
  - pint is optional; always handle absence gracefully; avoid coercing when units are ambiguous.
- Column name drift:
  - Match variables from mappings to df columns case-insensitively; log unmatched entries.

- ## Documentation updates
- README: Add a section “Using mappings/ to enrich metadata” with:
  - The curated JSON-LD file convention and an example.
  - How to run with --mapping auto.
  - Where artifacts are written (dataset.semmap.json, dataset.semmap.html, synthetic parquet/json files).
- docs/: Consider a short “Metadata model (SemMap)” page with links to the context and examples.

## Estimated effort
- Mapping translator + tests: 1–2 days (depends on mappings/ complexity and conventions).
- Pipeline wiring + report integration: 0.5–1 day.
- Synthetic serialization + links: 0.5 day.
- Documentation + polish: 0.5 day.
