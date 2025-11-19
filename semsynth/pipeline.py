"""Pipeline orchestration for dataset processing and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING, cast

from makeprov import OutPath, rule

from .backends.base import BackendModule, ensure_backend_contract
from .mappings import load_mapping_json, resolve_mapping_json
from .metadata import get_uciml_variable_descriptions
from .models import ModelConfigBundle, discover_model_runs, load_model_configs
from .semmap import Metadata

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    from .specs import DatasetSpec


_BACKEND_MODULE_PATHS = {
    "pybnesian": "semsynth.backends.pybnesian",
    "synthcity": "semsynth.backends.synthcity",
    "metasyn": "semsynth.backends.metasyn",
}

_BACKEND_CACHE: Dict[str, BackendModule] = {}


@dataclass
class PipelineConfig:
    """Configuration values controlling the reporting pipeline."""

    random_state: int = 42
    max_umap_sample: int = 1000
    fit_on_sample: Optional[int] = 1000
    synthetic_sample: int = 1000
    test_size: float = 0.2
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    generate_umap: bool = True
    compute_privacy: bool = False
    compute_downstream: bool = False
    overwrite_umap: bool = False
    enable_missingness_wrapping: bool = False
    missingness_random_state: Optional[int] = None


def _load_backend_module(name: str) -> BackendModule:
    module_path = _BACKEND_MODULE_PATHS.get(name)
    if not module_path:
        raise ValueError(f"Unknown backend '{name}'")
    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - optional deps
        message = f"Failed to import backend '{name}'. "
        message += f"Install with `pip install semsynth[{name}]`."
        raise RuntimeError(message) from exc
    ensure_backend_contract(module)
    typed_module = cast(BackendModule, module)
    _BACKEND_CACHE[name] = typed_module
    return typed_module


def _resolve_flag(
    default_value: bool,
    bundle_value: Optional[bool],
    spec_value: Optional[bool],
) -> bool:
    if spec_value is not None:
        return spec_value
    if bundle_value is not None:
        return bundle_value
    return default_value


def _import_utils():  # pragma: no cover - helper for lazy import
    from . import utils as _utils

    return _utils


def _import_umap_utils():  # pragma: no cover - helper for lazy import
    from . import umap_utils as _umap

    return _umap


def _import_reporting():  # pragma: no cover - helper for lazy import
    from . import reporting as _reporting

    return _reporting


def _import_privacy_summarizer():  # pragma: no cover - helper for lazy import
    from .privacy_metrics import summarize_privacy_synthcity

    return summarize_privacy_synthcity


def _import_downstream_compare():  # pragma: no cover - helper for lazy import
    from .downstream_fidelity import compare_real_vs_synth

    return compare_real_vs_synth


def _build_privacy_metadata(df: "pd.DataFrame", inferred: Dict[str, str]) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for column, kind in inferred.items():
        dtype = "numeric" if kind == "continuous" else "categorical"
        rows.append({"variable": column, "role": "qi", "type": dtype})
    return pd.DataFrame(rows)


def _build_downstream_meta(
    df: "pd.DataFrame",
    inferred: Dict[str, str],
    target_series: Optional["pd.Series"],
) -> Dict[str, Any]:
    import pandas as pd

    target_name: Optional[str] = None
    if isinstance(target_series, pd.Series):
        if target_series.attrs.get("prov:hadRole") == "target":
            target_name = target_series.name if target_series.name in df.columns else None

    columns_meta: List[Dict[str, Any]] = []
    for column in df.columns:
        role = "target" if target_name and column == target_name else "predictor"
        kind = inferred.get(column, "continuous")
        col_meta: Dict[str, Any] = {
            "schema:name": column,
            "prov:hadRole": role,
        }

        stats_type = "dsv:QuantitativeDataType" if kind == "continuous" else "dsv:NominalDataType"
        col_meta["dsv:summaryStatistics"] = {"dsv:statisticalDataType": stats_type}

        if kind != "continuous":
            series = df[column].astype("category")
            categories = [str(cat) for cat in series.cat.categories]
            if categories:
                column_property: Dict[str, Any] = {
                    "dsv:hasCodeBook": {
                        "skos:hasTopConcept": [
                            {"skos:notation": cat, "skos:prefLabel": cat}
                            for cat in categories
                        ]
                    }
                }
                col_meta["dsv:columnProperty"] = column_property
                col_meta["schema:defaultValue"] = categories[0]
        columns_meta.append(col_meta)

    return {"dsv:datasetSchema": {"dsv:column": columns_meta}}


@dataclass
class UmapArtifacts:
    """Artifacts produced when UMAP visualisations are generated."""

    transformer: Any
    real_png: Path
    limits: Optional[Any]


@dataclass
class PreprocessingResult:
    """Result values produced by :class:`DatasetPreprocessor`."""

    df_processed: "pd.DataFrame"
    df_no_na: "pd.DataFrame"
    df_fit_sample: "pd.DataFrame"
    disc_cols: List[str]
    cont_cols: List[str]
    inferred_types: Dict[str, str]
    semmap_export: Optional[Dict[str, Any]]
    semmap_metadata: Optional[Metadata]
    color_series: Optional["pd.Series"]
    umap_png_real: Optional[Path]
    umap_artifacts: Optional[UmapArtifacts]
    missingness_model: Optional["DataFrameMissingnessModel"]


class DatasetPreprocessor:
    """Prepare dataset inputs prior to backend execution."""

    def __init__(
        self,
        *,
        utils_module: Any,
        load_mapping: Any,
        resolve_mapping: Any,
    ) -> None:
        self._utils = utils_module
        self._load_mapping = load_mapping
        self._resolve_mapping = resolve_mapping

    @rule()
    def preprocess(
        self,
        dataset_spec: "DatasetSpec",
        df: "pd.DataFrame",
        color_series: Optional["pd.Series"],
        outdir: Path,
        cfg: PipelineConfig,
        rng: Any,
        *,
        generate_umap: bool,
        umap_utils: Optional[Any],
    ) -> PreprocessingResult:
        """Clean data, apply metadata and optionally prepare UMAP artifacts.

        Args:
            dataset_spec: Dataset description metadata.
            df: Input dataframe to process.
            color_series: Optional colour labels for UMAP plots.
            outdir: Directory where artefacts should be stored.
            cfg: Active pipeline configuration.
            rng: Random state helper returned by ``utils.seed_all``.
            generate_umap: Whether to build UMAP artefacts.
            umap_utils: Module implementing UMAP helper routines.

        Returns:
            PreprocessingResult: Structured container with processed dataset
            artefacts used by later pipeline stages.
        """

        import pandas as pd

        self._utils.ensure_dir(str(outdir))

        semmap_export: Optional[Dict[str, Any]] = None
        semmap_metadata: Optional[Metadata] = None
        mapping_path = self._resolve_mapping(dataset_spec)
        if mapping_path is not None:
            logging.info("Applying curated SemMap metadata from %s", mapping_path)
            try:
                curated = self._load_mapping(mapping_path)
                df.semmap.from_jsonld(curated, convert_pint=True)
                semmap_metadata = df.semmap.dataset_semmap
                semmap_export = (
                    semmap_metadata.to_jsonld()
                    if semmap_metadata
                    else df.semmap.to_jsonld()
                )
            except Exception:
                logging.exception("Failed to apply SemMap metadata", exc_info=True)
        elif isinstance(dataset_spec.meta, Metadata):
            semmap_metadata = dataset_spec.meta
            meta_jsonld = semmap_metadata.to_jsonld()
            df.semmap.from_jsonld(meta_jsonld)
            semmap_export = meta_jsonld
        elif isinstance(dataset_spec.meta, dict):
            try:
                semmap_metadata = Metadata.from_dcat_dsv(dataset_spec.meta)
                meta_jsonld = semmap_metadata.to_jsonld()
                df.semmap.from_jsonld(meta_jsonld)
                semmap_export = meta_jsonld
            except Exception:
                logging.exception("Failed to apply dataset_spec metadata", exc_info=True)
        if semmap_metadata is None:
            semmap_metadata = df.semmap()
            semmap_export = semmap_metadata.to_jsonld()

        disc_cols, cont_cols = self._utils.infer_types(df)
        df_processed = self._utils.coerce_discrete_to_category(df, disc_cols)
        df_processed = self._utils.rename_categorical_categories_to_str(
            df_processed, disc_cols
        )
        df_processed = self._utils.coerce_continuous_to_float(df_processed, cont_cols)

        inferred_map = {
            column: ("discrete" if column in disc_cols else "continuous")
            for column in df_processed.columns
        }

        df_no_na = df_processed.dropna(axis=0, how="any").reset_index(drop=True)
        if df_no_na.empty:
            df_no_na = (
                df_processed.fillna(method="ffill")
                .fillna(method="bfill")
                .reset_index(drop=True)
            )

        color_series2 = None
        if isinstance(color_series, pd.Series) and color_series.name in df_no_na.columns:
            color_series2 = df_no_na[color_series.name].copy()
            color_series2.attrs.update(color_series.attrs)

        umap_png_real: Optional[Path] = outdir / "umap_real.png"
        umap_artifacts: Optional[UmapArtifacts] = None
        if generate_umap and umap_utils is not None:
            logging.info("Fitting UMAP on real data sample")
            umap_art = umap_utils.build_umap(
                df_no_na,
                disc_cols,
                cont_cols,
                color_series=color_series2,
                rng=rng,
                random_state=cfg.random_state,
                max_sample=cfg.max_umap_sample,
                n_neighbors=cfg.umap_n_neighbors,
                min_dist=cfg.umap_min_dist,
                n_components=cfg.umap_n_components,
            )

            umap_lims = umap_utils.plot_umap(
                umap_art.embedding,
                str(umap_png_real),
                title=f"{dataset_spec.name}: real (sample)",
                color_labels=umap_art.color_labels,
            )
            umap_artifacts = UmapArtifacts(
                transformer=umap_art, real_png=umap_png_real, limits=umap_lims
            )
        elif not umap_png_real.exists():
            umap_png_real = None

        df_fit_sample = df_no_na
        if cfg.fit_on_sample and cfg.fit_on_sample < len(df_fit_sample):
            df_fit_sample = df_no_na.sample(
                cfg.fit_on_sample, random_state=cfg.random_state
            )

        missingness_model: Optional["DataFrameMissingnessModel"] = None
        if cfg.enable_missingness_wrapping:
            miss_state = cfg.missingness_random_state
            if miss_state is None:
                miss_state = cfg.random_state
            try:
                from . import missingness as missingness_module
            except ImportError:
                logging.warning(
                    "Missingness wrapping requested but dependencies are unavailable"
                )
            else:
                missingness_model = missingness_module.fit_missingness_model(
                    df_processed, random_state=miss_state
                )

        if semmap_metadata is not None:
            semmap_metadata.update_completeness_from_missingness(
                df_processed, missingness_model
            )
            semmap_export = semmap_metadata.to_jsonld()

        return PreprocessingResult(
            df_processed=df_processed,
            df_no_na=df_no_na,
            df_fit_sample=df_fit_sample,
            disc_cols=disc_cols,
            cont_cols=cont_cols,
            inferred_types=inferred_map,
            semmap_export=semmap_export,
            semmap_metadata=semmap_metadata,
            color_series=color_series2,
            umap_png_real=umap_png_real,
            umap_artifacts=umap_artifacts,
            missingness_model=missingness_model,
        )


class MetricWriter:
    """Persist privacy and downstream metrics for backend runs."""

    def __init__(
        self,
        *,
        privacy_summarizer: Optional[Any] = None,
        downstream_compare: Optional[Any] = None,
    ) -> None:
        self._privacy_summarizer = privacy_summarizer
        self._downstream_compare = downstream_compare

    @rule()
    def write_privacy(
        self,
        run_dir: Path,
        real_df: "pd.DataFrame",
        inferred: Dict[str, str],
        synth_df: Optional["pd.DataFrame"] = None,
        metadata: Optional[Metadata] = None,
    ) -> Dict[str, Any]:
        """Write privacy metrics to disk for a backend run.

        Args:
            run_dir: Directory containing backend outputs.
            real_df: Real dataset without missing values.
            inferred: Mapping of column names to inferred types.
            synth_df: Optional synthetic dataframe to reuse.

        Returns:
            Dict[str, Any]: Serialized payload written to disk.
        """

        summarizer = self._privacy_summarizer or _import_privacy_summarizer()

        if synth_df is None:
            synth_df = self._read_synthetic_df(run_dir)
        metadata_df = (
            metadata.to_privacy_frame(inferred)
            if isinstance(metadata, Metadata)
            else _build_privacy_metadata(real_df, inferred)
        )
        summary = summarizer(real_df, synth_df, metadata_df)
        payload = asdict(summary)
        (run_dir / "metrics.privacy.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload

    @rule()
    def write_downstream(
        self,
        run_dir: Path,
        real_df: "pd.DataFrame",
        synth_df: "pd.DataFrame",
        inferred: Dict[str, str],
        target_series: Optional["pd.Series"],
        metadata: Optional[Metadata] = None,
    ) -> Dict[str, Any]:
        """Write downstream metrics comparing synthetic and real data.

        Args:
            run_dir: Directory containing backend outputs.
            real_df: Real dataset used for comparisons.
            synth_df: Synthetic dataset produced by backend.
            inferred: Mapping of column names to inferred types.
            target_series: Optional target series annotated with ``prov:hadRole``.

        Returns:
            Dict[str, Any]: Serialized payload written to disk.
        """

        comparer = self._downstream_compare or _import_downstream_compare()

        unique_targets: Optional[int] = None
        if target_series is not None:
            unique_targets = int(target_series.dropna().nunique())
        if unique_targets is not None and unique_targets > 2:
            logging.info(
                "Skipping downstream metrics for %s due to multiclass target", run_dir.name
            )
            payload = {
                "formula": None,
                "sign_match_rate": None,
                "skipped_reason": "multiclass_target",
            }
            (run_dir / "metrics.downstream.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            return payload

        meta_jsonld = metadata.to_jsonld() if isinstance(metadata, Metadata) else None
        if meta_jsonld is None:
            meta_jsonld = _build_downstream_meta(real_df, inferred, target_series)
        results = comparer(real_df, synth_df, meta_jsonld)
        compare = results.get("compare")
        sign_match_rate = float("nan")
        if hasattr(compare, "__getitem__"):
            try:
                series = compare["sign_match"]  # type: ignore[index]
                sign_match_rate = float(series.astype(float).mean())
            except Exception:
                sign_match_rate = float("nan")
        payload = {"formula": results.get("formula"), "sign_match_rate": sign_match_rate}
        (run_dir / "metrics.downstream.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload

    @staticmethod
    def _read_synthetic_df(run_dir: Path) -> "pd.DataFrame":
        import pandas as pd

        return pd.read_csv(run_dir / "synthetic.csv").convert_dtypes()


class BackendExecutor:
    """Execute backend configurations and evaluate resulting data."""

    def __init__(
        self,
        cfg: PipelineConfig,
        *,
        load_backend: Any,
        metric_writer: MetricWriter,
    ) -> None:
        self._cfg = cfg
        self._load_backend = load_backend
        self._metric_writer = metric_writer

    @rule()
    def run_models(
        self,
        dataset_spec: "DatasetSpec",
        bundle: ModelConfigBundle,
        preprocessed: PreprocessingResult,
        outdir: Path,
    ) -> None:
        """Execute each model specification and compute metrics if requested.

        Args:
            dataset_spec: Dataset metadata describing the run.
            bundle: Bundle of backend model configurations.
            preprocessed: Preprocessed dataset artefacts.
            outdir: Output directory for backend artefacts.
        """

        import pandas as pd

        bundle_specs = bundle.specs if bundle.specs else []

        for idx, spec in enumerate(bundle_specs):
            label = spec.name or f"model_{idx + 1}"
            seed = int(spec.seed if spec.seed is not None else self._cfg.random_state)
            backend_name = spec.backend
            try:
                backend_module = self._load_backend(backend_name)
            except Exception:
                logging.exception("Failed to load backend %s", backend_name)
                continue

            rows = spec.rows if spec.rows is not None else self._cfg.synthetic_sample
            try:
                run_dir = backend_module.run_experiment(
                    df=preprocessed.df_fit_sample,
                    provider=dataset_spec.provider,
                    dataset_name=dataset_spec.name,
                    provider_id=dataset_spec.id,
                    outdir=str(outdir),
                    label=label,
                    model_info=dict(spec.model or {}),
                    rows=min(rows, len(preprocessed.df_processed)),
                    seed=seed,
                    test_size=self._cfg.test_size,
                    semmap_export=preprocessed.semmap_export,
                )
            except Exception:
                logging.exception("%s run failed for %s", backend_name, label)
                continue

            run_dir_path = Path(run_dir)
            synth_path = run_dir_path / "synthetic.csv"
            synth_df = pd.read_csv(synth_path).convert_dtypes()

            missingness_applied = False
            if preprocessed.missingness_model is not None:
                try:
                    from . import missingness as missingness_module
                except ImportError:
                    logging.warning(
                        "Missingness wrapping requested but dependencies are unavailable"
                    )
                else:
                    synth_df, missingness_applied = (
                        missingness_module.apply_missingness_to_outputs(
                            run_dir=run_dir_path,
                            synth_df=synth_df,
                            missingness_model=preprocessed.missingness_model,
                            real_df=preprocessed.df_no_na,
                            disc_cols=preprocessed.disc_cols,
                            cont_cols=preprocessed.cont_cols,
                            backend_name=backend_name,
                        )
                    )

            compute_privacy_flag = _resolve_flag(
                self._cfg.compute_privacy,
                bundle.compute_privacy,
                spec.compute_privacy,
            )
            compute_downstream_flag = _resolve_flag(
                self._cfg.compute_downstream,
                bundle.compute_downstream,
                spec.compute_downstream,
            )

            if compute_privacy_flag:
                try:
                    self._metric_writer.write_privacy(
                        run_dir_path,
                        preprocessed.df_no_na,
                        preprocessed.inferred_types,
                        synth_df,
                        preprocessed.semmap_metadata,
                    )
                    logging.info("Wrote privacy metrics for %s", label)
                except ImportError:
                    logging.warning(
                        "synthcity not installed; skipping privacy metrics for %s", label
                    )
                except Exception:
                    logging.exception("Failed to compute privacy metrics for %s", label)

            target_series = preprocessed.color_series
            has_target_series = (
                isinstance(target_series, pd.Series)
                and target_series.attrs.get("prov:hadRole") == "target"
            )
            if compute_downstream_flag and has_target_series:
                try:
                    self._metric_writer.write_downstream(
                        run_dir_path,
                        preprocessed.df_no_na,
                        synth_df,
                        preprocessed.inferred_types,
                        target_series,
                        preprocessed.semmap_metadata,
                    )
                    logging.info("Wrote downstream metrics for %s", label)
                except ImportError:
                    logging.warning(
                        "statsmodels/sklearn not installed; skipping downstream metrics for %s",
                        label,
                    )
                except Exception:
                    logging.exception("Failed to compute downstream metrics for %s", label)


class ReportWriter:
    """Generate report outputs once backend runs are complete."""

    def __init__(self, reporting_module: Any, umap_utils: Optional[Any]) -> None:
        self._reporting = reporting_module
        self._umap_utils = umap_utils

    @rule()
    def generate_synthetic_umaps(
        self,
        model_runs: Iterable[Any],
        dataset_spec: "DatasetSpec",
        preprocessed: PreprocessingResult,
        cfg: PipelineConfig,
        marker: OutPath = OutPath("output/umaps.marker"),
    ) -> None:
        """Create UMAP visualisations for synthetic datasets if required.

        Args:
            model_runs: Iterable of discovered model run descriptors.
            dataset_spec: Dataset metadata describing the run.
            preprocessed: Preprocessed dataset artefacts.
            cfg: Pipeline configuration controlling thresholds.
        """

        if not self._umap_utils or not preprocessed.umap_artifacts:
            return

        for run in model_runs:
            try:
                if run.umap_png and run.umap_png.exists() and not cfg.overwrite_umap:
                    continue
                s_df = self._read_synthetic_df(run)
                if len(s_df) > cfg.max_umap_sample:
                    s_df = s_df.sample(cfg.max_umap_sample)
                s_df = s_df.reindex(columns=preprocessed.df_no_na.columns)
                s_emb = self._umap_utils.transform_with_umap(
                    preprocessed.umap_artifacts.transformer,
                    s_df.dropna(axis=0, how="any"),
                )
                run.umap_png = run.run_dir / "umap.png"
                self._umap_utils.plot_umap(
                    s_emb,
                    str(run.umap_png),
                    title=f"{dataset_spec.name}: synthetic ({run.name})",
                    lims=preprocessed.umap_artifacts.limits,
                )
            except Exception:
                logging.exception("Failed to generate UMAP for %s", run.run_dir)

        marker_path = marker.path if isinstance(marker, OutPath) else None
        if marker_path is not None:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.touch(exist_ok=True)

    @staticmethod
    def _read_synthetic_df(run: Any) -> "pd.DataFrame":
        import pandas as pd

        return pd.read_csv(run.synthetic_csv).convert_dtypes()

    @rule()
    def write_report(
        self,
        *,
        outdir: Path,
        dataset_spec: "DatasetSpec",
        preprocessed: PreprocessingResult,
        model_runs: List[Any],
        inferred_types: Optional[Dict[str, str]],
        variable_descriptions: Optional[Dict[str, Any]],
    ) -> None:
        """Persist the Markdown summary report.

        Args:
            outdir: Output directory for report artefacts.
            dataset_spec: Dataset metadata describing the run.
            preprocessed: Preprocessed dataset artefacts.
            model_runs: Collection of discovered model runs.
            inferred_types: Mapping of column names to inferred types.
            variable_descriptions: Optional variable description mapping.
        """

        try:
            from . import missingness as missingness_module
        except ImportError:
            missingness_summary = None
        else:
            missingness_summary = missingness_module.summarize_missingness_model(
                preprocessed.missingness_model
            )

        self._reporting.write_report_md(
            outdir=str(outdir),
            dataset_name=dataset_spec.name,
            dataset_provider=dataset_spec.provider,
            dataset_provider_id=dataset_spec.id,
            df=preprocessed.df_no_na,
            disc_cols=preprocessed.disc_cols,
            cont_cols=preprocessed.cont_cols,
            umap_png_real=str(preprocessed.umap_png_real)
            if preprocessed.umap_png_real
            else None,
            inferred_types=inferred_types or None,
            variable_descriptions=variable_descriptions or None,
            semmap_jsonld=preprocessed.semmap_export,
            dataset_metadata=preprocessed.semmap_metadata,
            model_runs=model_runs,
            missingness_summary=missingness_summary,
        )


@rule()
def process_dataset(
    dataset_spec: "DatasetSpec",
    df: "pd.DataFrame",
    color_series: Optional["pd.Series"],
    base_outdir: str,
    *,
    model_bundle: Optional[ModelConfigBundle] = None,
    pipeline_config: Optional[PipelineConfig] = None,
) -> None:
    """Process a dataset end-to-end and generate a report."""

    from . import semmap  # noqa: F401  # register pandas accessor

    utils = _import_utils()
    reporting = _import_reporting()

    bundle = model_bundle or load_model_configs(None)
    cfg = pipeline_config or PipelineConfig()

    outdir = Path(base_outdir) / dataset_spec.name.replace("/", "_")
    rng = utils.seed_all(cfg.random_state)

    generate_umap_flag = _resolve_flag(
        cfg.generate_umap,
        bundle.generate_umap,
        None,
    )

    umap_utils = _import_umap_utils() if generate_umap_flag else None

    preprocessor = DatasetPreprocessor(
        utils_module=utils,
        load_mapping=load_mapping_json,
        resolve_mapping=resolve_mapping_json,
    )
    preprocessed = preprocessor.preprocess(
        dataset_spec,
        df,
        color_series,
        outdir,
        cfg,
        rng,
        generate_umap=generate_umap_flag,
        umap_utils=umap_utils,
    )
    if preprocessed.semmap_metadata is not None:
        dataset_spec.meta = preprocessed.semmap_metadata

    privacy_summarizer = None
    downstream_compare = None
    try:
        privacy_summarizer = _import_privacy_summarizer()
    except ImportError:
        privacy_summarizer = None
    try:
        downstream_compare = _import_downstream_compare()
    except ImportError:
        downstream_compare = None

    metric_writer = MetricWriter(
        privacy_summarizer=privacy_summarizer,
        downstream_compare=downstream_compare,
    )
    executor = BackendExecutor(
        cfg,
        load_backend=_load_backend_module,
        metric_writer=metric_writer,
    )

    executor.run_models(dataset_spec, bundle, preprocessed, outdir)

    var_desc_map: Dict[str, Any] = {}
    if dataset_spec.provider == "uciml" and isinstance(dataset_spec.id, int):
        try:
            var_desc_map = get_uciml_variable_descriptions(dataset_spec.id)
        except Exception:
            var_desc_map = {}

    model_runs: List[Any] = []
    try:
        model_runs = discover_model_runs(outdir)
    except Exception:
        logging.exception("Failed to discover model runs for %s", dataset_spec.name)
        model_runs = []

    reporter = ReportWriter(reporting, umap_utils)
    reporter.generate_synthetic_umaps(model_runs, dataset_spec, preprocessed, cfg)
    reporter.write_report(
        outdir=outdir,
        dataset_spec=dataset_spec,
        preprocessed=preprocessed,
        model_runs=model_runs,
        inferred_types=preprocessed.inferred_types,
        variable_descriptions=var_desc_map,
    )

