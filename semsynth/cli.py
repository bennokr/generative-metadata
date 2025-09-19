from __future__ import annotations

import sys
from typing import List, Optional
import logging
import json

import defopt

from .datasets import (
    DatasetSpec,
    list_openml,
    list_uciml,
    load_dataset,
    specs_from_input,
)
from .synth import run_from_yaml, run_synth_experiment
from .models import load_model_configs, model_run_root
from .backends import pybnesian as backend_pyb, synthcity as backend_syn
from .utils import ensure_dir


def search(provider: str, *, name_substr: Optional[str] = None, area: str = "Health and Medicine", cat_min: int = 1, num_min: int = 1, verbose: bool = False) -> None:
    """Search mixed-type datasets on OpenML or UCI ML Repo.

    Parameters:
        provider: 'openml' or 'uciml'.
        name_substr: Optional substring to filter dataset names (case-insensitive).
        area: For 'uciml', the Area to search (default 'Health and Medicine'). Ignored for 'openml'.
        cat_min: Minimum number of categorical columns
        num_min: Minimum number of numeric columns
        verbose: Print logging
    """
    if verbose:
        logging.root.setLevel(logging.INFO)

    p = provider.lower()
    if p == "openml":
        df = list_openml(name_substr=name_substr, cat_min=cat_min, num_min=num_min)
    elif p == "uciml":
        df = list_uciml(area=area, name_substr=name_substr, cat_min=cat_min, num_min=num_min)
    else:
        raise SystemExit("provider must be 'openml' or 'uciml'")
    print(df.to_csv(sep='\t', index=None))




def synth(
    dataset: str,
    *,
    provider: str = 'openml',
    backend: str = 'synthcity',
    generator: str = 'ctgan',
    gen_params_json: str = '',
    rows: Optional[int] = None,
    seed: int = 0,
    outdir: str = 'outputs',
    test_size: float = 0.25,
    configs_yaml: str = '',
    verbose: bool = False,
) -> None:
    """Fit-generate-evaluate a single model and write artifacts under the dataset report.

    Parameters:
        dataset: Dataset identifier; for 'openml' use a name; for 'uciml' use a numeric ID string.
        provider: 'openml' or 'uciml'.
        backend: 'synthcity' (default) or 'pybnesian'.
        generator: For synthcity, the plugin alias (e.g., 'ctgan', 'tvae'). For pybnesian, the BN type ('clg' or 'semiparametric').
        gen_params_json: JSON string with parameters. For synthcity, passed to plugin. For pybnesian, may include score, operators, max_indegree.
        rows: Number of synthetic rows to generate (defaults to train size if omitted).
        seed: Random seed.
        outdir: Root output directory (per-dataset subfolder is created).
        test_size: Fraction for test split.
        configs_yaml: Optional YAML file:
            - If backend='synthcity', supports a list under 'generators' (legacy) for multiple synthcity runs.
        verbose: Enable logging.
    """
    if verbose:
        logging.root.setLevel(logging.INFO)

    ensure_dir(outdir)
    specs = specs_from_input(provider=provider, datasets=[dataset])
    if not specs:
        raise SystemExit(f'No dataset found for provider={provider!r} input={dataset!r}')
    spec = specs[0]
    meta, df, _color = load_dataset(spec)
    dataset_display_name = getattr(meta, 'name', None)
    if not dataset_display_name and isinstance(meta, dict):
        dataset_display_name = meta.get('name')
    if not dataset_display_name:
        dataset_display_name = spec.name or str(spec.id or dataset)
    dataset_display_name = str(dataset_display_name)

    provider_id = None
    if spec.provider == 'uciml':
        provider_id = spec.id
    elif spec.provider == 'openml':
        for attr in ('dataset_id', 'did', 'id'):
            if hasattr(meta, attr):
                try:
                    provider_id = int(getattr(meta, attr))
                    break
                except Exception:
                    continue

    cfg_path = configs_yaml.strip()
    if cfg_path:
        # Unified configs for both backends
        dataset_dir = Path(outdir) / dataset_display_name
        ensure_dir(str(dataset_dir))
        try:
            specs = load_model_configs(cfg_path)
        except Exception as exc:
            raise SystemExit(str(exc))
        for ms in specs:
            b = (ms.backend or 'pybnesian').lower()
            if b == 'synthcity':
                mname = (ms.model or {}).get('name')
                params = (ms.model or {}).get('params', {})
                backend_syn.run_experiment(
                    df=df,
                    provider=spec.provider,
                    dataset_name=dataset_display_name,
                    provider_id=provider_id,
                    outdir=str(dataset_dir),
                    label=ms.name,
                    model_name=str(mname),
                    params=params,
                    rows=ms.rows,
                    seed=int(ms.seed or seed),
                    test_size=test_size,
                )
            elif b == 'pybnesian':
                backend_pyb.run_experiment(
                    df=df,
                    provider=spec.provider,
                    dataset_name=dataset_display_name,
                    provider_id=provider_id,
                    outdir=str(dataset_dir),
                    label=ms.name,
                    model_info=ms.model or {},
                    rows=ms.rows,
                    seed=int(ms.seed or seed),
                    test_size=test_size,
                )
            else:
                raise SystemExit(f"Unknown backend in config: {ms.backend}")
        return

    dataset_dir = Path(outdir) / dataset_display_name
    ensure_dir(str(dataset_dir))
    if backend.lower() == 'synthcity':
        backend_syn.run_experiment(
            df=df,
            provider=spec.provider,
            dataset_name=dataset_display_name,
            provider_id=provider_id,
            outdir=str(dataset_dir),
            label=generator,
            model_name=generator,
            params=(json.loads(gen_params_json) if gen_params_json else {}),
            rows=rows,
            seed=seed,
            test_size=test_size,
        )
    elif backend.lower() == 'pybnesian':
        backend_pyb.run_experiment(
            df=df,
            provider=spec.provider,
            dataset_name=dataset_display_name,
            provider_id=provider_id,
            outdir=str(dataset_dir),
            label=generator,
            model_info=(json.loads(gen_params_json) if gen_params_json else {"type": generator}),
            rows=rows,
            seed=seed,
            test_size=test_size,
        )
    else:
        raise SystemExit("backend must be 'synthcity' or 'pybnesian'")


def report(
    provider: str = "openml",
    *,
    datasets: List[str] = [],
    outdir: str = "docs",
    bn_types: List[str] = ("clg", "semiparametric"),
    configs_yaml: str = "",
    roots: List[str] = (),
    area: str = "Health and Medicine",
    metasyn: bool = True,
    verbose: bool = False
) -> None:
    """Run the BN report pipeline on a collection of datasets.

    Parameters:
        provider: 'openml' or 'uciml'.
        datasets: For 'openml', dataset names; for 'uciml', dataset IDs as strings. If omitted, defaults are used.
        outdir: Output directory where per-dataset reports are written.
        bn_types: One or more BN types to learn and compare. Supported: 'clg', 'semiparametric'. Ignored if configs_yaml is provided.
        configs_yaml: Optional YAML file defining a list of BN structure-learning configurations. Each item can include: name, bn_type, score, operators, max_indegree, seed. If provided, these override bn_types.
        roots: Optional list of variable names treated as roots; structure learning forbids arcs from these variables to others. If omitted, defaults to ['age','sex','race'] or for UCI ML, the 'demographics' metadata if present.
        area: For default UCI selection, which Area to pull mixed-dtype datasets from.
    """
    if verbose:
        logging.root.setLevel(logging.INFO)

    ensure_dir(outdir)
    from .pipeline import process_dataset  # defer heavy imports

    ds = datasets if datasets else None
    specs: List[DatasetSpec] = specs_from_input(provider=provider, datasets=ds, area=area)
    # Load unified model configs (pybnesian + synthcity). If not provided, load default_config.yaml
    cfg_path = configs_yaml.strip() or None
    try:
        model_specs = load_model_configs(cfg_path)
    except Exception as exc:
        raise SystemExit(str(exc))
    for spec in specs:
        logging.info(f'Loading {spec}')
        try:
            meta, df, color = load_dataset(spec)
            # Pass roots only if user provided it; otherwise None to use provider-specific default
            abl = list(roots) if isinstance(roots, (list, tuple)) and len(roots) else None
            # Try to extract provider_id for linking
            prov_id = None
            if spec.provider == 'uciml':
                prov_id = spec.id
            elif spec.provider == 'openml':
                for attr in ('dataset_id', 'did', 'id'):
                    if hasattr(meta, attr):
                        try:
                            prov_id = int(getattr(meta, attr))
                            break
                        except Exception:
                            pass
            process_dataset(
                meta,
                df,
                color,
                outdir,
                provider=spec.provider,
                provider_id=prov_id,
                model_configs=model_specs,
                roots=abl,
                run_metasyn=metasyn,
            )
        except Exception as e:
            print(f"[WARN] Skipped {spec.provider}:{spec.name} due to error: {e}")
            raise


def main(argv: Optional[List[str]] = None) -> None:
    defopt.run([search, report, synth], argv=argv)


if __name__ == "__main__":
    main(sys.argv[1:])
