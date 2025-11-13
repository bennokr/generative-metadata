import sys, json, hashlib, mimetypes, subprocess, logging, re
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from inspect import signature
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .jsonld import JSONLDMixin 

# ---- CLI type aliases for defopt ----
def InputPath(s: str) -> Path:   # annotate readable inputs
    return Path(s)

def OutputPath(s: str) -> Path:  # annotate writable outputs
    return Path(s)

# ---- small triple buffer for a named graph ----
class MiniGraph:
    def __init__(self):
        self.triples: List[tuple[str, str, Any]] = []

    def add(self, triple: tuple[str, str, Any]):
        s, p, o = triple
        self.triples.append((str(s), str(p), o if not isinstance(o, Path) else str(o)))

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _git(argv: list[str]) -> Optional[str]:
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None

def _file_iri(path: Path) -> str:
    return path.resolve().as_uri()

def _triples_to_nodes(triples: List[tuple[str, str, Any]]) -> List[dict]:
    nodes: Dict[str, dict] = {}
    for s, p, o in triples:
        node = nodes.setdefault(s, {"id": s})
        if p in ("a", "type"):
            tlist = node.setdefault("type", [])
            if isinstance(tlist, list):
                if o not in tlist:
                    tlist.append(o)
            else:
                node["type"] = [tlist, o] if tlist != o else tlist
            continue
        arr = node.setdefault(p, [])
        if isinstance(o, (int, float, bool)):
            arr.append(o)
        elif isinstance(o, str) and (o.startswith("http://") or o.startswith("https://") or o.startswith("urn:")):
            arr.append({"id": o})
        else:
            arr.append(o)
    return list(nodes.values())

# ---- compact JSON-LD context (override via Provenance(..., context=...)) ----
DEFAULT_CONTEXT = {
    "id": "@id",
    "type": "@type",
    "graph": "@graph",
    "prov": "http://www.w3.org/ns/prov#",
    "dct": "http://purl.org/dc/terms/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "generatedAtTime": "prov:generatedAtTime",
    "startedAtTime": "prov:startedAtTime",
    "endedAtTime": "prov:endedAtTime",
    "wasGeneratedBy": "prov:wasGeneratedBy",
    "wasAssociatedWith": "prov:wasAssociatedWith",
    "wasAttributedTo": "prov:wasAttributedTo",
    "wasDerivedFrom": "prov:wasDerivedFrom",
    "hadPrimarySource": "prov:hadPrimarySource",
    "used": "prov:used",
    "label": "rdfs:label",
    "modified": "dct:modified",
    "format": "dct:format",
    "extent": "dct:extent",
    "identifier": "dct:identifier",
    "hasVersion": "dct:hasVersion",
    "source": "dct:source",
}

# ---- JSON-LD dataclasses ----
@dataclass
class Activity(JSONLDMixin):
    id: str
    type: List[str] = field(default_factory=lambda: ["prov:Activity"])
    startedAtTime: Optional[str] = None
    endedAtTime: Optional[str] = None
    wasAssociatedWith: Optional[dict] = None
    used: List[dict] = field(default_factory=list)

@dataclass
class Agent(JSONLDMixin):
    id: str
    type: List[str] = field(default_factory=lambda: ["prov:SoftwareAgent"])
    label: Optional[str] = None

@dataclass
class Plan(JSONLDMixin):
    id: str
    type: List[str] = field(default_factory=lambda: ["prov:Plan"])
    label: Optional[str] = None
    hasVersion: Optional[str] = None
    source: Optional[dict] = None  # {"id": origin}

@dataclass
class FileEntity(JSONLDMixin):
    id: str
    type: List[str] = field(default_factory=lambda: ["prov:Entity"])
    format: Optional[str] = None
    extent: Optional[int] = None
    modified: Optional[str] = None
    identifier: Optional[str] = None
    wasGeneratedBy: Optional[dict] = None  # for outputs

@dataclass
class GraphEntity(JSONLDMixin):
    id: str
    type: List[str] = field(default_factory=lambda: ["prov:Entity"])
    wasGeneratedBy: Optional[dict] = None
    wasAttributedTo: Optional[dict] = None
    generatedAtTime: Optional[str] = None
    wasDerivedFrom: Optional[List[dict]] = None
    hadPrimarySource: Optional[List[dict]] = None
    graph: List[dict] = field(default_factory=list)  # named graph content

# ---- main Provenance manager ----
class Provenance:
    """Context manager + decorator emitting compact JSON-LD for a named graph and IO PROV."""
    def __init__(self, name: str, files: list, out_dir: str = "generated-rdf", context: Optional[dict] = None):
        self.name = name
        self.files = list(files)
        self.out_dir = Path(out_dir)
        self.context = context or DEFAULT_CONTEXT

        self.graph_id = f"urn:graph:{name}"
        self._gx = MiniGraph()

        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.script = Path(sys.argv[0]).resolve()
        self.activity_id = f"urn:run:{self.run_id}"
        self.agent_id = f"urn:agent:{self.script.name}"
        self.plan_id = f"urn:plan:{self.script.name}"

        self.commit = _git(["git", "rev-parse", "HEAD"])
        self.origin = _git(["git", "config", "--get", "remote.origin.url"])

        self.t0: Optional[datetime] = None
        self.t1: Optional[datetime] = None

    # context manager
    def __enter__(self):
        self.t0 = datetime.now(timezone.utc)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.t1 = datetime.now(timezone.utc)
        doc = self._build_document()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.out_dir / f"{self.name}.jsonld"
        logging.info(f"Writing {out_path}")
        with out_path.open("w", encoding="utf-8") as fw:
            json.dump(doc, fw, ensure_ascii=False, separators=(",", ":"))
        return False

    # decorator for defopt-style subcommands
    @classmethod
    def command(cls, name: str = None, out_dir: str = "generated-rdf", context: Optional[dict] = None):
        def decorator(fn):
            sig = signature(fn)
            ann = fn.__annotations__

            @wraps(fn)
            def wrapper(*args, **kwargs):
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()

                opened = []
                for pname, val in list(bound.arguments.items()):
                    typ = ann.get(pname)
                    if typ is InputPath and isinstance(val, Path):
                        fh = val.open("r", encoding="utf-8"); opened.append(fh)
                        bound.arguments[pname] = fh
                    elif typ is OutputPath and isinstance(val, Path):
                        val.parent.mkdir(parents=True, exist_ok=True)
                        fh = val.open("w", encoding="utf-8"); opened.append(fh)
                        bound.arguments[pname] = fh

                prov_name = name or fn.__name__
                with cls(prov_name, opened, out_dir=out_dir, context=context) as p:
                    if "prov" in sig.parameters and "prov" not in bound.arguments:
                        bound.arguments["prov"] = p
                    try:
                        return fn(*bound.args, **bound.kwargs)
                    finally:
                        for fh in opened:
                            try: fh.close()
                            except Exception: pass
            return wrapper
        return decorator

    # expose named-graph buffer
    @property
    def gx(self) -> MiniGraph:
        return self._gx

    # internals
    def _build_document(self) -> dict:
        inputs, outputs = self._classify_files()

        activity = Activity(
            id=self.activity_id,
            startedAtTime=self.t0.isoformat(),
            endedAtTime=self.t1.isoformat(),
            wasAssociatedWith={"id": self.agent_id},
            used=[{"id": self.plan_id}] + [{"id": _file_iri(p)} for p in inputs],
        )

        agent = Agent(id=self.agent_id, label=self.script.name)
        plan = Plan(
            id=self.plan_id,
            label=self.script.name,
            hasVersion=self.commit,
            source={"id": self.origin} if self.origin else None,
        )

        file_nodes: List[dict] = []
        for p, mode in self._file_modes().items():
            node = self._file_node(p, mode)
            if "w" in mode or "a" in mode:
                node.wasGeneratedBy = {"id": self.activity_id}
            file_nodes.append(node.to_jsonld(with_context=False))

        graph_node = GraphEntity(
            id=self.graph_id,
            wasGeneratedBy={"id": self.activity_id},
            wasAttributedTo={"id": self.agent_id},
            generatedAtTime=self.t1.isoformat(),
            wasDerivedFrom=[{"id": _file_iri(p)} for p in inputs] or None,
            hadPrimarySource=[{"id": _file_iri(p)} for p in inputs] or None,
            graph=_triples_to_nodes(self._gx.triples),
        )

        return {
            "@context": self.context,
            "@graph": [
                activity.to_jsonld(with_context=False),
                agent.to_jsonld(with_context=False),
                plan.to_jsonld(with_context=False),
                graph_node.to_jsonld(with_context=False),
                *file_nodes,
            ],
        }

    def _file_modes(self) -> Dict[Path, str]:
        m: Dict[Path, str] = {}
        for fh in self.files:
            try:
                m[Path(fh.name)] = getattr(fh, "mode", "")
            except Exception:
                continue
        return m

    def _classify_files(self):
        modes = self._file_modes()
        inputs = [p for p, mode in modes.items() if "r" in mode and "w" not in mode]
        outputs = [p for p, mode in modes.items() if "w" in mode or "a" in mode]
        return inputs, outputs

    def _file_node(self, path: Path, mode: str) -> FileEntity:
        mtype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        extent = path.stat().st_size if path.exists() else None
        modified = (
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            if path.exists() else None
        )
        identifier = None
        if "r" in mode and path.exists():
            try:
                identifier = f"sha256:{_sha256(path)}"
            except Exception:
                pass
        return FileEntity(
            id=_file_iri(path),
            format=mtype,
            extent=extent,
            modified=modified,
            identifier=identifier,
        )

# ---- optional add_dict utility (same signature as before) ----
def add_dict(g_or_ctx: MiniGraph, mapping: dict):
    for s, po in mapping.items():
        for p, o in po.items():
            objs = o if isinstance(o, (list, tuple, set)) else [o]
            for obj in objs:
                g_or_ctx.add((s, p, obj))
