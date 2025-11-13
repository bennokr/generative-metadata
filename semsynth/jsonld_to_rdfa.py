#!/usr/bin/env python3
"""
jsonld_to_rdfa.py
Generate static HTML with RDFa from nested compact JSON-LD.

Features:
- Supports compact JSON-LD with @context, @type, @id.
- Infers @vocab and prefixes; defaults to schema.org.
- Hyperlinks all absolute IRIs (including @id and property values).
- Recursively renders nested objects and arrays.
- Detects repeating lists of similar objects and renders them as tables.
- No third-party dependencies.
"""

from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

Value = Union[str, int, float, bool, dict, list, None]
SCHEMA_ORG = "https://schema.org/"

# ---------------- Context & IRI helpers ----------------


def is_iri(v: Any) -> bool:
    return isinstance(v, str) and (v.startswith("http://") or v.startswith("https://"))


def ensure_ns(v: str) -> str:
    return v if v.endswith(("/", "#")) else v + "/"


def extract_vocab_and_prefixes(context: Any) -> Tuple[str, Dict[str, str]]:
    vocab = ""
    prefixes: Dict[str, str] = {}
    if isinstance(context, str) and is_iri(context):
        vocab = SCHEMA_ORG if "schema.org" in context else ensure_ns(context)
    elif isinstance(context, dict):
        voc = context.get("@vocab")
        if isinstance(voc, str) and is_iri(voc):
            vocab = ensure_ns(voc)
        for k, v in context.items():
            if k == "@vocab":
                continue
            if isinstance(v, str) and is_iri(v):
                prefixes[k] = ensure_ns(v)
    if not vocab:
        vocab = SCHEMA_ORG
    return vocab, prefixes


def to_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else [x]


def types_to_iris(types: Union[str, List[str]], vocab: str) -> List[str]:
    out: List[str] = []
    for t in to_list(types):
        if is_iri(t):
            out.append(t)
        elif isinstance(t, str):
            if ":" in t:
                prefix, local = t.split(":", 1)
                if prefix == "schema":
                    out.append(ensure_ns(SCHEMA_ORG) + local)
                else:
                    out.append(t)  # leave compact as-is
            else:
                out.append(ensure_ns(vocab) + t)
    return out


# ---------------- HTML builders ----------------


def tag(
    name: str,
    attrs: Dict[str, Optional[str]] | None = None,
    content: str = "",
    self_close: bool = False,
) -> str:
    """Minimal HTML tag builder. Escapes attribute values; content is trusted to be already-escaped as needed."""
    attrs = attrs or {}
    parts = [name]
    for k, v in attrs.items():
        if v is None or v is False:
            continue
        parts.append(f'{k}="{html.escape(str(v), quote=True)}"')
    if self_close and not content:
        return "<" + " ".join(parts) + " />"
    return "<" + " ".join(parts) + ">" + content + f"</{name}>"


# ---------------- Repeating-structure detection ----------------


def repeating_headers(lst: List[dict]) -> List[str]:
    """Return a stable list of shared keys for a list of dicts (ignoring @-keys)."""
    if len(lst) < 2:
        return []
    key_sets = [tuple(k for k in d.keys() if not k.startswith("@")) for d in lst]
    if not key_sets:
        return []
    allk = set().union(*map(set, key_sets))
    # Try to find keyset that contains all headers
    for ks in key_sets:
        if all([k in ks for k in allk]):
            return ks
    # Otherwise, stable sort by frequency desc then name
    freq = {k: sum(1 for s in key_sets if k in s) for k in allk}
    return sorted(allk, key=lambda k: (-freq[k], k))[:12]


# ---------------- Rendering (RDFa only) ----------------


def render_literal(prop: str, value: Union[str, int, float, bool]) -> str:
    text = "true" if value is True else "false" if value is False else str(value)
    return tag("span", {"property": prop}, html.escape(text).replace("\n", "<br>"))


def render_iri(prop: str, iri: str) -> str:
    return tag("a", {"rel": prop, "href": iri}, html.escape(iri))


def render_property(
    prop: str, value: Value, vocab: str, prefixes: Dict[str, str], depth: int
) -> str:
    """Render a property value in RDFa. Handles dict, list, and literals. Recursive."""
    if value is None:
        return ""
    # Object
    if isinstance(value, dict):
        typeof = " ".join(to_list(value.get("@type", "Thing")))
        about = value.get("@id")
        inner = render_object(value, vocab, prefixes, depth + 1)
        attrs = {"property": prop, "typeof": typeof}
        if isinstance(about, str):
            attrs["resource"] = about
        return tag("div", attrs, inner)
    # List
    if isinstance(value, list):
        dicts = [v for v in value if isinstance(v, dict)]
        if len(dicts) == len(value) and len(value) >= 2:
            # Try table
            headers = repeating_headers(dicts)
            if headers:
                rows: List[str] = []
                thead = tag(
                    "thead",
                    {},
                    tag(
                        "tr",
                        {},
                        "".join(tag("th", {}, html.escape(h)) for h in headers),
                    ),
                )
                for obj in value:  # type: ignore
                    typeof = " ".join(to_list(obj.get("@type", "Thing")))
                    resource = obj.get("@id")
                    row_cells: List[str] = []
                    for h in headers:
                        cell_val = obj.get(h)
                        cell_html = render_cell(h, cell_val, vocab, prefixes, depth + 2)
                        row_cells.append(tag("td", {}, cell_html))
                    tr_attrs = {"property": prop, "typeof": typeof}
                    if isinstance(resource, str):
                        tr_attrs["resource"] = resource
                    rows.append(tag("tr", tr_attrs, "".join(row_cells)))
                table = tag(
                    "table",
                    {"class": "prop-table", "data-prop": prop},
                    thead + "".join(rows),
                )
                wrapper = tag(
                    "div",
                    {"class": "prop"},
                    tag("span", {"class": "name"}, html.escape(prop)),
                )
                return wrapper + table
        # Fallback: render each item
        return "\n".join(
            render_property(prop, v, vocab, prefixes, depth) for v in value
        )
    # Literal or IRI
    if isinstance(value, str) and is_iri(value):
        return tag(
            "div",
            {"class": "prop"},
            tag("span", {"class": "name"}, html.escape(prop)) + render_iri(prop, value),
        )
    if isinstance(value, (str, int, float, bool)):
        return tag(
            "div",
            {"class": "prop"},
            tag("span", {"class": "name"}, html.escape(prop))
            + render_literal(prop, value),
        )
    return ""


def render_cell(
    prop: str, value: Value, vocab: str, prefixes: Dict[str, str], depth: int
) -> str:
    """Cell-friendly rendering (no leading label)."""
    if value is None:
        return ""
    if isinstance(value, dict):
        typeof = " ".join(to_list(value.get("@type", "Thing")))
        resource = value.get("@id")
        inner = render_object(value, vocab, prefixes, depth + 1)
        attrs = {"property": prop, "typeof": typeof}
        if isinstance(resource, str):
            attrs["resource"] = resource
        return tag("div", attrs, inner)
    if isinstance(value, list):
        return "\n".join(
            render_property(prop, v, vocab, prefixes, depth) for v in value
        )
    if isinstance(value, str) and is_iri(value):
        return render_iri(prop, value)
    if isinstance(value, (str, int, float, bool)):
        return render_literal(prop, value)
    return ""


def render_object(
    obj: Dict[str, Any], vocab: str, prefixes: Dict[str, str], depth: int
) -> str:
    parts: List[str] = []
    # Visible title (optional)
    title = obj.get("name") or obj.get("headline") or obj.get("@type")
    if isinstance(title, (str, int, float)):
        parts.append(tag("h2", {"class": "item-title"}, html.escape(str(title))))
    # Visible @id link
    rid = obj.get("@id")
    if isinstance(rid, str) and is_iri(rid):
        parts.append(
            tag("div", {"class": "id-link"}, tag("a", {"href": rid}, html.escape(rid)))
        )
    # Properties
    for k, v in obj.items():
        if k.startswith("@"):
            continue
        parts.append(render_property(k, v, vocab, prefixes, depth + 1))
    return "\n".join(p for p in parts if p)


def render_rdfa(root: Value, context: Any, title: str) -> str:
    vocab, prefixes = extract_vocab_and_prefixes(context)
    prefix_attr = " ".join(f"{p}: {iri}" for p, iri in prefixes.items())
    prefix_html = {"prefix": prefix_attr} if prefix_attr else {}
    items: List[str] = []

    def wrap_item(item: Dict[str, Any]) -> str:
        typeof = " ".join(to_list(item.get("@type", "Thing")))
        about = item.get("@id")
        attrs = {"class": "item", "vocab": vocab, "typeof": typeof}
        attrs.update(prefix_html)
        if isinstance(about, str):
            attrs["about"] = about
        return tag("div", attrs, render_object(item, vocab, prefixes, 1))

    if isinstance(root, list):
        items.extend(wrap_item(x) for x in root if isinstance(x, dict))
    elif isinstance(root, dict):
        items.append(wrap_item(root))
    else:
        items.append(tag("pre", {}, html.escape(str(root))))
    return HTML_HEAD.format(title=html.escape(title)) + "\n".join(items) + HTML_FOOT


# ---------------- HTML shell ----------------

HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, Arial, sans-serif; line-height: 1.5; margin: 2rem; }}
    .item {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
    .item-title {{ margin: 0 0 .5rem; font-size: 1.1rem; }}
    .prop {{ margin: .2rem 0; }}
    .name {{ font-weight: 600; margin-right: .25rem; }}
    .id-link {{ margin: .2rem 0; }}
    table.prop-table {{ border-collapse: collapse; margin: .25rem 1rem 1rem; }}
    table.prop-table th, table.prop-table td {{ border: 1px solid #ddd; padding: .35rem .5rem; text-align: left; vertical-align: top; }}
    table.prop-table th {{ background: #f3f3f3; }}
    div[property] {{ margin-left: 1rem; }}
    div[property]:before {{ content: attr(property); font-weight:bold; margin-left: -1rem; }}
  </style>
</head>
<body>
<h1>{title}</h1>
"""
HTML_FOOT = """
</body>
</html>
"""

# ---------------- CLI ----------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert compact JSON-LD to RDFa-annotated HTML."
    )
    ap.add_argument("input", type=Path, help="Path to JSON(-LD) file")
    ap.add_argument(
        "--out", type=Path, default=Path("out.html"), help="Output HTML file"
    )
    ap.add_argument("--title", default="JSON-LD to RDFa", help="HTML <title>")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    context = None
    if isinstance(data, dict):
        context = data.get("@context")
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        context = data[0].get("@context")
    if context is None:
        context = SCHEMA_ORG

    html_out = render_rdfa(data, context, args.title)
    args.out.write_text(html_out, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
