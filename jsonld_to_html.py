
#!/usr/bin/env python3
"""
jsonld_to_html.py
Convert nested compact JSON-LD to static HTML using RDFa or Microdata.
"""

import argparse
import json
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

Value = Union[str, int, float, bool, dict, list, None]

SCHEMA_ORG = "https://schema.org/"

def is_absolute_iri(v: str) -> bool:
    return isinstance(v, str) and (v.startswith("http://") or v.startswith("https://"))

def ensure_trailing_slash(v: str) -> str:
    return v if not v or v.endswith(("/", "#")) else v + "/"

def extract_vocab_and_prefixes(context: Any) -> Tuple[str, Dict[str, str]]:
    vocab = ""
    prefixes: Dict[str, str] = {}
    if isinstance(context, str):
        if "schema.org" in context:
            vocab = SCHEMA_ORG
        elif is_absolute_iri(context):
            vocab = ensure_trailing_slash(context)
    elif isinstance(context, dict):
        v = context.get("@vocab")
        if isinstance(v, str):
            vocab = ensure_trailing_slash(v) if is_absolute_iri(v) else vocab
        for k, val in context.items():
            if k == "@vocab":
                continue
            if isinstance(val, str) and is_absolute_iri(val):
                prefixes[k] = ensure_trailing_slash(val)
    if not vocab:
        vocab = SCHEMA_ORG
    return vocab, prefixes

def to_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else [x]

def types_to_iris(types: Union[str, List[str]], vocab: str) -> List[str]:
    result = []
    for t in to_list(types):
        if is_absolute_iri(t):
            result.append(t)
        elif isinstance(t, str):
            if ":" in t:
                prefix, local = t.split(":", 1)
                if prefix == "schema":
                    result.append(ensure_trailing_slash(SCHEMA_ORG) + local)
                else:
                    result.append(t)
            else:
                result.append(ensure_trailing_slash(vocab) + t)
    return result

def attr_join(items: List[str]) -> str:
    return " ".join(i for i in items if i)

# ---------- Hyperlink helpers ----------

def render_uri_or_text_micro(prop: str, value: str) -> str:
    if is_absolute_iri(value):
        return f'<a itemprop="{html.escape(prop)}" href="{html.escape(value)}">{html.escape(value)}</a>'
    return f'<span itemprop="{html.escape(prop)}">{html.escape(value)}</span>'

def render_uri_or_text_rdfa(prop: str, value: str) -> str:
    if is_absolute_iri(value):
        return f'<a rel="{html.escape(prop)}" href="{html.escape(value)}">{html.escape(value)}</a>'
    return f'<span property="{html.escape(prop)}">{html.escape(value)}</span>'

# ---------- List structure analysis & table helpers ----------

def _is_repeating_dict_list(lst):
    if not isinstance(lst, list) or len(lst) < 2:
        return (False, [])
    dicts = [x for x in lst if isinstance(x, dict)]
    if len(dicts) < len(lst):
        return (False, [])
    key_sets = []
    for d in dicts:
        ks = tuple(sorted([k for k in d.keys() if not k.startswith("@")]))
        key_sets.append(set(ks))
    common = set.intersection(*key_sets) if key_sets else set()
    if not common:
        all_keys = set().union(*key_sets) if key_sets else set()
        common = set([k for k in all_keys if sum(1 for s in key_sets if k in s) >= max(2, len(key_sets)//2 + 1)])
    freq = {k: sum(1 for s in key_sets if k in s) for k in common}
    headers = sorted(common, key=lambda k: (-freq[k], k))
    if len(headers) == 0:
        return (False, [])
    return (True, headers[:10])

def _get_item_types(obj, vocab):
    return " ".join(html.escape(t) for t in types_to_iris(obj.get("@type", "Thing"), vocab))

def _render_cell_microdata(prop, value, vocab, depth):
    indent = "  " * depth
    frag = render_value_microdata(prop, value, vocab, depth)
    if 'class="prop"' in frag and frag.strip().startswith(f'{indent}<div'):
        inner = frag.split(":</span>", 1)[-1] if ":</span>" in frag else frag
        start = inner.find(">")
        if start != -1:
            inner = inner[start+1:]
        inner = inner.strip()
        if inner.endswith("</div>"):
            inner = inner[:-6]
        return inner
    return frag

def _render_list_table_microdata(prop, lst, vocab, depth):
    indent = "  " * depth
    ok, headers = _is_repeating_dict_list(lst)
    if not ok:
        return None
    rows = []
    thead = indent + "<thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead>"
    for obj in lst:
        types_attr = _get_item_types(obj, vocab)
        rid = obj.get("@id")
        id_attr = f' itemid="{html.escape(str(rid))}"' if rid else ""
        row = [f'{indent}<tr itemprop="{html.escape(prop)}" itemscope itemtype="{types_attr}"{id_attr}>']
        for h in headers:
            val = obj.get(h)
            cell = _render_cell_microdata(h, val, vocab, depth + 1) if h in obj else ""
            row.append(f'{indent}  <td>{cell}</td>')
        row.append(f"{indent}</tr>")
        rows.append("\n".join(row))
    table = [
        f'{indent}<div class="prop"><span class="name">{html.escape(prop)}</span>:</div>',
        f'{indent}<table class="prop-table" data-prop="{html.escape(prop)}">',
        thead,
        "\n".join(rows),
        f'{indent}</table>'
    ]
    return "\n".join(table)

def _render_cell_rdfa(prop, value, vocab, prefixes, depth):
    indent = "  " * depth
    frag = render_value_rdfa(prop, value, vocab, prefixes, depth)
    if 'class="prop"' in frag and frag.strip().startswith(f'{indent}<div'):
        inner = frag.split(":</span>", 1)[-1] if ":</span>" in frag else frag
        start = inner.find(">")
        if start != -1:
            inner = inner[start+1:]
        inner = inner.strip()
        if inner.endswith("</div>"):
            inner = inner[:-6]
        return inner
    return frag

def _render_list_table_rdfa(prop, lst, vocab, prefixes, depth):
    indent = "  " * depth
    ok, headers = _is_repeating_dict_list(lst)
    if not ok:
        return None
    rows = []
    thead = indent + "<thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead>"
    for obj in lst:
        typeof = " ".join(to_list(obj.get("@type", "Thing")))
        about = obj.get("@id")
        typeof_attr = f' typeof="{html.escape(typeof)}"'
        about_attr = f' resource="{html.escape(str(about))}"' if about else ""
        row = [f'{indent}<tr property="{html.escape(prop)}"{typeof_attr}{about_attr}>']
        for h in headers:
            val = obj.get(h)
            cell = _render_cell_rdfa(h, val, vocab, prefixes, depth + 1) if h in obj else ""
            row.append(f'{indent}  <td>{cell}</td>')
        row.append(f"{indent}</tr>")
        rows.append("\n".join(row))
    table = [
        f'{indent}<div class="prop"><span class="name">{html.escape(prop)}</span>:</div>',
        f'{indent}<table class="prop-table" data-prop="{html.escape(prop)}">',
        thead,
        "\n".join(rows),
        f'{indent}</table>'
    ]
    return "\n".join(table)

# ---------- Microdata rendering ----------

def render_value_microdata(prop: str, value: Value, vocab: str, depth: int) -> str:
    indent = "  " * depth
    if isinstance(value, dict):
        return (f'{indent}<div itemprop="{html.escape(prop)}" itemscope itemtype="{" ".join(html.escape(t) for t in types_to_iris(value.get("@type", "Thing"), vocab))}">\n'
                f'{render_object_microdata(value, vocab, depth + 1)}\n'
                f'{indent}</div>')
    elif isinstance(value, list):
        table = _render_list_table_microdata(prop, value, vocab, depth)
        if table is not None:
            return table
        parts = [render_value_microdata(prop, v, vocab, depth) for v in value]
        return "\n".join(parts)
    else:
        if value is None:
            return ""
        if isinstance(value, bool):
            content = "true" if value else "false"
        else:
            content = str(value)
        return f'{indent}<div class="prop"><span class="name">{html.escape(prop)}</span>: ' + render_uri_or_text_micro(prop, content) + '</div>'

def render_object_microdata(obj: Dict[str, Any], vocab: str, depth: int = 0) -> str:
    indent = "  " * depth
    lines: List[str] = []
    title = obj.get("name") or obj.get("headline") or obj.get("@type")
    if isinstance(title, (str, int, float)):
        lines.append(f'{indent}<h2 class="item-title">{html.escape(str(title))}</h2>')
    rid = obj.get("@id")
    if isinstance(rid, str) and is_absolute_iri(rid):
        lines.append(f'{indent}<div class="id-link"><a href="{html.escape(rid)}">{html.escape(rid)}</a></div>')
    for k, v in obj.items():
        if k.startswith("@"):
            continue
        lines.append(render_value_microdata(k, v, vocab, depth))
    return "\n".join(l for l in lines if l)

def render_microdata(root: Value, context: Any) -> str:
    vocab, _ = extract_vocab_and_prefixes(context)
    body_parts: List[str] = []
    if isinstance(root, list):
        for item in root:
            if isinstance(item, dict):
                types = item.get("@type", "Thing")
                itemid = item.get("@id")
                type_iris = " ".join(html.escape(t) for t in types_to_iris(types, vocab))
                id_attr = f' itemid="{html.escape(str(itemid))}"' if itemid else ""
                body_parts.append(f'<div class="item" itemscope itemtype="{type_iris}"{id_attr}>\n{render_object_microdata(item, vocab, 1)}\n</div>')
    elif isinstance(root, dict):
        types = root.get("@type", "Thing")
        itemid = root.get("@id")
        type_iris = " ".join(html.escape(t) for t in types_to_iris(types, vocab))
        id_attr = f' itemid="{html.escape(str(itemid))}"' if itemid else ""
        body_parts.append(f'<div class="item" itemscope itemtype="{type_iris}"{id_attr}>\n{render_object_microdata(root, vocab, 1)}\n</div>')
    else:
        body_parts.append(f"<pre>{html.escape(str(root))}</pre>")
    return "\n".join(body_parts)

# ---------- RDFa rendering ----------

def render_value_rdfa(prop: str, value: Value, vocab: str, prefixes: Dict[str, str], depth: int) -> str:
    indent = "  " * depth
    prop_attr = f' property="{html.escape(prop)}"'
    if isinstance(value, dict):
        typeof = value.get("@type", "Thing")
        typeof_attr = f' typeof="{html.escape(" ".join(to_list(typeof)))}"'
        about = value.get("@id")
        about_attr = f' resource="{html.escape(str(about))}"' if about else ""
        return (f'{indent}<div{prop_attr}{typeof_attr}{about_attr}>\n'
                f'{render_object_rdfa(value, vocab, prefixes, depth + 1)}\n'
                f'{indent}</div>')
    elif isinstance(value, list):
        table = _render_list_table_rdfa(prop, value, vocab, prefixes, depth)
        if table is not None:
            return table
        return "\n".join(render_value_rdfa(prop, v, vocab, prefixes, depth) for v in value)
    else:
        if value is None:
            return ""
        if isinstance(value, bool):
            content = "true" if value else "false"
        else:
            content = str(value)
        return f'{indent}<div class="prop"><span class="name">{html.escape(prop)}</span>:' + render_uri_or_text_rdfa(prop, content) + '</div>'

def render_object_rdfa(obj: Dict[str, Any], vocab: str, prefixes: Dict[str, str], depth: int = 0) -> str:
    indent = "  " * depth
    lines: List[str] = []
    title = obj.get("name") or obj.get("headline") or obj.get("@type")
    if isinstance(title, (str, int, float)):
        lines.append(f'{indent}<h2 class="item-title">{html.escape(str(title))}</h2>')
    rid = obj.get("@id")
    if isinstance(rid, str) and is_absolute_iri(rid):
        lines.append(f'{indent}<div class="id-link"><a href="{html.escape(rid)}">{html.escape(rid)}</a></div>')
    for k, v in obj.items():
        if k.startswith("@"):
            continue
        lines.append(render_value_rdfa(k, v, vocab, prefixes, depth))
    return "\n".join(l for l in lines if l)

def render_rdfa(root: Value, context: Any) -> str:
    vocab, prefixes = extract_vocab_and_prefixes(context)
    prefix_attr = " ".join(f'{p}: {iri}' for p, iri in prefixes.items())
    prefix_html = f' prefix="{html.escape(prefix_attr)}"' if prefix_attr else ""
    body_parts: List[str] = []

    if isinstance(root, list):
        for item in root:
            if isinstance(item, dict):
                typeof = item.get("@type", "Thing")
                about = item.get("@id")
                typeof_attr = f' typeof="{html.escape(" ".join(to_list(typeof)))}"'
                about_attr = f' about="{html.escape(str(about))}"' if about else ""
                body_parts.append(f'<div class="item" vocab="{html.escape(vocab)}"{prefix_html}{typeof_attr}{about_attr}>\n{render_object_rdfa(item, vocab, prefixes, 1)}\n</div>')
    elif isinstance(root, dict):
        typeof = root.get("@type", "Thing")
        about = root.get("@id")
        typeof_attr = f' typeof="{html.escape(" ".join(to_list(typeof)))}"'
        about_attr = f' about="{html.escape(str(about))}"' if about else ""
        body_parts.append(f'<div class="item" vocab="{html.escape(vocab)}"{prefix_html}{typeof_attr}{about_attr}>\n{render_object_rdfa(root, vocab, prefixes, 1)}\n</div>')
    else:
        body_parts.append(f"<pre>{html.escape(str(root))}</pre>")

    return "\n".join(body_parts)

# ---------- HTML shell ----------

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
    code, pre {{ background: #f6f8fa; padding: .25rem .4rem; border-radius: 4px; }}
    table.prop-table {{ border-collapse: collapse; width: 100%; margin: .25rem 0 1rem; }}
    table.prop-table th, table.prop-table td {{ border: 1px solid #ddd; padding: .35rem .5rem; text-align: left; }}
    table.prop-table th {{ background: #f3f3f3; }}
  </style>
</head>
<body>
<h1>{title}</h1>
"""

HTML_FOOT = """
</body>
</html>
"""

def wrap_html(body: str, title: str) -> str:
    return HTML_HEAD.format(title=html.escape(title)) + body + HTML_FOOT

def main():
    parser = argparse.ArgumentParser(description="Convert compact JSON-LD to HTML (RDFa or Microdata).")
    parser.add_argument("input", type=Path, help="Path to JSON(-LD) file")
    parser.add_argument("--out", type=Path, default=Path("out.html"), help="Output HTML file")
    parser.add_argument("--format", choices=["microdata", "rdfa"], default="microdata", help="Output annotation style")
    parser.add_argument("--title", default="JSON-LD to HTML", help="HTML <title>")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    context = None
    if isinstance(data, dict):
        context = data.get("@context")
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        context = data[0].get("@context")
    if context is None:
        context = SCHEMA_ORG

    if args.format == "microdata":
        body = render_microdata(data, context)
    else:
        body = render_rdfa(data, context)

    html_out = wrap_html(body, args.title)
    args.out.write_text(html_out, encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
