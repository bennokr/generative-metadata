from datetime import datetime

# ------- tiny utilities -------
EMPTY = (None, "", [], {},)

def present(v):
    return v not in EMPTY

def put(d, k, v):
    """Set d[k]=v if v is present, return d (allows chaining)."""
    if present(v):
        d[k] = v
    return d

def add(d, k, v):
    """Append v to list d[k] if v is present."""
    if present(v):
        d.setdefault(k, [])
        d[k].append(v)
    return d

def prune(x):
    """Recursively drop empty fields/lists/dicts."""
    if isinstance(x, dict):
        return {k: prune(v) for k, v in x.items() if present(prune(v))}
    if isinstance(x, list):
        px = [prune(v) for v in x]
        return [v for v in px if present(v)]
    return x

def iso_date(s):
    try:
        return datetime.strptime(s, "%a %b %d %Y").date().isoformat()
    except Exception:
        return s

def stat_type(term: str):
    """Map source 'type' → DSV statistical data type class IRI."""
    t = (term or "").lower()
    if t == "categorical":
        return "dsv:CategoricalDataType"
    if t in {"integer", "real", "numeric", "number"}:
        return "dsv:NumericalDataType"  # refine to Ratio/Interval if you can infer it
    return "dsv:StatisticalDataType"

# ------- transform -------
def to_dcat_dsv(src: dict) -> dict:
    ctx = {
        "dcat": "http://www.w3.org/ns/dcat#",
        "dcterms": "http://purl.org/dc/terms/",
        "schema": "http://schema.org/",
        "dsv": "https://w3id.org/dsv-ontology#",
        "xsd": "http://www.w3.org/2001/XMLSchema#"
    }

    out = {
        "@context": ctx,
        "@type": "dcat:Dataset",
    }

    # Core DCAT / DCT
    put(out, "dcterms:title", src.get("name"))
    put(out, "dcterms:abstract", src.get("abstract"))
    put(out, "dcterms:created", src.get("year_of_dataset_creation"))
    put(out, "dcterms:modified", iso_date(src.get("last_updated") or ""))
    put(out, "dcat:landingPage", src.get("repository_url"))
    put(out, "dcat:theme", src.get("area"))

    # Identifiers (UCI id + DOI) as PropertyValue(s)
    ids = []
    add(ids := {"ids": []}, "ids", {"@type": "schema:PropertyValue", "schema:propertyID": "uci", "schema:value": src.get("uci_id")})
    add(ids, "ids", {"@type": "schema:PropertyValue", "schema:propertyID": "DOI", "schema:value": src.get("dataset_doi")})
    if ids.get("ids"):
        out["dcterms:identifier"] = ids["ids"]

    # Keywords: demographics + tasks + characteristics
    kws = (src.get("demographics") or []) + (src.get("tasks") or []) + (src.get("characteristics") or [])
    if kws:
        out["dcat:keyword"] = kws

    put(out, "schema:numberOfItems", src.get("num_instances"))
    if src.get("num_features") is not None:
        out["dcterms:extent"] = f'{src["num_features"]} features'

    # Creators
    creators = [{"@type": "schema:Person", "schema:name": n} for n in (src.get("creators") or []) if present(n)]
    if creators:
        out["dcterms:creator"] = creators

    # Distribution (CSV)
    if present(src.get("data_url")):
        out["dcat:distribution"] = [{
            "@type": "dcat:Distribution",
            "dcat:downloadURL": src["data_url"],
            "dcat:mediaType": "text/csv",
            "dcterms:format": "text/csv"
        }]

    # ----- DSV structural metadata -----
    dataset_schema = {"@type": "dsv:DatasetSchema"}
    columns = []

    for v in (src.get("variables") or []):
        col = {"@type": "dsv:Column"}
        put(col, "schema:name", v.get("name"))
        put(col, "dcterms:description", v.get("description"))
        # Units as plain text
        put(col, "schema:unitText", v.get("units"))
        # Role / demographic as lightweight annotations
        put(col, "prov:hadRole", v.get("role"))
        put(col, "schema:about", v.get("demographic"))

        # Per-column summary stats node, including statistical data type and missingness
        ss = {"@type": "dsv:SummaryStatistics"}
        put(ss, "dsv:statisticalDataType", stat_type(v.get("type")))
        if (v.get("missing_values") or "").lower() == "yes":
            put(ss, "dsv:missingValueFormat", src.get("missing_values_symbol"))
        if present(ss := prune(ss)):
            col["dsv:summaryStatistics"] = ss

        columns.append(prune(col))


    if columns:
        dataset_schema["dsv:column"] = columns
        out["dsv:datasetSchema"] = dataset_schema


    # Dataset-level summary statistics (rows/cols + missing value token)
    ds_ss = {"@type": "dsv:SummaryStatistics"}
    put(ds_ss, "dsv:numberOfRows", src.get("num_instances"))
    put(ds_ss, "dsv:numberOfColumns", src.get("num_features"))
    put(ds_ss, "dsv:missingValueFormat", src.get("missing_values_symbol"))
    if present(ds_ss := prune(ds_ss)):
        out["dsv:summaryStatistics"] = ds_ss

    # Citation for intro paper
    paper = src.get("intro_paper") or {}
    if paper:
        cit = {
            "@type": "schema:ScholarlyArticle",
            "dcterms:title": paper.get("title"),
            "schema:author": [a.strip() for a in (paper.get("authors") or "").split(",") if a.strip()],
            "schema:isPartOf": paper.get("venue") or paper.get("journal"),
            "schema:datePublished": paper.get("year"),
            "schema:url": paper.get("URL"),
        }
        ids2 = []
        if paper.get("DOI"):
            ids2.append({"@type": "schema:PropertyValue", "schema:propertyID": "DOI", "schema:value": paper["DOI"]})
        if paper.get("pmid"):
            ids2.append({"@type": "schema:PropertyValue", "schema:propertyID": "PMID", "schema:value": paper["pmid"]})
        if ids2:
            cit["schema:identifier"] = ids2
        out["schema:citation"] = prune(cit)

    # Additional info
    addi = src.get("additional_info") or {}
    put(out, "dcterms:description", addi.get("summary"))
    put(out, "dcterms:purpose", addi.get("purpose"))
    put(out, "schema:funding", addi.get("funded_by"))
    put(out, "schema:populationType", addi.get("instances_represent"))
    put(out, "dcterms:tableOfContents", addi.get("variable_info"))
    if present(addi.get("preprocessing_description")):
        out["prov:wasGeneratedBy"] = {"@type": "prov:Activity", "dcterms:description": addi["preprocessing_description"]}
    if addi.get("sensitive_data") is not None:
        out["dcterms:accessRights"] = addi.get("sensitive_data")

    return prune(out)

if __name__ == "__main__":
    import defopt, json
    from pathlib import Path

    def main(input_json: Path, output_json: Path = None, indent: int = 2):
        """
        Transform UCI ML style metadata json to DCAT+DSV JSON-LD.

        :param input_json: Path to source JSON file.
        :param output_json: Path to write result (default: stdout).
        :param indent: JSON indent.
        """
        src = json.loads(Path(input_json).read_text(encoding="utf-8"))
        out = to_dcat_dsv(src)
        if output_json:
            Path(output_json).write_text(json.dumps(out, indent=indent, ensure_ascii=False), encoding="utf-8")
        else:
            json.dump(out, sys.stdout, indent=indent, ensure_ascii=False)
            sys.stdout.write("\n")

    defopt.run(main)