"""
Fetch UniProt function descriptions and GO Biological Process terms
for neurotransmitter system genes (dopamine, serotonin, oxytocin).

Uses the UniProt REST API to query human (organism_id:9606) proteins.
"""

import requests
import time
import csv
import sys
import re

OUTPUT_FILE = "/home/jw3514/Work/ASD_Circuits_CellType/dat/NeuralSystem_annotations.tsv"

# 27 genes: 14 dopamine + 11 serotonin + 2 oxytocin (no PCSK1, no acetylcholine)
GENES = [
    # Dopamine
    "TH", "DDC", "GCH1", "DRD1", "DRD2", "DRD3", "DRD4", "DRD5",
    "SLC6A3", "SLC22A3", "COMT", "MAOA", "MAOB", "SLC18A2",
    # Serotonin
    "TPH2", "HTR1A", "HTR1B", "HTR2A", "HTR2C", "HTR3A", "HTR4",
    "HTR5A", "HTR6", "HTR7", "SLC6A4",
    # Oxytocin
    "OXT", "OXTR",
]

# Map genes to their neurotransmitter system for filtering GO terms
GENE_SYSTEM = {}
for g in ["TH", "DDC", "GCH1", "DRD1", "DRD2", "DRD3", "DRD4", "DRD5",
           "SLC6A3", "SLC22A3", "COMT", "MAOA", "MAOB", "SLC18A2"]:
    GENE_SYSTEM[g] = "dopamine"
for g in ["TPH2", "HTR1A", "HTR1B", "HTR2A", "HTR2C", "HTR3A", "HTR4",
           "HTR5A", "HTR6", "HTR7", "SLC6A4"]:
    GENE_SYSTEM[g] = "serotonin"
for g in ["OXT", "OXTR"]:
    GENE_SYSTEM[g] = "oxytocin"

# Keywords to match relevant GO BP terms per system
SYSTEM_KEYWORDS = {
    "dopamine": ["dopamin", "catecholamin", "monoamin", "tyrosine hydroxyl",
                 "reward", "locomotor", "motor behav"],
    "serotonin": ["serotonin", "5-hydroxytryptamin", "tryptophan", "monoamin"],
    "oxytocin": ["oxytocin", "neuropeptide", "social behav", "parturition",
                 "lactation", "uterine"],
}

# Additional general NT-relevant keywords that apply to all systems
GENERAL_KEYWORDS = [
    "neurotransmitt", "synaptic", "signal transduction", "G protein-coupled",
    "adenylate cyclase", "calcium", "cAMP", "transport", "vesicl",
    "biosynthetic process", "metabolic process", "catabolic process",
    "receptor signal", "ion channel", "membrane potential",
]

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"


def fetch_gene_annotation(gene_symbol):
    """Fetch UniProt function and GO BP terms for a human gene."""
    params = {
        "query": f"gene_exact:{gene_symbol} AND organism_id:9606",
        "fields": "gene_names,protein_name,cc_function,go_p",
        "format": "tsv",
        "size": "1",  # Take the top reviewed entry
    }
    # Prefer reviewed (Swiss-Prot) entries
    params_reviewed = dict(params)
    params_reviewed["query"] = f"gene_exact:{gene_symbol} AND organism_id:9606 AND reviewed:true"

    headers = {"Accept": "text/plain"}

    # Try reviewed first
    resp = requests.get(BASE_URL, params=params_reviewed, headers=headers, timeout=30)
    resp.raise_for_status()

    lines = resp.text.strip().split("\n")
    if len(lines) < 2:
        # Fall back to unreviewed
        resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")

    if len(lines) < 2:
        print(f"  WARNING: No UniProt entry found for {gene_symbol}")
        return "", ""

    # Parse TSV - header + data
    header = lines[0].split("\t")
    data = lines[1].split("\t")
    record = dict(zip(header, data))

    # Extract function
    function_raw = record.get("Function [CC]", "")
    # Clean up UniProt function text
    function_text = function_raw
    # Handle multiple FUNCTION: blocks (e.g., OXT has two)
    # Remove all "FUNCTION: " prefixes
    function_text = function_text.replace("FUNCTION: ", "")
    # Remove evidence tags like {ECO:...}
    function_text = re.sub(r'\s*\{ECO:\d+\|[^}]*\}', '', function_text)
    function_text = re.sub(r'\s*\{ECO:\d+\}', '', function_text)
    # Remove PubMed/Ref citations like (PubMed:12345, PubMed:67890) or (Ref.18)
    function_text = re.sub(r'\s*\((?:PubMed:\d+[,\s]*)+(?:Ref\.\d+)?\)', '', function_text)
    function_text = re.sub(r'\s*\(Ref\.\d+\)', '', function_text)
    function_text = re.sub(r'\s*\(By similarity\)', '', function_text)
    # Clean up double spaces and trailing dots
    function_text = re.sub(r'\s+', ' ', function_text)
    function_text = re.sub(r'\.\.+', '.', function_text)
    # Take first 1-2 sentences (up to second period)
    sentences = re.split(r'(?<=[.])\s+', function_text)
    if len(sentences) > 2:
        function_text = " ".join(sentences[:2])
    function_text = function_text.strip()

    # Extract GO Biological Process terms
    go_bp_raw = record.get("Gene Ontology (biological process)", "")
    go_terms = [t.strip() for t in go_bp_raw.split(";") if t.strip()]

    # Filter for relevant GO terms based on neurotransmitter system
    system = GENE_SYSTEM[gene_symbol]
    system_kw = SYSTEM_KEYWORDS[system]
    all_keywords = system_kw + GENERAL_KEYWORDS

    relevant_go = []
    for term in go_terms:
        term_lower = term.lower()
        for kw in system_kw:
            if kw.lower() in term_lower:
                relevant_go.append(term)
                break
    # If we got fewer than 3 system-specific, add general NT terms
    if len(relevant_go) < 3:
        for term in go_terms:
            if term in relevant_go:
                continue
            term_lower = term.lower()
            for kw in GENERAL_KEYWORDS:
                if kw.lower() in term_lower:
                    relevant_go.append(term)
                    break
            if len(relevant_go) >= 3:
                break

    # Cap at 3 most relevant
    relevant_go = relevant_go[:3]
    go_text = "; ".join(relevant_go) if relevant_go else ""

    return function_text, go_text


def main():
    print(f"Fetching UniProt annotations for {len(GENES)} genes...")
    results = []

    for i, gene in enumerate(GENES):
        print(f"  [{i+1}/{len(GENES)}] {gene}...", end=" ", flush=True)
        try:
            function_text, go_text = fetch_gene_annotation(gene)
            results.append({
                "gene_symbol": gene,
                "uniprot_function": function_text,
                "go_biological_process": go_text,
            })
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "gene_symbol": gene,
                "uniprot_function": f"ERROR: {e}",
                "go_biological_process": "",
            })
        # Rate limiting: 0.5s between requests
        if i < len(GENES) - 1:
            time.sleep(0.5)

    # Write output
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gene_symbol", "uniprot_function", "go_biological_process"],
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! Wrote {len(results)} entries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
