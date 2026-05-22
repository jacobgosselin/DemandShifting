"""
Digitize IRS SOI Corporate Complete Report PDFs using Gemini Flash (2-pass).

Pass 1: List all tables in the PDF (title, table number, page number).
Pass 2: Extract each table as structured JSON.

Usage:
    # Single year (test):
    python digitize_irs_pdfs.py --year 1970

    # All years:
    python digitize_irs_pdfs.py --all

    # Set API key via env var or paste when prompted:
    GEMINI_API_KEY=your_key python digitize_irs_pdfs.py --year 1970
"""

import subprocess
import sys

# Auto-install new SDK if needed
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Installing google-genai...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])
    from google import genai
    from google.genai import types

import argparse
import csv
import json
import os
import pathlib
import re
import time

# ── Paths ─────────────────────────────────────────────────────────────────────

PDF_DIR = pathlib.Path(
    "/Users/jacobgosselin/Library/CloudStorage/"
    "GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/"
    "research_ideas/negative_earnings/data/raw/irs_data_downloaded/"
    "irs_pre94_pdf/full"
)

OUT_DIR = pathlib.Path(
    "/Users/jacobgosselin/Library/CloudStorage/"
    "GoogleDrive-jacob.gosselin@u.northwestern.edu/My Drive/"
    "research_ideas/negative_earnings/data/raw/irs_data_downloaded/"
    "irs_pre94_csv"
)

# Map year → filename (some years have non-standard names)
YEAR_TO_FILE = {
    1970: "70cocrar.pdf",
    1971: "71cocrar.pdf",
    1972: "72cocrar.pdf",
    1973: "73cocrar.pdf",
    1974: "74cocrar.pdf",
    1975: "75cocrar.pdf",
    1976: "76cocrar.pdf",
    1977: "77cocrar.pdf",
    1978: "78.5_cocomprept.pdf",
    1980: "80cocrar.pdf",
    1981: "81cocrar.pdf",
    1982: "82cocrar.pdf",
    1983: "83cocrar.pdf",
    1984: "84cocrar.pdf",
    1985: "85cocrar.pdf",
    1986: "86cocrar.pdf",
    1987: "87cocomprept.pdf",
    1988: "88cocrar.pdf",
    1989: "89cocrar.pdf",
    1990: "90cocrar.pdf",
    1991: "91cocrar.pdf",
    1992: "92cocrar.pdf",
    1993: "93cocrar.pdf",
}

MODEL = "gemini-2.5-flash-lite"

# ── Prompts ────────────────────────────────────────────────────────────────────

PASS1_PROMPT = """This is an IRS Statistics of Income (SOI) Corporate Complete Report.

List every table in this document. For each table return a JSON array where each element has:
  - "table_number": the table number as printed (e.g. "1", "A-1", etc.)
  - "title": the full table title
  - "page": the page number where the table starts

Return ONLY a valid JSON array, no other text. Example format:
[
  {"table_number": "1", "title": "Returns of Active Corporations...", "page": 3},
  ...
]"""

PASS2_PROMPT_TEMPLATE = """This is an IRS Statistics of Income (SOI) Corporate Complete Report.

Extract Table {table_number} titled "{title}" (starts on page {page}).

Return the table as a JSON object with these keys:
  - "table_number": "{table_number}"
  - "title": "{title}"
  - "year": the tax year this report covers (integer)
  - "units": the unit of measurement noted in the table header (e.g. "thousands of dollars", "number")
  - "notes": any footnotes or notes beneath the table (string, empty if none)
  - "headers": a list of column header strings (use empty string "" for blank headers)
  - "rows": a list of rows, each row a list of cell values (strings). Include ALL rows verbatim.

Rules:
- Do NOT summarize or omit any rows.
- Preserve numeric values exactly as printed (commas, dashes for zero, asterisks, etc.).
- If a row is a subtotal or section header with no data, still include it with empty strings for numeric cells.
- Return ONLY valid JSON, no other text."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        key = input("Paste your Gemini API key: ").strip()
    if not key:
        sys.exit("No API key provided.")
    return key


def upload_pdf(client, pdf_path: pathlib.Path):
    print(f"  Uploading {pdf_path.name}...")
    with open(pdf_path, "rb") as f:
        file = client.files.upload(
            file=f,
            config=types.UploadFileConfig(mime_type="application/pdf", display_name=pdf_path.name),
        )
    # Wait for processing
    while file.state.name == "PROCESSING":
        time.sleep(2)
        file = client.files.get(name=file.name)
    if file.state.name != "ACTIVE":
        raise RuntimeError(f"File upload failed: {file.state.name}")
    print(f"  Uploaded: {file.uri}")
    return file


def call_model(client, prompt: str, file, retries=3, backoff=30):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_uri(file_uri=file.uri, mime_type="application/pdf"),
                    prompt,
                ],
            )
            return response.text
        except Exception as e:
            if attempt < retries - 1:
                wait = backoff * (attempt + 1)
                print(f"  Rate limit / error ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def parse_json_response(text: str):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())


def table_to_csv(table_data: dict, out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# year", table_data.get("year", "")])
        writer.writerow(["# table_number", table_data.get("table_number", "")])
        writer.writerow(["# title", table_data.get("title", "")])
        writer.writerow(["# units", table_data.get("units", "")])
        writer.writerow(["# notes", table_data.get("notes", "")])
        writer.writerow([])
        writer.writerow(table_data.get("headers", []))
        for row in table_data.get("rows", []):
            writer.writerow(row)


# ── Core logic ─────────────────────────────────────────────────────────────────

def process_year(year: int, client):
    filename = YEAR_TO_FILE.get(year)
    if not filename:
        print(f"No file mapped for year {year}, skipping.")
        return

    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}, skipping.")
        return

    year_out_dir = OUT_DIR / str(year)
    year_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {year} ({filename})")
    print(f"{'='*60}")

    pdf_file = upload_pdf(client, pdf_path)

    # ── Pass 1: Get table of contents ──────────────────────────────────────
    print("  Pass 1: Discovering tables...")
    raw_toc = call_model(client, PASS1_PROMPT, pdf_file)

    toc_path = year_out_dir / "_table_of_contents.json"
    toc_path.write_text(raw_toc, encoding="utf-8")

    try:
        tables = parse_json_response(raw_toc)
        print(f"  Found {len(tables)} tables.")
    except json.JSONDecodeError as e:
        print(f"  ERROR parsing table of contents JSON: {e}")
        print(f"  Raw response saved to {toc_path}. Fix and re-run.")
        client.files.delete(name=pdf_file.name)
        return

    # ── Pass 2: Extract each table ─────────────────────────────────────────
    for i, tbl in enumerate(tables):
        tbl_num = tbl.get("table_number", str(i + 1))
        tbl_title = tbl.get("title", "unknown")
        tbl_page = tbl.get("page", "?")

        safe_num = re.sub(r"[^\w\-]", "_", str(tbl_num))
        csv_path = year_out_dir / f"table_{safe_num}.csv"

        if csv_path.exists():
            print(f"  [{i+1}/{len(tables)}] Table {tbl_num} already extracted, skipping.")
            continue

        print(f"  [{i+1}/{len(tables)}] Extracting Table {tbl_num}: {tbl_title[:60]}...")

        prompt = PASS2_PROMPT_TEMPLATE.format(
            table_number=tbl_num,
            title=tbl_title,
            page=tbl_page,
        )

        raw_table = call_model(client, prompt, pdf_file)

        json_path = year_out_dir / f"table_{safe_num}.json"
        json_path.write_text(raw_table, encoding="utf-8")

        try:
            table_data = parse_json_response(raw_table)
            table_to_csv(table_data, csv_path)
            print(f"    Saved: {csv_path.name}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    ERROR parsing table JSON: {e}. Raw saved to {json_path}.")

        time.sleep(1)

    client.files.delete(name=pdf_file.name)
    print(f"  Done with {year}. Output in: {year_out_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Digitize IRS SOI PDFs with Gemini Flash.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, help="Process a single year (e.g. --year 1970)")
    group.add_argument("--all", action="store_true", help="Process all years")
    args = parser.parse_args()

    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    if args.year:
        process_year(args.year, client)
    else:
        for year in sorted(YEAR_TO_FILE.keys()):
            process_year(year, client)
            time.sleep(2)

    print("\nAll done.")


if __name__ == "__main__":
    main()
