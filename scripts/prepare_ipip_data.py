from __future__ import annotations

import csv
import json
import re
import zipfile
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


class TableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "tr":
            self._current_row = []
        elif tag.lower() == "td" and self._current_row is not None:
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "td" and self._current_row is not None and self._current_cell is not None:
            text = " ".join(" ".join(self._current_cell).split())
            self._current_row.append(unescape(text))
            self._current_cell = None
        elif tag.lower() == "tr" and self._current_row is not None:
            if any(cell.strip() for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)


def parse_alphabetical_items() -> list[dict[str, str | int]]:
    source = RAW_DIR / "ipip_alphabetical_item_list.html"
    parser = TableTextParser()
    parser.feed(source.read_text(encoding="windows-1252", errors="replace"))

    items: list[dict[str, str | int]] = []
    code_pattern = re.compile(r"^[A-Z]\d+[A-Za-z]?$")
    for row in parser.rows:
        cells = [cell.strip() for cell in row if cell.strip()]
        if len(cells) < 2:
            continue
        text, raw_code = cells[0], cells[1]
        codes = [code.replace("*", "").strip() for code in raw_code.split(",")]
        codes = [code for code in codes if code]
        if not codes or any(not code_pattern.match(code) for code in codes):
            continue
        if len(text) < 2 or "IPIP Items" in text:
            continue
        items.append(
            {
                "id": f"IPIP_FULL_{len(items) + 1:04d}",
                "text": text,
                "survey_item_code": codes[0],
                "survey_item_codes": codes,
                "raw_survey_item_code": raw_code,
                "source": "IPIP Alphabetical Item List",
            }
        )
    return items


def column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value - 1


def parse_xlsx_sheet() -> list[list[str]]:
    source = RAW_DIR / "TedoneItemAssignmentTable30APR21.xlsx"
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(source) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings: list[str] = []
        for si in shared_root.findall("x:si", ns):
            parts = [node.text or "" for node in si.findall(".//x:t", ns)]
            shared_strings.append("".join(parts))

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in sheet_root.findall(".//x:sheetData/x:row", ns):
            values: list[str] = []
            for cell in row.findall("x:c", ns):
                ref = cell.attrib.get("r", "")
                idx = column_index(ref)
                while len(values) <= idx:
                    values.append("")
                value_node = cell.find("x:v", ns)
                value = "" if value_node is None else value_node.text or ""
                if cell.attrib.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                values[idx] = value.strip()
            if any(values):
                rows.append(values)
    return rows


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    full_items = parse_alphabetical_items()
    full_payload = {
        "inventory": "IPIP full alphabetical item list",
        "source_url": "https://ipip.ori.org/AlphabeticalItemList.htm",
        "item_count": len(full_items),
        "items": full_items,
    }
    write_json(DATA_DIR / "ipip_full_item_bank.json", full_payload)
    write_csv(
        DATA_DIR / "ipip_full_item_bank.csv",
        full_items,
        ["id", "text", "survey_item_code", "survey_item_codes", "raw_survey_item_code", "source"],
    )

    assignment_rows = parse_xlsx_sheet()
    headers = assignment_rows[0]
    normalized: list[dict[str, str]] = []
    for row in assignment_rows[1:]:
        entry = {headers[i] if i < len(headers) and headers[i] else f"column_{i + 1}": value for i, value in enumerate(row)}
        if any(entry.values()):
            normalized.append(entry)

    assignment_payload = {
        "inventory": "IPIP item assignment table",
        "source_url": "https://ipip.ori.org/TedoneItemAssignmentTable30APR21.xlsx",
        "row_count": len(normalized),
        "headers": headers,
        "rows": normalized,
    }
    write_json(DATA_DIR / "ipip_item_assignment_table.json", assignment_payload)
    write_csv(DATA_DIR / "ipip_item_assignment_table.csv", normalized, list(normalized[0].keys()))

    print(f"full_items={len(full_items)}")
    print(f"assignment_rows={len(normalized)}")


if __name__ == "__main__":
    main()
