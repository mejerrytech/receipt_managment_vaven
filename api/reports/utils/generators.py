from __future__ import annotations

import csv
import io
from uuid import UUID

from pathlib import Path
from typing import Any


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_text_pdf(path: Path, title: str, lines: list[str]) -> None:
    """Minimal text-only PDF (no external dependency)."""
    content_parts = ["BT /F1 11 Tf"]
    y = 800
    content_parts.append(f"50 {y} Td ({_escape_pdf_text(title)}) Tj")
    y -= 22
    content_parts.append(f"0 -22 Td ({_escape_pdf_text('-' * 72)}) Tj")
    for line in lines:
        y -= 14
        if y < 50:
            break
        safe = _escape_pdf_text(line[:110])
        content_parts.append(f"0 -14 Td ({safe}) Tj")
    content_parts.append("ET")
    stream = "\n".join(content_parts)
    stream_bytes = stream.encode("latin-1", errors="replace")

    objects: list[bytes] = []
    objects.append(b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n")
    objects.append(b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n")
    objects.append(
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
        b"/Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj\n"
    )
    objects.append(
        f"4 0 obj<< /Length {len(stream_bytes)} >>stream\n".encode()
        + stream_bytes
        + b"\nendstream endobj\n"
    )
    objects.append(
        b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
    )

    pdf = io.BytesIO()
    pdf.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(pdf.tell())
        pdf.write(obj)
    xref_pos = pdf.tell()
    pdf.write(f"xref\n0 {len(offsets)}\n".encode())
    pdf.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.write(f"{off:010d} 00000 n \n".encode())
    pdf.write(
        f"trailer<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    )
    path.write_bytes(pdf.getvalue())


def build_report_lines(
    preview: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    include_itc: bool,
    include_categories: bool,
    itc: dict[str, Any] | None = None,
    categories: list[dict[str, Any]] | None = None,
) -> list[str]:
    lines = [
        f"Period: {preview['period_label']}",
        f"Total Invoices: {preview['total_invoices']}",
        f"Total Amount: {preview['currency']} {preview['total_amount']:,.2f}",
        f"Total GST: {preview['currency']} {preview['total_gst']:,.2f}",
        f"ITC Eligible: {preview['currency']} {preview['itc_eligible']:,.2f}",
        "",
        "Confirmed Receipts:",
    ]
    for row in rows:
        lines.append(
            f"- {row['date']} | {row['vendor']} | {row['currency']} {row['amount']:,.2f} "
            f"| GST {row['gst_amount']:,.2f} | {row['category']}"
        )
    if include_itc and itc:
        lines.extend(
            [
                "",
                "ITC Summary:",
                f"  Eligible invoices: {itc['eligible_invoices']}",
                f"  ITC eligible GST: {preview['currency']} {itc['itc_eligible_total']:,.2f}",
                f"  Ineligible invoices: {itc['ineligible_invoices']}",
            ]
        )
    if include_categories and categories:
        lines.extend(["", "Category Breakdown:"])
        for cat in categories:
            lines.append(
                f"  {cat['category']}: {cat['count']} invoices, "
                f"{preview['currency']} {cat['amount']:,.2f}, GST {cat['gst']:,.2f}"
            )
    return lines


def write_csv_report(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "date",
        "vendor",
        "invoice_number",
        "gstin",
        "amount",
        "gst_amount",
        "itc_eligible",
        "category",
        "source",
        "currency",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def generate_report_files(
    user_id: UUID,
    report_id: str,
    preview: dict[str, Any],
    rows: list[dict[str, Any]],
    report_format: str,
    *,
    include_itc: bool,
    include_categories: bool,
    itc: dict[str, Any] | None = None,
    categories: list[dict[str, Any]] | None = None,
) -> dict[str, str | None]:
    from api.reports.utils.storage import report_file_path

    files: dict[str, str | None] = {"pdf": None, "csv": None}
    lines = build_report_lines(
        preview,
        rows,
        include_itc=include_itc,
        include_categories=include_categories,
        itc=itc,
        categories=categories,
    )

    if report_format in ("pdf", "both"):
        pdf_path = report_file_path(user_id, report_id, "pdf")
        write_text_pdf(pdf_path, "Expense Report", lines)
        files["pdf"] = str(pdf_path)

    if report_format in ("csv", "both"):
        csv_path = report_file_path(user_id, report_id, "csv")
        write_csv_report(csv_path, rows)
        files["csv"] = str(csv_path)

    return files
