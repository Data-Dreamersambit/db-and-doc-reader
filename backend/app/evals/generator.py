"""Asset generator for evaluation dataset and multi-page handbook PDF.

Trade-off: Pure Python PDF generation avoids external heavy PDF authoring libraries.
Deterministic page boundaries ensure exact page attribution testing.
"""
import io
from pathlib import Path
import pdfplumber

HANDBOOK_PAGES: list[list[str]] = [
    [
        "OneLogica Corporate Overview and Executive Governance",
        "",
        "OneLogica Technologies Inc. was founded in 2021 by CEO Elena Vance and CTO Marcus Zhao.",
        "The global headquarters is located in San Francisco, California.",
        "The core mission is providing local-first, privacy-preserving AI platforms for enterprise analytics.",
        "The fiscal year commences on February 1 and concludes annually on January 31.",
        "Governance is overseen by a five-member Board of Directors with quarterly meetings.",
        "Corporate policies apply to all full-time, part-time, and contract personnel globally."
    ],
    [
        "Business Travel and Expense Reimbursement Policy",
        "",
        "Employees traveling on official company business are entitled to travel reimbursement.",
        "The standard domestic meal per diem allowance is 75 dollars per day.",
        "For commercial flights, any flight duration exceeding 4 hours qualifies for premium economy.",
        "Hotel lodging is capped at a maximum of 250 dollars per night in tier-one metropolitan cities.",
        "Ground transportation via rideshare or public transit is fully reimbursable.",
        "All travel expense reimbursement reports must be submitted within 30 days of trip completion."
    ],
    [
        "Paid Time Off Holidays and Parental Leave Policy",
        "",
        "Regular full-time salaried employees accrue 20 days of paid vacation annually.",
        "In addition to annual vacation, the company formally observes 11 official federal holidays.",
        "Employees may roll over up to 5 unused PTO days into the following calendar year.",
        "Primary caregiver parental leave provides 16 consecutive weeks of fully paid leave.",
        "Secondary caregivers receive 6 weeks of fully paid parental leave upon birth or adoption.",
        "Bereavement leave provides up to 5 consecutive paid days for immediate family members."
    ],
    [
        "Health Coverage Wellness Stipend and 401k Retirement Plan",
        "",
        "OneLogica provides comprehensive health insurance covering medical, dental, and vision.",
        "The employer covers 90 percent of monthly health insurance premiums for employees.",
        "Full-time team members receive an annual wellness stipend of 600 dollars for fitness and gym.",
        "Through the corporate 401k plan, the company matches contributions dollar-for-dollar up to 5 percent.",
        "Employer 401k matching contributions vest 100 percent immediately on day one of employment.",
        "Health savings accounts HSA feature an employer seed contribution of 500 dollars annually."
    ],
    [
        "Remote Work Hardware Lifecycle and Security Standards",
        "",
        "Remote employees receive a one-time home office equipment stipend of 1000 dollars for furniture.",
        "Company-issued work laptops follow a scheduled hardware refresh cycle every 3 years.",
        "Information security policy mandates multi-factor authentication with FIDO2 hardware keys.",
        "All corporate account passwords must be at least 16 characters in length.",
        "Unattended workstations must automatically lock after 5 minutes of inactivity.",
        "Confidential customer data must never be exported to unencrypted personal storage devices."
    ]
]


def create_handbook_pdf_bytes(pages: list[list[str]] | None = None) -> bytes:
    """Construct a compliant 5-page PDF binary with standard font and line leading."""
    page_content = pages or HANDBOOK_PAGES
    objects: list[str] = []
    num_pages = len(page_content)
    font_id = 3 + 2 * num_pages
    page_ids = [3 + 2 * i for i in range(num_pages)]
    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)

    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>")

    for i, lines in enumerate(page_content):
        content_id = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {content_id} 0 R >>"
        )

        stream_cmds = ["BT", "/F1 12 Tf", "18 TL", "50 720 Td"]
        for l_idx, line in enumerate(lines):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if l_idx == 0:
                stream_cmds.append(f"({escaped}) Tj")
            else:
                stream_cmds.append(f"T* ({escaped}) Tj")
        stream_cmds.append("ET")

        stream_content = "\n".join(stream_cmds)
        stream_bytes = stream_content.encode("latin1")
        objects.append(f"<< /Length {len(stream_bytes)} >>\nstream\n{stream_content}\nendstream")

    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, obj_body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n{obj_body}\nendobj\n".encode("latin1"))

    startxref = out.tell()
    out.write(b"xref\n")
    out.write(f"0 {len(objects) + 1}\n".encode("latin1"))
    out.write(b"0000000000 65535 f \n")
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode("latin1"))
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{startxref}\n%%EOF\n".encode("latin1")
    )
    return out.getvalue()


def ensure_eval_assets(base_dir: Path | None = None) -> tuple[Path, Path]:
    """Ensure sales.csv and employee_handbook.pdf exist on disk and return paths."""
    root = base_dir or Path.cwd()
    sample_dir = root / "sample_data"
    sample_dir.mkdir(parents=True, exist_ok=True)

    csv_path = sample_dir / "sales.csv"
    pdf_path = sample_dir / "employee_handbook.pdf"

    if not pdf_path.exists():
        pdf_bytes = create_handbook_pdf_bytes()
        pdf_path.write_bytes(pdf_bytes)

    return csv_path, pdf_path
