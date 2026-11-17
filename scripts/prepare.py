#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

from gcommon import (
    ensure_folder,
    folder_url,
    get_services,
    list_folder_files,
    read_task_row,
    resolve_project_folder,
    update_cells,
    upsert_file,
    download_pdf_source,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ0-9\"'“‘(\[])")


def slugify_name(filename: str) -> str:
    stem = Path(filename).stem
    norm = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    clean = re.sub(r"[^A-Za-z0-9]+", "-", norm).strip("-")
    return clean[:48] or "pdf-doc"


def is_noise_block(text: str, y0_norm: float, y1_norm: float) -> bool:
    t = text.strip()
    if not t:
        return True
    if re.fullmatch(r"(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?", t, flags=re.IGNORECASE):
        return True
    if (y0_norm < 0.055 or y1_norm > 0.945) and len(t) <= 60 and re.fullmatch(r"[\d\s./\-–—|]+", t):
        return True
    return False


def extract_manifest_and_pages(pdf_path: str, work_dir: Path, dpi: int = 220):
    doc = fitz.open(pdf_path)
    pages_dir = work_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    pages_meta = []
    chunks = []
    chunk_idx = 0

    for pno in range(len(doc)):
        page = doc[pno]
        pw = float(page.rect.width) or 1.0
        ph = float(page.rect.height) or 1.0
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img_rel = f"pages/page_{pno + 1:03d}.png"
        pix.save(str(work_dir / img_rel))
        pages_meta.append({
            "page": pno + 1,
            "image": img_rel,
            "width": pix.width,
            "height": pix.height,
            "pt_width": round(pw, 2),
            "pt_height": round(ph, 2),
        })

        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (round(b[1] / 12.0), b[0]))
        for b in blocks:
            if len(b) >= 7 and b[6] != 0:
                continue
            bx0, by0, bx1, by1, raw_txt = b[0], b[1], b[2], b[3], b[4]
            y0n, y1n = by0 / ph, by1 / ph
            cleaned_block = re.sub(r"\s+", " ", (raw_txt or "").replace("-\n", "")).strip()
            if is_noise_block(cleaned_block, y0n, y1n):
                continue

            parts = [s.strip() for s in SENTENCE_SPLIT_RE.split(cleaned_block) if s.strip()]
            for sent in parts:
                found_rects = page.search_for(sent[:120])
                norm_rects = []
                for r in found_rects:
                    if r.y1 < by0 - 8 or r.y0 > by1 + 8:
                        continue
                    norm_rects.append([
                        round(max(0.0, min(1.0, r.x0 / pw)), 5),
                        round(max(0.0, min(1.0, r.y0 / ph)), 5),
                        round(max(0.0, min(1.0, r.x1 / pw)), 5),
                        round(max(0.0, min(1.0, r.y1 / ph)), 5),
                    ])
                if not norm_rects:
                    norm_rects = [[
                        round(max(0.0, min(1.0, bx0 / pw)), 5),
                        round(max(0.0, min(1.0, by0 / ph)), 5),
                        round(max(0.0, min(1.0, bx1 / pw)), 5),
                        round(max(0.0, min(1.0, by1 / ph)), 5),
                    ]]
                chunks.append({
                    "index": chunk_idx,
                    "page": pno + 1,
                    "text": sent,
                    "rects": norm_rects,
                })
                chunk_idx += 1

    doc.close()
    return pages_meta, chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--pdf", default="")
    ap.add_argument("--project", default="")
    args = ap.parse_args()

    drive, sheets = get_services()
    row_info = read_task_row(sheets, args.row)
    pdf_link = (args.pdf or row_info["pdf_link"]).strip()
    if not pdf_link:
        raise RuntimeError(f"Row {args.row} has empty PDF_Link")

    work_dir = Path("work") / f"row_{args.row}"
    work_dir.mkdir(parents=True, exist_ok=True)
    local_pdf = work_dir / "source.pdf"

    stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime("%Y%m%d-%H%M")
    existing_proj_id = ""
    try:
        existing_proj_id, existing_proj_name = resolve_project_folder(
            drive, sheets, args.row, args.project.strip(), create_if_missing=False
        )
    except Exception:
        existing_proj_id, existing_proj_name = "", ""

    if existing_proj_id:
        existing_files = list_folder_files(drive, existing_proj_id)
        if "manifest.json" in existing_files:
            update_cells(sheets, {f"C{args.row}": folder_url(existing_proj_id), f"B{args.row}": "Gdrive"})
            print(f"Row {args.row}: manifest.json already exists in {existing_proj_name}; skipped prepare.")
            return
        if "source.pdf" in existing_files:
            from gcommon import download_drive_file
            download_drive_file(drive, existing_files["source.pdf"]["id"], str(local_pdf))
            orig_name = existing_proj_name + ".pdf"
        else:
            orig_name = download_pdf_source(drive, pdf_link, str(local_pdf))
            upsert_file(drive, existing_proj_id, str(local_pdf), "source.pdf", "application/pdf")
        proj_id, proj_name = existing_proj_id, existing_proj_name
    else:
        orig_name = download_pdf_source(drive, pdf_link, str(local_pdf))
        default_proj = args.project.strip() or f"{slugify_name(orig_name)}_{stamp}"
        proj_id, proj_name = resolve_project_folder(drive, sheets, args.row, default_proj, create_if_missing=True)
        existing_files = list_folder_files(drive, proj_id)
        if "manifest.json" in existing_files:
            update_cells(sheets, {f"C{args.row}": folder_url(proj_id), f"B{args.row}": "Gdrive"})
            print(f"Row {args.row}: manifest.json already exists in {proj_name}; skipped prepare.")
            return
        upsert_file(drive, proj_id, str(local_pdf), "source.pdf", "application/pdf")
    pages_meta, chunks = extract_manifest_and_pages(str(local_pdf), work_dir)
    if not chunks:
        update_cells(sheets, {f"C{args.row}": folder_url(proj_id), f"B{args.row}": "Error"})
        raise RuntimeError(f"Row {args.row}: PDF contains no extractable text chunks")

    pages_folder_id = ensure_folder(drive, proj_id, "pages")
    for pmeta in pages_meta:
        img_file = work_dir / pmeta["image"]
        upsert_file(drive, pages_folder_id, str(img_file), img_file.name, "image/png")

    manifest = {
        "schema_version": "2.4",
        "prepared_by": "prepare.py_fallback",
        "row": args.row,
        "project_name": proj_name,
        "project_folder_id": proj_id,
        "source_pdf_name": orig_name,
        "page_count": len(pages_meta),
        "chunk_count": len(chunks),
        "pages": pages_meta,
        "chunks": chunks,
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    upsert_file(drive, proj_id, str(manifest_path), "manifest.json", "application/json")

    update_cells(sheets, {f"C{args.row}": folder_url(proj_id), f"B{args.row}": "Gdrive"})
    print(f"Prepared row {args.row}: {len(pages_meta)} pages, {len(chunks)} chunks -> {folder_url(proj_id)}")


if __name__ == "__main__":
    main()
