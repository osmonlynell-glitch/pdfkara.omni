#!/usr/bin/env python3
"""
mushu_word_kit.py — Mushu (#46) Official Word-Spotlight Karaoke Kit (v2.6-column)
==================================================================================
Exclusively runs on Mushu VM for:
  1) `prep`: Render 220 DPI PDF pages + extract verbatim body chunks in true
             column-by-column reading order (multi-column Recursive XY-Cut +
             left-margin callout anchoring, excluding running headers/footers)
             WITH exact per-word bounding boxes (`words: [{"w": ..., "r": [x0,y0,x1,y1]}]`).
  2) `html`: Build 100% deterministic HyperFrames 0.9.37 `1920x1080` (`16:9`)
             Word-Spotlight `karaoke_16x9.html` (highlights ONLY the single word
             currently spoken directly on the PDF page, with smooth deterministic
             line-level Smart Camera Pan, C0-continuous). Automatically upgrades
             v2.4 manifests using `source.pdf` when `words` boxes are missing.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import fitz  # PyMuPDF

HAS_ALNUM_RE = re.compile(r"[A-Za-z0-9\u00C0-\u1EF9]")
COMPOUND_SPLIT_RE = re.compile(r"(?<=[A-Za-z0-9\u00C0-\u1EF9])([—–/])(?=[A-Za-z0-9\u00C0-\u1EF9])")
STANDALONE_PAGENO_RE = re.compile(r"(?:page\s*)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?", re.IGNORECASE)
LEADING_NEW_ITEM_RE = re.compile(
    r"^(?:[✓✔•◦▪▫\-–—\*»]|(?:\d{1,3}|[A-Za-z])[.)]\s|[A-Z]{3,}[—–:-]|(?:Core Vocabulary|Language of Instruction|Big Question|Lesson Guiding Question|Today’s Question|Today's Question|Tie to|For detailed|See the Activity|What |How |Why |Is |Are |Do |Does |Where |When |Ask |Take |Keep |Lead |Write |Call |Observe |Explain |Prepare |Identify |Notice |Use |Review )\b)"
)
ABBREV_END_RE = re.compile(r"(?:\b(?:e\.g|i\.e|vs|etc|fig|ch|ap|st|mr|mrs|ms|dr)|^\d{1,3}|\b\d{1,2})\.$", re.IGNORECASE)
SENTENCE_END_CAND_RE = re.compile(
    r"[.!?…]\s+(?=[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ0-9\"'“‘(\[•])"
)
LIGATURE_MAP = str.maketrans({
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
})


def clean_pdf_text(raw: str) -> str:
    s = (raw or "").translate(LIGATURE_MAP).replace("-\n", "")
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^([✓✔•◦▪▫])\s+\1\s*", r"\1 ", s)
    return s


def norm_tok(s: str) -> str:
    s2 = unicodedata.normalize("NFKC", s or "").replace("đ", "d").replace("Đ", "d")
    return re.sub(r"[^\w]+", "", s2.lower(), flags=re.UNICODE)


def split_visual_tokens(text: str) -> list[str]:
    expanded = COMPOUND_SPLIT_RE.sub(r"\1 ", (text or "").strip())
    return [t for t in expanded.split() if t.strip()]


def split_clean_sentences(text: str) -> list[str]:
    """Split block text into sentences without cutting at list numbers ('1. Foo') or abbreviations."""
    t = (text or "").strip()
    if not t:
        return []
    out: list[str] = []
    start = 0
    for m in SENTENCE_END_CAND_RE.finditer(t):
        end_pos = m.start() + 1
        head = t[start:end_pos].strip()
        last_tok = head.split()[-1] if head.split() else ""
        if ABBREV_END_RE.search(last_tok) or (len(last_tok) == 2 and last_tok[0].isalpha() and last_tok[1] == "."):
            continue
        out.append(head)
        start = m.end()
    tail = t[start:].strip()
    if tail:
        out.append(tail)
    return out


def hf_signature(text: str) -> str:
    return re.sub(r"\d+", "#", (text or "").lower()).strip()


def is_noise_block(text: str, y0_norm: float, y1_norm: float) -> bool:
    t = text.strip()
    if not t or not HAS_ALNUM_RE.search(t):
        return True
    if STANDALONE_PAGENO_RE.fullmatch(t):
        return True
    if (y0_norm < 0.055 or y1_norm > 0.945) and len(t) <= 60 and re.fullmatch(r"[\d\s./\-–—|]+", t):
        return True
    return False


def merge_line_rects(word_boxes: list[dict]) -> list[list[float]]:
    if not word_boxes:
        return []
    lines: list[list[float]] = []
    for wb in word_boxes:
        r = wb["r"]
        if not lines:
            lines.append(list(r))
            continue
        prev = lines[-1]
        prev_cy = (prev[1] + prev[3]) * 0.5
        cur_cy = (r[1] + r[3]) * 0.5
        prev_h = max(0.005, prev[3] - prev[1])
        if abs(cur_cy - prev_cy) <= prev_h * 0.65 and r[0] >= prev[0] - 0.03:
            prev[0] = round(min(prev[0], r[0]), 5)
            prev[1] = round(min(prev[1], r[1]), 5)
            prev[2] = round(max(prev[2], r[2]), 5)
            prev[3] = round(max(prev[3], r[3]), 5)
        else:
            lines.append(list(r))
    return lines


def split_chunk_clauses(sent: str, max_chars: int = 240) -> list[str]:
    sent = sent.strip()
    if len(sent) <= max_chars:
        return [sent] if sent else []
    parts = re.split(r"(?<=[,;:])\s+|(?<=[—–])\s*", sent)
    out: list[str] = []
    cur = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_chars:
            cur = cur + " " + p
        else:
            out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out


def extract_page_words(page: fitz.Page) -> list[dict]:
    pw = float(page.rect.width) or 1.0
    ph = float(page.rect.height) or 1.0
    out = []
    for idx_w, w in enumerate(page.get_text("words")):
        x0, y0, x1, y1, txt, bno, lno, wno = w
        out.append({
            "wid": idx_w,
            "w": txt,
            "norm": norm_tok(txt),
            "r": [
                round(max(0.0, min(1.0, x0 / pw)), 5),
                round(max(0.0, min(1.0, y0 / ph)), 5),
                round(max(0.0, min(1.0, x1 / pw)), 5),
                round(max(0.0, min(1.0, y1 / ph)), 5),
            ],
            "bno": bno,
            "lno": lno,
            "wno": wno,
            "used": False,
        })
    return out


def resolve_words_for_chunks_from_pdf(doc: fitz.Document, chunks: list[dict]) -> None:
    page_words_cache = {pno + 1: extract_page_words(doc[pno]) for pno in range(len(doc))}

    for ch in chunks:
        pnum = int(ch.get("page") or 1)
        pwords = page_words_cache.get(pnum) or []
        rects = ch.get("rects") or []
        toks = split_visual_tokens(ch.get("text") or "")
        if not toks:
            ch["words"] = []
            continue

        touched_blocks = set()
        for w in pwords:
            rx0, ry0, rx1, ry1 = w["r"]
            cx, cy = (rx0 + rx1) * 0.5, (ry0 + ry1) * 0.5
            if any((r[0] - 0.015) <= cx <= (r[2] + 0.015) and (r[1] - 0.008) <= cy <= (r[3] + 0.008) for r in rects):
                touched_blocks.add(w["bno"])

        if touched_blocks:
            cands = [w for w in pwords if not w["used"] and w["bno"] in touched_blocks]
        else:
            cands = [w for w in pwords if not w["used"]]

        tok_norms = [norm_tok(t) or t for t in toks]
        cand_norms = [c["norm"] or c["w"] for c in cands]
        sm = difflib.SequenceMatcher(a=tok_norms, b=cand_norms, autojunk=False)
        matched_cnt = sum(n for _, _, n in sm.get_matching_blocks())
        if matched_cnt < len(toks) and touched_blocks:
            cands2 = [w for w in pwords if w["bno"] in touched_blocks]
            sm2 = difflib.SequenceMatcher(a=tok_norms, b=[c["norm"] or c["w"] for c in cands2], autojunk=False)
            if sum(n for _, _, n in sm2.get_matching_blocks()) > matched_cnt:
                cands, sm = cands2, sm2

        assigned: list[dict | None] = [None] * len(toks)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    cw = cands[j1 + k]
                    cw["used"] = True
                    assigned[i1 + k] = {"w": toks[i1 + k], "r": cw["r"]}
            elif tag == "replace":
                sa, sb = i2 - i1, j2 - j1
                if sa == 1 and sb >= 1:
                    sub = cands[j1:j2]
                    for x in sub:
                        x["used"] = True
                    if abs(sub[0]["r"][1] - sub[-1]["r"][1]) < 0.012:
                        assigned[i1] = {
                            "w": toks[i1],
                            "r": [
                                round(min(x["r"][0] for x in sub), 5),
                                round(min(x["r"][1] for x in sub), 5),
                                round(max(x["r"][2] for x in sub), 5),
                                round(max(x["r"][3] for x in sub), 5),
                            ],
                        }
                    else:
                        assigned[i1] = {"w": toks[i1], "r": sub[0]["r"], "r2": sub[-1]["r"]}
                elif sb == 1 and sa >= 1:
                    br = cands[j1]["r"]
                    cands[j1]["used"] = True
                    tot_c = sum(max(1, len(toks[i1 + k])) for k in range(sa))
                    cur_x, bw = br[0], br[2] - br[0]
                    for k in range(sa):
                        nx = cur_x + bw * (max(1, len(toks[i1 + k])) / tot_c)
                        assigned[i1 + k] = {
                            "w": toks[i1 + k],
                            "r": [round(cur_x, 5), br[1], round(nx, 5), br[3]],
                        }
                        cur_x = nx
                elif sa == sb and sa > 0:
                    for k in range(sa):
                        cw = cands[j1 + k]
                        cw["used"] = True
                        assigned[i1 + k] = {"w": toks[i1 + k], "r": cw["r"]}
                elif sb > 0 and sa > 0:
                    for k in range(sa):
                        pick = cands[j1 + min(sb - 1, int(k * sb / sa))]
                        pick["used"] = True
                        assigned[i1 + k] = {"w": toks[i1 + k], "r": pick["r"]}

        # Interpolate any rare remaining None from neighboring words or chunk rect
        fallback_rect = rects[0] if rects else [0.1, 0.1, 0.8, 0.14]
        final_words: list[dict] = []
        for idx_t, item_w in enumerate(assigned):
            if item_w is not None:
                final_words.append(item_w)
                continue
            prev_r = next((assigned[p]["r"] for p in range(idx_t - 1, -1, -1) if assigned[p]), None)
            next_r = next((assigned[n]["r"] for n in range(idx_t + 1, len(assigned)) if assigned[n]), None)
            if prev_r and next_r and abs(prev_r[1] - next_r[1]) < 0.015 and next_r[0] >= prev_r[2]:
                final_words.append({"w": toks[idx_t], "r": [prev_r[2], prev_r[1], next_r[0], prev_r[3]]})
            elif prev_r:
                w_est = min(0.12, max(0.02, 0.008 * len(toks[idx_t])))
                x0 = min(0.95, prev_r[2] + 0.003)
                final_words.append({"w": toks[idx_t], "r": [round(x0, 5), prev_r[1], round(min(0.98, x0 + w_est), 5), prev_r[3]]})
            elif next_r:
                w_est = min(0.12, max(0.02, 0.008 * len(toks[idx_t])))
                x1 = max(0.05, next_r[0] - 0.003)
                final_words.append({"w": toks[idx_t], "r": [round(max(0.02, x1 - w_est), 5), next_r[1], round(x1, 5), next_r[3]]})
            else:
                final_words.append({"w": toks[idx_t], "r": list(fallback_rect)})

        ch["words"] = final_words
        if final_words:
            ch["rects"] = merge_line_rects(final_words)


def extract_raw_page_blocks(page: fitz.Page) -> list[dict]:
    """Extract normalized horizontal text blocks with geometry and font sizes."""
    pw = float(page.rect.width) or 1.0
    ph = float(page.rect.height) or 1.0
    d = page.get_text("dict")
    items: list[dict] = []
    bullet_pts: list[tuple[float, float, float, float]] = []
    for blk in d.get("blocks") or []:
        if blk.get("type") != 0:
            continue
        lines_kept = []
        sizes: list[float] = []
        for ln in blk.get("lines") or []:
            dv = ln.get("dir") or (1.0, 0.0)
            if abs(float(dv[0])) < 0.6:
                continue
            ltxt = "".join(str(sp.get("text") or "") for sp in (ln.get("spans") or []))
            if not ltxt.strip():
                continue
            bb = ln.get("bbox")
            if bb and clean_pdf_text(ltxt) in ("•", "✓", "✔", "◦", "▪", "▫"):
                bullet_pts.append((float(bb[0]) / pw, float(bb[1]) / ph, float(bb[2]) / pw, float(bb[3]) / ph))
            if not clean_pdf_text(ltxt):
                continue
            lines_kept.append((ltxt, bb))
            for sp in (ln.get("spans") or []):
                if str(sp.get("text") or "").strip():
                    sizes.append(float(sp.get("size") or 12.0))
        if not lines_kept:
            continue
        txt = clean_pdf_text("\n".join(lt for lt, _ in lines_kept))
        xs0 = [float(bb[0]) / pw for _, bb in lines_kept if bb]
        ys0 = [float(bb[1]) / ph for _, bb in lines_kept if bb]
        xs1 = [float(bb[2]) / pw for _, bb in lines_kept if bb]
        ys1 = [float(bb[3]) / ph for _, bb in lines_kept if bb]
        if not xs0:
            continue
        x0 = max(0.0, min(1.0, min(xs0)))
        y0 = max(0.0, min(1.0, min(ys0)))
        x1 = max(0.0, min(1.0, max(xs1)))
        y1 = max(0.0, min(1.0, max(ys1)))
        if is_noise_block(txt, y0, y1):
            continue
        items.append({
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "cx": (x0 + x1) * 0.5,
            "cy": (y0 + y1) * 0.5,
            "w": x1 - x0,
            "h": y1 - y0,
            "max_size": max(sizes) if sizes else 12.0,
            "text": txt,
            "is_badge": False,
            "has_bullet": False,
        })
    for it in items:
        if any(
            bx0 <= it["x0"] + 0.015 and (it["x0"] - bx0) <= 0.05 and by1 >= it["y0"] - 0.005 and by0 <= it["y0"] + 0.025
            for bx0, by0, bx1, by1 in bullet_pts
        ):
            it["has_bullet"] = True
    return items


def filter_headers_and_footers(pages_blocks: list[list[dict]]) -> list[list[dict]]:
    """Exclude repeating or margin running headers and footers across all pages."""
    top_sig_pages: dict[str, set[int]] = {}
    bot_sig_pages: dict[str, set[int]] = {}
    for pno, blks in enumerate(pages_blocks):
        for b in blks:
            sig = hf_signature(b["text"])
            if b["y1"] <= 0.068 and b["h"] <= 0.045:
                top_sig_pages.setdefault(sig, set()).add(pno)
            if b["y0"] >= 0.920 and b["h"] <= 0.045:
                bot_sig_pages.setdefault(sig, set()).add(pno)

    rep_top = {s for s, ps in top_sig_pages.items() if len(ps) >= 2}
    rep_bot = {s for s, ps in bot_sig_pages.items() if len(ps) >= 2}

    cleaned_pages: list[list[dict]] = []
    for blks in pages_blocks:
        kept: list[dict] = []
        for b in blks:
            t = b["text"]
            sig = hf_signature(t)
            # Bottom margin footer: starts in bottom ~7.5% and is thin running bar / small font / page+title pattern
            if b["y0"] >= 0.922 and b["h"] <= 0.045:
                if (
                    sig in rep_bot
                    or b["max_size"] <= 10.5
                    or "|" in t
                    or re.search(r"^(?:\d{1,4}\s+\S|\S.*\s+\d{1,4}$)", t)
                ):
                    continue
            # Top margin header: ends in top ~5.5% and is thin running bar / small font / repeated
            if b["y1"] <= 0.055 and b["h"] <= 0.042:
                if (
                    sig in rep_top
                    or b["max_size"] <= 10.2
                    or "|" in t
                    or re.search(r"^(?:\d{1,4}\s+[A-Z]|[A-Z].*\s+\d{1,4}$)", t)
                ):
                    continue
            kept.append(b)
        cleaned_pages.append(kept)
    return cleaned_pages


def find_best_vertical_gutter(items: list[dict], min_gap: float = 0.014) -> tuple[float, float] | None:
    """Return widest clean interior vertical whitespace gutter separating >=2 columns."""
    if len(items) < 2:
        return None
    s_items = sorted(items, key=lambda b: (b["x0"], b["x1"]))
    max_x1 = s_items[0]["x1"]
    min_all_x0 = s_items[0]["x0"]
    max_all_x1 = max(b["x1"] for b in items)
    best: tuple[float, float] | None = None
    best_w = 0.0
    for b in s_items[1:]:
        gap = b["x0"] - max_x1
        if gap >= min_gap:
            gmid = (max_x1 + b["x0"]) * 0.5
            if (gmid - min_all_x0) >= 0.08 and (max_all_x1 - gmid) >= 0.08 and 0.18 <= gmid <= 0.85:
                if gap > best_w:
                    best_w = gap
                    best = (max_x1, b["x0"])
        if b["x1"] > max_x1:
            max_x1 = b["x1"]
    return best


def find_horizontal_gaps(items: list[dict], min_gap: float = 0.005) -> list[tuple[float, float]]:
    if len(items) < 2:
        return []
    s_items = sorted(items, key=lambda b: (b["y0"], b["y1"]))
    max_y1 = s_items[0]["y1"]
    gaps: list[tuple[float, float]] = []
    for b in s_items[1:]:
        if b["y0"] - max_y1 >= min_gap:
            gaps.append((max_y1, b["y0"]))
        if b["y1"] > max_y1:
            max_y1 = b["y1"]
    return gaps


def _has_both_sides(sub: list[dict], gmid: float) -> bool:
    has_l = any(b["cx"] < gmid for b in sub)
    has_r = any(b["cx"] >= gmid for b in sub)
    return has_l and has_r


def _is_spanning_transition(top: list[dict], bot: list[dict]) -> bool:
    """True when one side has a vertical column gutter or consists only of wider/single-column spanning blocks."""
    tg = find_best_vertical_gutter(top, min_gap=0.014)
    bg = find_best_vertical_gutter(bot, min_gap=0.014)
    if tg is not None or bg is not None:
        return True
    # Also prioritize isolating outer full-width header/footer bands so inner columns remain intact
    t_wide = any(b["w"] >= 0.48 for b in top)
    b_wide = any(b["w"] >= 0.48 for b in bot)
    return t_wide != b_wide


def xy_cut_order(items: list[dict]) -> list[dict]:
    """Column-preserving Recursive XY-Cut: finishes each column completely before the next."""
    if len(items) <= 1:
        return list(items)

    v_gutter = find_best_vertical_gutter(items, min_gap=0.014)
    h_gaps = find_horizontal_gaps(items, min_gap=0.005)

    if v_gutter is not None:
        gmid_x = (v_gutter[0] + v_gutter[1]) * 0.5
        # Only cut horizontally inside a multi-column region when it truly terminates a band
        # (either one/both columns end across the gap, or a large >= 0.055 section break occurs).
        valid_h: list[tuple[float, float, float]] = []
        for gy0, gy1 in h_gaps:
            ymid = (gy0 + gy1) * 0.5
            top = [b for b in items if b["cy"] < ymid]
            bot = [b for b in items if b["cy"] >= ymid]
            if not top or not bot:
                continue
            gap_h = gy1 - gy0
            both_stay_split = _has_both_sides(top, gmid_x) and _has_both_sides(bot, gmid_x)
            if not both_stay_split or gap_h >= 0.055:
                valid_h.append((gap_h, gy0, gy1))
        if valid_h:
            valid_h.sort(key=lambda t: (-t[0], t[1]))
            _, gy0, gy1 = valid_h[0]
            ymid = (gy0 + gy1) * 0.5
            top = [b for b in items if b["cy"] < ymid]
            bot = [b for b in items if b["cy"] >= ymid]
            return xy_cut_order(top) + xy_cut_order(bot)

        left = [b for b in items if b["cx"] < gmid_x]
        right = [b for b in items if b["cx"] >= gmid_x]
        if left and right:
            return xy_cut_order(left) + xy_cut_order(right)

    # Full-width (or mixed spanning + multi-column) region:
    # first prefer horizontal cuts that cleanly isolate a multi-column or spanning boundary without splitting mid-column.
    if h_gaps:
        scored: list[tuple[int, float, float, float]] = []
        for gy0, gy1 in h_gaps:
            ymid = (gy0 + gy1) * 0.5
            top = [b for b in items if b["cy"] < ymid]
            bot = [b for b in items if b["cy"] >= ymid]
            if not top or not bot:
                continue
            prio = 1 if _is_spanning_transition(top, bot) else 0
            scored.append((prio, gy1 - gy0, -gy0, ymid))
        if scored:
            scored.sort(reverse=True)
            _, _, _, ymid = scored[0]
            top = [b for b in items if b["cy"] < ymid]
            bot = [b for b in items if b["cy"] >= ymid]
            return xy_cut_order(top) + xy_cut_order(bot)

    return sorted(items, key=lambda b: (round(b["y0"] / 0.015), b["x0"]))


def should_coalesce_blocks(prev: dict, curr: dict) -> bool:
    """Merge multi-line body blocks split by indentation inside the same column."""
    if prev.get("is_badge") or curr.get("is_badge") or curr.get("has_bullet"):
        return False
    if prev["max_size"] >= 13.5 or curr["max_size"] >= 13.5:
        return False
    if abs(prev["max_size"] - curr["max_size"]) > 1.3:
        return False
    if curr["y0"] < prev["y0"] - 0.004 or (curr["y0"] - prev["y1"]) > 0.005:
        return False
    # Must belong to the same column (horizontal overlap & compatible left edge)
    overlap_x = min(prev["x1"], curr["x1"]) - max(prev["x0"], curr["x0"])
    if overlap_x <= 0.03 or abs(curr["x0"] - prev["x0"]) > 0.09:
        return False
    pt = prev["text"].rstrip()
    ct = curr["text"].lstrip()
    if not pt or not ct:
        return False
    if pt[-1] in ".!?…:":
        return False
    if LEADING_NEW_ITEM_RE.match(ct) or re.match(r"^[A-Z][A-Za-z0-9 &]{1,40}:", ct):
        return False
    return True


def order_and_coalesce_page_blocks(blocks: list[dict]) -> list[dict]:
    """Order page body column-by-column, anchor left-margin badges, and merge split lines."""
    if not blocks:
        return []

    badges: list[dict] = []
    content: list[dict] = []
    for b in blocks:
        is_left_badge = (
            b["x1"] <= 0.215
            and b["max_size"] <= 9.6
            and len(b["text"]) <= 80
            and any(
                other is not b
                and 0.215 < other["x0"] <= 0.36
                and (other["x0"] - b["x1"]) <= 0.145
                and other["y1"] >= b["y0"] - 0.08
                and other["y0"] <= b["y1"] + 0.08
                for other in blocks
            )
        )
        if is_left_badge:
            badges.append({**b, "is_badge": True})
        else:
            content.append(b)

    ordered = xy_cut_order(content)

    # Coalesce consecutive broken lines within the same ordered column
    merged: list[dict] = []
    for b in ordered:
        if merged and should_coalesce_blocks(merged[-1], b):
            prev = merged[-1]
            merged[-1] = {
                **prev,
                "x0": min(prev["x0"], b["x0"]),
                "y0": min(prev["y0"], b["y0"]),
                "x1": max(prev["x1"], b["x1"]),
                "y1": max(prev["y1"], b["y1"]),
                "text": prev["text"].rstrip() + " " + b["text"].lstrip(),
            }
            merged[-1]["cx"] = (merged[-1]["x0"] + merged[-1]["x1"]) * 0.5
            merged[-1]["cy"] = (merged[-1]["y0"] + merged[-1]["y1"]) * 0.5
        else:
            merged.append(b)

    if not badges:
        return merged

    # Group contiguous left-margin badges (e.g. "Activity Page" + [icon] + "AP 1.1.1")
    badges.sort(key=lambda b: (b["y0"], b["x0"]))
    badge_groups: list[list[dict]] = []
    for bg in badges:
        if badge_groups and (bg["y0"] - badge_groups[-1][-1]["y1"]) <= 0.070 and bg["x1"] <= 0.215:
            badge_groups[-1].append(bg)
        else:
            badge_groups.append([bg])

    # Coalesce each badge group into a single clean callout block anchored before its adjacent column block
    if not merged:
        return badges
    anchor_map: dict[int, list[dict]] = {}
    for grp in badge_groups:
        gy0 = grp[0]["y0"]
        gy1 = grp[-1]["y1"]
        joined_text = " ".join(g["text"].strip() for g in grp if g["text"].strip())
        grp_block = {
            "x0": min(g["x0"] for g in grp),
            "y0": gy0,
            "x1": max(g["x1"] for g in grp),
            "y1": gy1,
            "cx": (min(g["x0"] for g in grp) + max(g["x1"] for g in grp)) * 0.5,
            "cy": (gy0 + gy1) * 0.5,
            "w": max(g["x1"] for g in grp) - min(g["x0"] for g in grp),
            "h": gy1 - gy0,
            "max_size": max(g["max_size"] for g in grp),
            "text": joined_text,
            "is_badge": True,
        }
        best_idx = 0
        best_score = 1e9
        for pass_wide in (False, True):
            found_pass = False
            for idx_m, mb in enumerate(merged):
                if mb["x0"] <= 0.215 and mb["max_size"] >= 13.5:
                    continue
                if not pass_wide and mb["x0"] > 0.45:
                    continue
                if mb["y1"] >= gy0 - 0.020:
                    score = abs(mb["y0"] - gy0)
                    if score < best_score:
                        best_score = score
                        best_idx = idx_m
                        found_pass = True
            if found_pass:
                break
        anchor_map.setdefault(best_idx, []).append(grp_block)

    final_blocks: list[dict] = []
    for idx_m, mb in enumerate(merged):
        if idx_m in anchor_map:
            final_blocks.extend(anchor_map[idx_m])
        final_blocks.append(mb)
    return final_blocks


def cmd_prep(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf)
    out_manifest = Path(args.out_manifest)
    out_zip = Path(args.out_zip)
    work_dir = out_manifest.parent / "_prep_pages"
    pages_dir = work_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    pages_meta = []
    raw_pages_blocks: list[list[dict]] = []

    for pno in range(len(doc)):
        page = doc[pno]
        pw = float(page.rect.width) or 1.0
        ph = float(page.rect.height) or 1.0
        pix = page.get_pixmap(dpi=args.dpi, alpha=False)
        img_rel = f"pages/page_{pno + 1:03d}.png"
        img_abs = work_dir / img_rel
        pix.save(str(img_abs))
        pages_meta.append({
            "page": pno + 1,
            "image": img_rel,
            "width": pix.width,
            "height": pix.height,
            "pt_width": round(pw, 2),
            "pt_height": round(ph, 2),
        })
        raw_pages_blocks.append(extract_raw_page_blocks(page))

    filtered_pages_blocks = filter_headers_and_footers(raw_pages_blocks)

    chunks = []
    chunk_idx = 0
    for pno, page_blocks in enumerate(filtered_pages_blocks):
        ordered_blocks = order_and_coalesce_page_blocks(page_blocks)
        para_idx = 0
        for b in ordered_blocks:
            raw_sents = split_clean_sentences(b["text"])
            for raw_sent in raw_sents:
                for sent in split_chunk_clauses(raw_sent, max_chars=240):
                    if not HAS_ALNUM_RE.search(sent):
                        continue
                    seed_rect = [[
                        round(b["x0"], 5),
                        round(b["y0"], 5),
                        round(b["x1"], 5),
                        round(b["y1"], 5),
                    ]]
                    chunks.append({
                        "index": chunk_idx,
                        "id": f"chunk_{chunk_idx:04d}",
                        "page": pno + 1,
                        "para": para_idx,
                        "text": sent,
                        "rects": seed_rect,
                    })
                    chunk_idx += 1
            para_idx += 1

    resolve_words_for_chunks_from_pdf(doc, chunks)
    doc.close()

    with zipfile.ZipFile(str(out_zip), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pm in pages_meta:
            p_abs = work_dir / pm["image"]
            zf.write(str(p_abs), arcname=pm["image"])

    manifest_obj = {
        "schema_version": "2.6-column",
        "prepared_by": "Mushu_#46",
        "task_id": args.task_id,
        "page_count": len(pages_meta),
        "chunk_count": len(chunks),
        "pages": pages_meta,
        "chunks": chunks,
    }
    out_manifest.write_text(json.dumps(manifest_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mushu_word_kit prep] pages={len(pages_meta)} chunks={len(chunks)} words={sum(len(c.get('words',[])) for c in chunks)}")


WORD_SPOTLIGHT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=1920, height=1080" />
<title>__TITLE__ — Karaoke Word Spotlight 16:9</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 1920px;
    height: 1080px;
    overflow: hidden;
    background: #0b1020;
    font-family: "Segoe UI", "Noto Sans", -apple-system, BlinkMacSystemFont, Arial, sans-serif;
  }
  #root {
    position: relative;
    width: 1920px;
    height: 1080px;
    overflow: hidden;
    background: radial-gradient(1200px 700px at 50% 18%, #16204a 0%, #0b1020 62%, #070b18 100%);
  }
  .stage {
    position: absolute;
    left: 260px;
    top: 0;
    width: 1400px;
    height: 1080px;
    overflow: hidden;
    background: #ffffff;
    box-shadow: 0 30px 90px rgba(0, 0, 0, 0.58), 0 0 0 1px rgba(255, 255, 255, 0.08);
  }
  .sheet {
    position: absolute;
    left: 0;
    top: 0;
    width: 1400px;
    opacity: 0;
    visibility: hidden;
    will-change: transform;
  }
  .sheet.active {
    opacity: 1;
    visibility: visible;
  }
  .sheet img {
    display: block;
    width: 1400px;
    height: auto;
    user-select: none;
  }
  .word-spot {
    position: absolute;
    background: rgba(253, 224, 71, 0.64);
    mix-blend-mode: multiply;
    border-radius: 6px;
    box-shadow: 0 0 0 2.5px rgba(245, 158, 11, 0.52);
    pointer-events: none;
    opacity: 0;
  }
  .progress {
    position: absolute;
    left: 0;
    bottom: 0;
    width: 1920px;
    height: 6px;
    background: rgba(255, 255, 255, 0.12);
    z-index: 20;
  }
  .progress > i {
    display: block;
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #facc15, #f59e0b);
  }
  .pagelabel {
    position: absolute;
    right: 28px;
    bottom: 22px;
    z-index: 21;
    color: #e2e8f0;
    background: rgba(11, 16, 32, 0.76);
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 20px;
    letter-spacing: 0.08em;
  }
  .brand {
    position: absolute;
    left: 28px;
    bottom: 22px;
    z-index: 21;
    color: rgba(226, 232, 240, 0.78);
    background: rgba(11, 16, 32, 0.58);
    border: 1px solid rgba(255, 255, 255, 0.10);
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 18px;
    letter-spacing: 0.12em;
  }
</style>
</head>
<body>
<div id="root" data-composition-id="main" data-start="0" data-duration="__DURATION__" data-width="1920" data-height="1080">
  <div class="stage" id="stage">
__SHEETS_HTML__
  </div>
  <div class="progress"><i id="prog"></i></div>
  <div class="brand">MUSHU &middot; WORD KARAOKE 16:9</div>
  <div class="pagelabel" id="pagelabel">Page 1 / __PAGE_COUNT__</div>
</div>
<script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
<script>
(function () {
  "use strict";
  var TOTAL_S = __DURATION__;
  var PAGE_COUNT = __PAGE_COUNT__;
  var FRAME_H = 1080;
  var PAD_X = 4.5;
  var PAD_Y = 3.0;
  var PAN_DUR = 0.34;
  var SHEET_H = __SHEET_H_JSON__;
  var WORDS = __WORDS_JSON__;
  var CAM_ANCHORS = __CAM_ANCHORS_JSON__;

  var prog = document.getElementById("prog");
  var pagelabel = document.getElementById("pagelabel");
  var sheets = {};
  var spots1 = {};
  var spots2 = {};

  Array.prototype.forEach.call(document.querySelectorAll(".sheet"), function (el) {
    var p = parseInt(el.getAttribute("data-page"), 10);
    sheets[p] = el;
    spots1[p] = el.querySelector(".spot-a");
    spots2[p] = el.querySelector(".spot-b");
  });

  var curPage = -1;

  function findWordIdx(t) {
    var lo = 0, hi = WORDS.length - 1, ans = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (WORDS[mid].s <= t) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    if (ans >= 0 && t < WORDS[ans].e) return ans;
    return -1;
  }

  function findAnchorIdx(t) {
    if (!CAM_ANCHORS.length) return -1;
    var lo = 0, hi = CAM_ANCHORS.length - 1, ans = 0;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (CAM_ANCHORS[mid].s <= t) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    return ans;
  }

  function showPage(p) {
    if (p === curPage) return;
    if (curPage > 0 && spots1[curPage]) {
      spots1[curPage].style.opacity = "0";
      spots2[curPage].style.opacity = "0";
    }
    Object.keys(sheets).forEach(function (k) { sheets[k].classList.remove("active"); });
    if (sheets[p]) sheets[p].classList.add("active");
    curPage = p;
    pagelabel.textContent = "Page " + p + " / " + PAGE_COUNT;
  }

  function placeSpot(el, boxPx) {
    if (!boxPx) {
      el.style.opacity = "0";
      return;
    }
    el.style.left = (boxPx[0] - PAD_X).toFixed(2) + "px";
    el.style.top = (boxPx[1] - PAD_Y).toFixed(2) + "px";
    el.style.width = (boxPx[2] + PAD_X * 2).toFixed(2) + "px";
    el.style.height = (boxPx[3] + PAD_Y * 2).toFixed(2) + "px";
    el.style.opacity = "1";
  }

  function update(t) {
    t = Math.max(0, Math.min(TOTAL_S, t));
    var ai = findAnchorIdx(t);
    var page = ai >= 0 ? CAM_ANCHORS[ai].p : 1;
    showPage(page);

    var sh = sheets[page];
    if (sh && ai >= 0) {
      var anc = CAM_ANCHORS[ai];
      var u = Math.max(0, Math.min(1, (t - anc.s) / PAN_DUR));
      var ez = u * u * (3 - 2 * u);
      var camY = anc.sy + (anc.ty - anc.sy) * ez;
      sh.style.transform = "translate3d(0, " + (-camY).toFixed(2) + "px, 0)";
    }

    var wi = findWordIdx(t);
    if (wi >= 0 && WORDS[wi].p === page) {
      placeSpot(spots1[page], WORDS[wi].b);
      placeSpot(spots2[page], WORDS[wi].b2 || null);
    } else if (spots1[page]) {
      spots1[page].style.opacity = "0";
      spots2[page].style.opacity = "0";
    }

    prog.style.width = (t / TOTAL_S * 100).toFixed(3) + "%";
  }

  window.__timelines = window.__timelines || {};
  if (window.gsap) {
    var tl = gsap.timeline({ paused: true });
    var proxy = { v: 0 };
    tl.to(proxy, {
      v: 1,
      duration: TOTAL_S,
      ease: "none",
      onUpdate: function () { update(proxy.v * TOTAL_S); }
    });
    tl.eventCallback("onUpdate", function () { update(proxy.v * TOTAL_S); });
    window.__timelines["main"] = tl;
    tl.seek(0);
  } else {
    var shim = {
      _t: 0,
      duration: function () { return TOTAL_S; },
      time: function () { return this._t; },
      seek: function (t) { this._t = Math.max(0, Math.min(TOTAL_S, t)); update(this._t); return this; },
      play: function () { return this; },
      pause: function () { return this; },
      eventCallback: function () { return this; }
    };
    window.__timelines["main"] = shim;
    shim.seek(0);
  }
  update(0);
})();
</script>
</body>
</html>
"""


def pair_word_boxes_and_timings(box_words: list[dict], timed_words: list[dict], chunk_start: float, speech_end: float) -> list[dict]:
    """Pair PDF word bounding boxes `{w, r, r2?}` with acoustic word timings `{w, s, e}`."""
    spoken_boxes = [bw for bw in box_words if HAS_ALNUM_RE.search(bw.get("w") or "")]
    if not spoken_boxes:
        return []

    if not timed_words:
        # Proportional character-weighted fallback if timings.json lacked word timings
        tot_chars = sum(max(1, len(norm_tok(bw["w"]))) for bw in spoken_boxes)
        dur = max(0.08, speech_end - chunk_start)
        cur_t = chunk_start
        synth = []
        for bw in spoken_boxes:
            nt = cur_t + dur * (max(1, len(norm_tok(bw["w"]))) / tot_chars)
            synth.append({**bw, "s": round(cur_t, 3), "e": round(nt, 3)})
            cur_t = nt
        return synth

    if len(spoken_boxes) == len(timed_words):
        return [{**bw, "s": tw["s"], "e": tw["e"]} for bw, tw in zip(spoken_boxes, timed_words)]

    sm = difflib.SequenceMatcher(
        a=[norm_tok(bw["w"]) or bw["w"] for bw in spoken_boxes],
        b=[norm_tok(tw["w"]) or tw["w"] for tw in timed_words],
        autojunk=False,
    )
    out: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out.append({**spoken_boxes[i1 + k], "s": timed_words[j1 + k]["s"], "e": timed_words[j1 + k]["e"]})
        elif (i2 - i1) > 0 and (j2 - j1) > 0:
            t_start = timed_words[j1]["s"]
            t_end = timed_words[j2 - 1]["e"]
            sub_boxes = spoken_boxes[i1:i2]
            tot_c = sum(max(1, len(norm_tok(b["w"]))) for b in sub_boxes)
            cur_t = t_start
            for b in sub_boxes:
                nt = cur_t + max(0.04, t_end - t_start) * (max(1, len(norm_tok(b["w"]))) / tot_c)
                out.append({**b, "s": round(cur_t, 3), "e": round(nt, 3)})
                cur_t = nt
        elif (i2 - i1) > 0:
            base_t = out[-1]["e"] if out else chunk_start
            for b in spoken_boxes[i1:i2]:
                out.append({**b, "s": round(base_t, 3), "e": round(base_t + 0.08, 3)})
                base_t += 0.08
    return out


def cmd_html(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    timings = json.loads(Path(args.timings).read_text(encoding="utf-8"))

    chunks = manifest.get("chunks") or []
    need_pdf_words = any("words" not in c or not c.get("words") for c in chunks)
    if need_pdf_words and args.pdf and Path(args.pdf).is_file():
        doc = fitz.open(str(args.pdf))
        resolve_words_for_chunks_from_pdf(doc, chunks)
        doc.close()

    pages = manifest.get("pages") or []
    stage_w = 1400.0
    sheet_h_map: dict[str, float] = {}
    sheets_html_lines: list[str] = []

    for pm in pages:
        pnum = int(pm["page"])
        pw = float(pm.get("width") or 1870)
        ph = float(pm.get("height") or 2420)
        sh_px = round(stage_w * (ph / pw), 2)
        sheet_h_map[str(pnum)] = sh_px
        img_src = pm.get("image") or f"pages/page_{pnum:03d}.png"
        sheets_html_lines.append(
            f'    <div class="sheet" data-page="{pnum}" data-h="{sh_px}" style="height:{sh_px}px">'
            f'<img src="{img_src}" alt="Page {pnum}" draggable="false">'
            f'<div class="word-spot spot-a"></div><div class="word-spot spot-b"></div></div>'
        )

    chunk_by_idx = {int(c["index"]): c for c in chunks}
    words_timeline: list[dict] = []
    cam_anchors: list[dict] = []

    for it in timings.get("items") or []:
        cidx = int(it["index"])
        pnum = int(it.get("page") or 1)
        sh_px = sheet_h_map.get(str(pnum), 1811.76)
        ch = chunk_by_idx.get(cidx) or it
        box_words = ch.get("words") or []
        timed_words = it.get("words") or []
        c_start = float(it.get("start_s") or 0.0)
        c_speech_end = float(it.get("speech_end_s") or it.get("end_s") or (c_start + 0.5))

        paired = pair_word_boxes_and_timings(box_words, timed_words, c_start, c_speech_end)
        if not paired and (it.get("rects") or ch.get("rects")):
            # Fallback if neither source.pdf nor word boxes were present: subdivide line rects by words
            rects = it.get("rects") or ch.get("rects")
            toks = [t for t in split_visual_tokens(it.get("text") or "") if HAS_ALNUM_RE.search(t)]
            if toks and rects:
                tot_w = sum(max(0.01, r[2] - r[0]) for r in rects)
                tot_c = sum(max(1, len(norm_tok(t))) for t in toks)
                synth_boxes = []
                r_idx = 0
                cur_x = rects[0][0]
                for tok in toks:
                    need_w = tot_w * (max(1, len(norm_tok(tok))) / tot_c)
                    while r_idx < len(rects) - 1 and (cur_x + need_w * 0.45) > rects[r_idx][2]:
                        r_idx += 1
                        cur_x = rects[r_idx][0]
                    rr = rects[r_idx]
                    x1 = min(rr[2], cur_x + need_w)
                    synth_boxes.append({"w": tok, "r": [cur_x, rr[1], max(cur_x + 0.012, x1), rr[3]]})
                    cur_x = x1
                paired = pair_word_boxes_and_timings(synth_boxes, timed_words, c_start, c_speech_end)

        for pw_item in paired:
            r = pw_item["r"]
            b_px = [
                round(r[0] * stage_w, 2),
                round(r[1] * sh_px, 2),
                round(max(12.0, (r[2] - r[0]) * stage_w), 2),
                round(max(14.0, (r[3] - r[1]) * sh_px), 2),
            ]
            entry: dict = {
                "p": pnum,
                "s": round(float(pw_item["s"]), 3),
                "e": round(float(pw_item["e"]), 3),
                "b": b_px,
            }
            if pw_item.get("r2"):
                r2 = pw_item["r2"]
                entry["b2"] = [
                    round(r2[0] * stage_w, 2),
                    round(r2[1] * sh_px, 2),
                    round(max(12.0, (r2[2] - r2[0]) * stage_w), 2),
                    round(max(14.0, (r2[3] - r2[1]) * sh_px), 2),
                ]
            words_timeline.append(entry)

            w_cy = round(b_px[1] + b_px[3] * 0.5, 2)
            if not cam_anchors or cam_anchors[-1]["p"] != pnum or abs(cam_anchors[-1]["cy"] - w_cy) >= 34.0:
                cam_anchors.append({"p": pnum, "s": entry["s"], "cy": w_cy})

    # Precompute C0-continuous deterministic camera start/target offsets
    frame_h = 1080.0
    focus_y = 0.42 * frame_h
    pan_dur = 0.34
    for idx_a, anc in enumerate(cam_anchors):
        max_scroll = max(0.0, sheet_h_map.get(str(anc["p"]), 1811.76) - frame_h)
        target_y = round(max(0.0, min(max_scroll, anc["cy"] - focus_y)), 2)
        if idx_a == 0 or cam_anchors[idx_a - 1]["p"] != anc["p"]:
            start_y = target_y
        else:
            prev = cam_anchors[idx_a - 1]
            u_p = max(0.0, min(1.0, (anc["s"] - prev["s"]) / pan_dur))
            ez_p = u_p * u_p * (3.0 - 2.0 * u_p)
            start_y = round(prev["sy"] + (prev["ty"] - prev["sy"]) * ez_p, 2)
        anc["sy"] = start_y
        anc["ty"] = target_y

    # Enforce strict monotonic non-overlapping word intervals for O(log N) binary search
    words_timeline.sort(key=lambda x: (x["s"], x["e"]))
    for idx_w in range(len(words_timeline) - 1):
        if words_timeline[idx_w]["e"] > words_timeline[idx_w + 1]["s"]:
            words_timeline[idx_w]["e"] = words_timeline[idx_w + 1]["s"]

    total_s = float(timings.get("total_s") or 1.0)
    html_out = (
        WORD_SPOTLIGHT_HTML_TEMPLATE
        .replace("__TITLE__", str(args.task_id))
        .replace("__DURATION__", f"{total_s:.4f}")
        .replace("__PAGE_COUNT__", str(max(1, len(pages))))
        .replace("__SHEETS_HTML__", "\n".join(sheets_html_lines))
        .replace("__SHEET_H_JSON__", json.dumps(sheet_h_map))
        .replace("__WORDS_JSON__", json.dumps(words_timeline, separators=(",", ":")))
        .replace("__CAM_ANCHORS_JSON__", json.dumps(cam_anchors, separators=(",", ":")))
    )

    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_out, encoding="utf-8")
    print(
        f"[mushu_word_kit html] Built Word-Spotlight HTML ({out_path.stat().st_size} bytes): "
        f"words={len(words_timeline)}, cam_anchors={len(cam_anchors)}, total_s={total_s:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prep")
    p_prep.add_argument("--pdf", required=True)
    p_prep.add_argument("--out-manifest", required=True)
    p_prep.add_argument("--out-zip", required=True)
    p_prep.add_argument("--task-id", default="KARAOKE_PREP")
    p_prep.add_argument("--dpi", type=int, default=220)

    p_html = sub.add_parser("html")
    p_html.add_argument("--manifest", required=True)
    p_html.add_argument("--timings", required=True)
    p_html.add_argument("--pdf", default="")
    p_html.add_argument("--out-html", required=True)
    p_html.add_argument("--task-id", default="KARAOKE_HTML")

    args = ap.parse_args()
    if args.cmd == "prep":
        cmd_prep(args)
    elif args.cmd == "html":
        cmd_html(args)


if __name__ == "__main__":
    main()
