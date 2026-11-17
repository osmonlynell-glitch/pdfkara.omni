#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from gcommon import (
    _bridge_post,
    download_drive_file,
    ensure_folder,
    extract_drive_id,
    file_view_url,
    folder_url,
    get_services,
    list_folder_files,
    read_task_row,
    resolve_project_folder,
    update_cells,
    upsert_file,
)

SR = 24000
FPS = 30
LEAD_IN_S = 0.30
CHUNK_GAP_S = 0.50  # Khoảng nghỉ chuẩn 0.5 giây giữa các câu
TAIL_S = 0.50

HAS_ALNUM_RE = re.compile(r"[A-Za-z0-9\u00C0-\u1EF9]")
COMPOUND_SPLIT_RE = re.compile(r"(?<=[A-Za-z0-9\u00C0-\u1EF9])([—–/])(?=[A-Za-z0-9\u00C0-\u1EF9])")

EN_ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
}
EN_TENS = {
    2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
    6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety",
}
VI_DIGITS = {
    "0": "khong", "1": "mot", "2": "hai", "3": "ba", "4": "bon",
    "5": "nam", "6": "sau", "7": "bay", "8": "tam", "9": "chin",
}


def detect_lang(text: str) -> str:
    vi_chars = set("àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ")
    low = text.lower()
    if any(ch in vi_chars for ch in low):
        return "vi"
    return "en"


def split_spoken_visual_tokens(text: str) -> list[str]:
    expanded = COMPOUND_SPLIT_RE.sub(r"\1 ", (text or "").strip())
    return [t for t in expanded.split() if t.strip() and HAS_ALNUM_RE.search(t)]


def _num_to_en_words(num_str: str) -> list[str]:
    if len(num_str) > 4:
        return [EN_ONES[int(d)] for d in num_str]
    val = int(num_str)
    if val in EN_ONES:
        return [EN_ONES[val]]
    if val < 100:
        t, r = divmod(val, 10)
        return [EN_TENS[t]] + ([EN_ONES[r]] if r else [])
    if val < 1000:
        h, rem = divmod(val, 100)
        out = [EN_ONES[h], "hundred"]
        if rem:
            out.extend(_num_to_en_words(str(rem)))
        return out
    # 1000..9999 (e.g. years like 2026 -> twenty twenty six or two thousand...)
    if 1900 <= val <= 2099:
        hi, lo = divmod(val, 100)
        out = _num_to_en_words(str(hi))
        if lo == 0:
            out.append("hundred")
        elif lo < 10:
            out.extend(["oh", EN_ONES[lo]])
        else:
            out.extend(_num_to_en_words(str(lo)))
        return out
    th, rem = divmod(val, 1000)
    out = _num_to_en_words(str(th)) + ["thousand"]
    if rem:
        out.extend(_num_to_en_words(str(rem)))
    return out


def _num_to_vi_words(num_str: str) -> list[str]:
    if len(num_str) == 1:
        return [VI_DIGITS[num_str]]
    if len(num_str) == 2:
        val = int(num_str)
        if val == 10:
            return ["muoi"]
        if val < 20:
            return ["muoi", "lam" if val == 15 else VI_DIGITS[num_str[1]]]
        t, r = divmod(val, 10)
        out = [VI_DIGITS[str(t)], "muoi"]
        if r == 1:
            out.append("mot")
        elif r == 5:
            out.append("lam")
        elif r > 0:
            out.append(VI_DIGITS[str(r)])
        return out
    return [VI_DIGITS[d] for d in num_str]


def visual_token_to_uroman_words(tok: str, lang: str, valid_chars: set[str]) -> list[str]:
    s = unicodedata.normalize("NFKC", tok).replace("đ", "d").replace("Đ", "d")
    s = s.replace("’", "'").replace("‘", "'")
    # Strip thousands-separator commas inside numbers (e.g. "1,500" -> "1500")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # Split decimal numbers cleanly: e.g. "1.1.1" -> ["1", ".", "1", ".", "1"]
    pieces = re.findall(r"[A-Za-z\u00C0-\u1EF9']+|(?<=\d)\.(?=\d)|\d+|%", s)
    ord_map = {"1st": "first", "2nd": "second", "3rd": "third", "5th": "fifth", "8th": "eighth", "9th": "ninth", "12th": "twelfth"}
    low_clean = re.sub(r"[^\w]+", "", s.lower())
    if lang == "en" and low_clean in ord_map:
        return [ord_map[low_clean]]
    uwords: list[str] = []
    for p in pieces:
        if p == ".":
            cand = ["cham"] if lang == "vi" else ["point"]
        elif p == "%":
            cand = ["phan", "tram"] if lang == "vi" else ["percent"]
        elif p.isdigit():
            cand = _num_to_vi_words(p) if lang == "vi" else _num_to_en_words(p)
        else:
            ascii_s = unicodedata.normalize("NFKD", p).encode("ascii", "ignore").decode("ascii").lower()
            cleaned = "".join(ch for ch in ascii_s if ch in valid_chars).strip("'")
            cand = [cleaned] if cleaned else []
        for cw in cand:
            flt = "".join(ch for ch in cw if ch in valid_chars)
            if flt:
                uwords.append(flt)
    return uwords


def proportional_word_timings(vtoks: list[str], start_s: float, speech_end_s: float, end_s: float) -> list[dict]:
    if not vtoks:
        return []
    dur = max(0.08 * len(vtoks), speech_end_s - start_s)
    weights = [max(1, len(re.sub(r"[^\w]+", "", t))) for t in vtoks]
    tot_w = sum(weights)
    cur = start_s
    raw_words = []
    for t, w in zip(vtoks, weights):
        nxt = cur + dur * (w / tot_w)
        raw_words.append({"w": t, "s": cur, "e": nxt})
        cur = nxt
    return postprocess_spotlight_words(raw_words, start_s, speech_end_s, end_s)


def postprocess_spotlight_words(raw_words: list[dict], start_s: float, speech_end_s: float, end_s: float) -> list[dict]:
    if not raw_words:
        return []
    out = []
    for rw in raw_words:
        ws = max(start_s, float(rw["s"]))
        we = max(ws + 0.07, float(rw["e"]))
        out.append({"w": rw["w"], "s": ws, "e": we})

    for i in range(1, len(out)):
        if out[i]["s"] < out[i - 1]["s"] + 0.04:
            out[i]["s"] = out[i - 1]["s"] + 0.04
        if out[i]["e"] < out[i]["s"] + 0.07:
            out[i]["e"] = out[i]["s"] + 0.07

    # Seamless Word-Spotlight hop across normal inter-word articulation (< 260ms),
    # while preserving genuine comma / clause breath pauses (> 260ms).
    for i in range(len(out) - 1):
        gap = out[i + 1]["s"] - out[i]["e"]
        if gap <= 0.26:
            out[i]["e"] = out[i + 1]["s"]
        else:
            out[i]["e"] = min(out[i + 1]["s"] - 0.04, out[i]["e"] + 0.08)

    out[-1]["e"] = min(
        max(start_s + 0.08, end_s - 0.05),
        max(out[-1]["s"] + 0.08, min(speech_end_s, out[-1]["e"] + 0.16)),
    )

    for rw in out:
        rw["s"] = round(rw["s"], 4)
        rw["e"] = round(max(rw["s"] + 0.04, rw["e"]), 4)
    return out


class MmsWordAligner:
    def __init__(self):
        import torch
        import torchaudio

        self.torch = torch
        self.torchaudio = torchaudio
        torch.set_num_threads(max(1, os.cpu_count() or 4))
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model()
        self.model.eval()
        self.tokenizer = self.bundle.get_tokenizer()
        self.aligner = self.bundle.get_aligner()
        self.target_sr = int(self.bundle.sample_rate)  # 16000
        self.valid_chars = set(self.bundle.get_labels()) - {"-", "*"}

    def align_chunk(
        self,
        pcm_mono: np.ndarray,
        pcm_sr: int,
        text: str,
        start_s: float,
        speech_end_s: float,
        end_s: float,
    ) -> list[dict]:
        vtoks = split_spoken_visual_tokens(text)
        if not vtoks:
            return []
        if pcm_mono.size < int(0.12 * pcm_sr):
            return proportional_word_timings(vtoks, start_s, speech_end_s, end_s)

        lang = detect_lang(text)
        uroman_flat: list[str] = []
        vtok_slices: list[tuple[int, int]] = []
        for vt in vtoks:
            uw = visual_token_to_uroman_words(vt, lang, self.valid_chars)
            s_idx = len(uroman_flat)
            uroman_flat.extend(uw)
            vtok_slices.append((s_idx, len(uroman_flat)))

        if not uroman_flat:
            return proportional_word_timings(vtoks, start_s, speech_end_s, end_s)

        try:
            torch = self.torch
            wav_t = torch.from_numpy(np.ascontiguousarray(pcm_mono, dtype=np.float32)).unsqueeze(0)
            if pcm_sr != self.target_sr:
                wav_t = self.torchaudio.functional.resample(wav_t, pcm_sr, self.target_sr)

            tokenized = self.tokenizer(uroman_flat)
            flat_ids = [tid for wt in tokenized for tid in wt]
            repeats = sum(1 for k in range(1, len(flat_ids)) if flat_ids[k] == flat_ids[k - 1])

            with torch.inference_mode():
                emission, _ = self.model(wav_t)
                num_frames = int(emission.size(1))
                if num_frames <= len(flat_ids) + repeats + 1:
                    return proportional_word_timings(vtoks, start_s, speech_end_s, end_s)
                spans_per_uword = self.aligner(emission[0], tokenized)

            sec_per_frame = (wav_t.size(1) / float(num_frames)) / float(self.target_sr)
            raw_words: list[dict] = []
            for vt, (u0, u1) in zip(vtoks, vtok_slices):
                if u1 > u0 and spans_per_uword[u0] and spans_per_uword[u1 - 1]:
                    rel_s = spans_per_uword[u0][0].start * sec_per_frame
                    rel_e = spans_per_uword[u1 - 1][-1].end * sec_per_frame
                    raw_words.append({"w": vt, "s": start_s + rel_s, "e": start_s + rel_e})
                else:
                    prev_e = raw_words[-1]["e"] if raw_words else start_s
                    raw_words.append({"w": vt, "s": prev_e, "e": prev_e + 0.08})

            return postprocess_spotlight_words(raw_words, start_s, speech_end_s, end_s)
        except Exception as e:
            print(f"[MMS_FA warn] Fallback proportional timing on chunk ({text[:60]!r}): {e}", flush=True)
            return proportional_word_timings(vtoks, start_s, speech_end_s, end_s)


def populate_word_alignments_from_wav(wav_path: Path, timing_items: list[dict]) -> int:
    t0 = time.monotonic()
    full_pcm, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if full_pcm.ndim > 1:
        full_pcm = full_pcm.mean(axis=1)

    aligner = MmsWordAligner()
    total_words = 0
    for item in timing_items:
        s_s = float(item["start_s"])
        se_s = float(item.get("speech_end_s") or item["end_s"])
        e_s = float(item["end_s"])
        i0 = max(0, int(round(s_s * sr)))
        i1 = min(len(full_pcm), max(i0 + 1, int(round(se_s * sr))))
        clip = full_pcm[i0:i1]
        words = aligner.align_chunk(clip, sr, str(item.get("text") or ""), s_s, se_s, e_s)
        item["words"] = words
        total_words += len(words)

    elapsed = time.monotonic() - t0
    sample_words = timing_items[2].get("words", [])[:6] if len(timing_items) > 2 else (timing_items[0].get("words", [])[:6] if timing_items else [])
    print(
        f"[MMS_FA] Aligned {total_words} words across {len(timing_items)} chunks in {elapsed:.1f}s (20ms CTC precision). Sample: {sample_words}",
        flush=True,
    )
    return total_words


def notify_bridge_row_html_ready(row: int):
    if os.environ.get("BRIDGE_UPLOAD_URL", "").strip() and os.environ.get("BRIDGE_UPLOAD_SECRET", "").strip():
        try:
            _bridge_post({"action": "reset_row_html_lock", "row": int(row)})
        except Exception as e:
            print(f"[Bridge notify warn] row {row}: {e}", flush=True)


def loudnorm_wav(in_wav: Path, out_wav: Path, target_lufs: float = -14.0):
    probe = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(in_wav),
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stderr = probe.stderr or ""
    m_start = stderr.rfind("{")
    m_end = stderr.rfind("}")
    if m_start >= 0 and m_end > m_start:
        stats = json.loads(stderr[m_start : m_end + 1])
        af = (
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
            f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
            f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
            f"offset={stats['target_offset']}:linear=true"
        )
    else:
        af = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"

    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-y", "-i", str(in_wav),
            "-af", af, "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le",
            str(out_wav),
        ],
        check=True,
    )


def list_all_chunks_across_folders(drive, proj_id: str) -> tuple[str, dict]:
    """Lấy folder chunks chính và gộp danh sách file từ mọi folder tên 'chunks' (chống race condition khi 10 luồng chạy song song)."""
    q = (
        f"'{proj_id}' in parents and name = 'chunks' "
        "and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    res = drive.files().list(
        q=q, fields="files(id,createdTime)", pageSize=20, supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute()
    folders = sorted(res.get("files") or [], key=lambda x: x.get("createdTime") or "")
    if not folders:
        primary_id = ensure_folder(drive, proj_id, "chunks")
        return primary_id, {}

    primary_id = folders[0]["id"]
    merged = {}
    for fld in folders:
        files_map = list_folder_files(drive, fld["id"])
        for fname, fmeta in files_map.items():
            if fname not in merged or int(fmeta.get("size") or 0) > int(merged[fname].get("size") or 0):
                merged[fname] = fmeta
    return primary_id, merged


def run_shard_synthesis(
    drive,
    sheets,
    args,
    proj_id: str,
    proj_name: str,
    chunks_spec: list,
    chunks_drive_id: str,
    existing_chunk_files: dict,
    t_job_start: float,
):
    shard_missing = []
    for item in chunks_spec:
        idx = int(item["index"])
        if args.mode == "shard" and (idx % args.num_shards) != args.shard_id:
            continue
        cname = f"chunk_{idx:04d}.wav"
        fmeta = existing_chunk_files.get(cname)
        if not fmeta or int(fmeta.get("size") or 0) < 512:
            shard_missing.append(idx)

    print(
        f"[Shard {args.shard_id}/{args.num_shards}] Row {args.row} ({proj_name}): "
        f"shard_remaining={len(shard_missing)} (total_chunks={len(chunks_spec)})",
        flush=True,
    )
    if not shard_missing:
        return

    import torch
    from omnivoice import OmniVoice

    torch.set_num_threads(max(1, os.cpu_count() or 4))
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32, asr_device="cpu"
    )
    voice_prompt = model.create_voice_clone_prompt(ref_audio=args.ref_audio, ref_text=None)
    soft_deadline_s = args.max_minutes * 60.0

    chunks_local_dir = Path("work") / f"row_{args.row}" / f"shard_{args.shard_id}"
    chunks_local_dir.mkdir(parents=True, exist_ok=True)
    spec_by_idx = {int(c["index"]): c for c in chunks_spec}

    for pos, idx in enumerate(shard_missing):
        c_item = spec_by_idx[idx]
        text = str(c_item["text"]).strip()
        lang = c_item.get("lang") or detect_lang(text)
        t0 = time.monotonic()

        if not HAS_ALNUM_RE.search(text):
            audio = np.zeros(int(0.15 * SR), dtype=np.float32)
        else:
            try:
                raw_gen = model.generate(
                    text=text,
                    language=lang,
                    voice_clone_prompt=voice_prompt,
                    speed=args.speed,
                    num_step=args.num_step,
                )
                audio = raw_gen[0].astype(np.float32) if len(raw_gen) > 0 else np.zeros(0, dtype=np.float32)
            except Exception as gen_err:
                print(f"[Shard {args.shard_id}] Warn empty/failed synth on idx={idx} ({text!r}): {gen_err}", flush=True)
                audio = np.zeros(int(0.15 * SR), dtype=np.float32)

        if audio.size == 0:
            audio = np.zeros(int(0.15 * SR), dtype=np.float32)

        peak = float(np.max(np.abs(audio))) if audio.size > 0 else 1.0
        if peak > 0.96:
            audio = audio / peak * 0.96

        cname = f"chunk_{idx:04d}.wav"
        local_chunk = chunks_local_dir / cname
        sf.write(str(local_chunk), audio, SR, subtype="PCM_16")

        fid = upsert_file(drive, chunks_drive_id, str(local_chunk), cname, "audio/wav")
        existing_chunk_files[cname] = {"id": fid, "name": cname, "size": local_chunk.stat().st_size}

        gen_s = time.monotonic() - t0
        dur_s = len(audio) / SR
        elapsed_min = (time.monotonic() - t_job_start) / 60.0
        print(
            f"[Shard {args.shard_id} | chunk {idx + 1}/{len(chunks_spec)}] "
            f"{len(text)} chars -> {dur_s:.2f}s audio in {gen_s:.1f}s (uploaded {cname}, elapsed={elapsed_min:.1f}m)",
            flush=True,
        )

        if args.mode != "shard" or args.shard_id == 0:
            try:
                _, latest_map = list_all_chunks_across_folders(drive, proj_id)
                done_cnt = sum(
                    1 for k, v in latest_map.items()
                    if k.startswith("chunk_") and k.endswith(".wav") and int(v.get("size") or 0) >= 512
                )
                tot_cnt = len(chunks_spec)
                pct = int(round(done_cnt * 100.0 / max(1, tot_cnt)))
                update_cells(sheets, {
                    f"B{args.row}": "Voiceover",
                    f"G{args.row}": f"🎙 Đang đọc (10 luồng song song): {done_cnt}/{tot_cnt} câu ({pct}%)",
                })
            except Exception:
                pass

        if pos + 1 < len(shard_missing) and (time.monotonic() - t_job_start) >= soft_deadline_s:
            print(f"[Shard {args.shard_id}] Reached soft timeout ({elapsed_min:.1f}m); yielding to merge/continuation.")
            return


def run_merge_final(
    drive,
    sheets,
    args,
    row_info: dict,
    proj_id: str,
    proj_name: str,
    chunks_spec: list,
):
    _, all_chunk_files = list_all_chunks_across_folders(drive, proj_id)
    missing = []
    for item in chunks_spec:
        idx = int(item["index"])
        cname = f"chunk_{idx:04d}.wav"
        fmeta = all_chunk_files.get(cname)
        if not fmeta or int(fmeta.get("size") or 0) < 512:
            missing.append(idx)

    if missing:
        ckpt = {
            "needs_continuation": True,
            "row": args.row,
            "pdf_link": row_info["pdf_link"] or "resume",
            "project_folder_url": folder_url(proj_id),
            "next_chunk": missing[0],
            "completed_chunks": len(chunks_spec) - len(missing),
            "total_chunks": len(chunks_spec),
        }
        Path(args.checkpoint_state).parent.mkdir(parents=True, exist_ok=True)
        Path(args.checkpoint_state).write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
        upsert_file(drive, proj_id, args.checkpoint_state, "checkpoint.json", "application/json")
        done_cnt = len(chunks_spec) - len(missing)
        update_cells(sheets, {
            f"G{args.row}": f"🔄 Đã xong {done_cnt}/{len(chunks_spec)} câu — đang nối ca đọc nốt {len(missing)} câu còn lại..."
        })
        print(f"[Job 11 Merge] Still missing {len(missing)} chunks ({missing[:10]}...); triggered continuation dispatch.")
        return

    print(
        f"[Job 11 Merge] All {len(chunks_spec)} chunks verified on Drive! "
        f"Merging ({args.gap_sec:.2f}s spacing) + MMS_FA word alignment...",
        flush=True,
    )
    update_cells(sheets, {
        f"G{args.row}": f"🎚 Tiến trình #11 đang ghép {len(chunks_spec)} câu (nghỉ {args.gap_sec:.1f}s), chuẩn hóa -14 LUFS & căn chỉnh nhịp từng chữ..."
    })

    work_dir = Path("work") / f"row_{args.row}"
    chunks_local_dir = work_dir / "merge_chunks"
    chunks_local_dir.mkdir(parents=True, exist_ok=True)

    pieces = [np.zeros(int(round(LEAD_IN_S * SR)), dtype=np.float32)]
    cursor_samples = len(pieces[0])
    timing_items = []
    gap_samples = int(round(args.gap_sec * SR))
    tail_samples = int(round(TAIL_S * SR))

    for pos, item in enumerate(chunks_spec):
        idx = int(item["index"])
        cname = f"chunk_{idx:04d}.wav"
        local_chunk = chunks_local_dir / cname
        if not local_chunk.is_file():
            download_drive_file(drive, all_chunk_files[cname]["id"], str(local_chunk))

        pcm, sr = sf.read(str(local_chunk), dtype="float32", always_2d=False)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        if sr != SR:
            raise RuntimeError(f"Unexpected sample rate {sr} in {cname}; expected {SR}")

        start_s = cursor_samples / SR
        speech_end_s = (cursor_samples + len(pcm)) / SR
        pieces.append(pcm)
        cursor_samples += len(pcm)

        if pos < len(chunks_spec) - 1:
            pieces.append(np.zeros(gap_samples, dtype=np.float32))
            cursor_samples += gap_samples
        else:
            pieces.append(np.zeros(tail_samples, dtype=np.float32))
            cursor_samples += tail_samples

        end_s = cursor_samples / SR
        timing_items.append({
            "index": idx,
            "page": int(item.get("page") or 1),
            "text": item["text"],
            "rects": item.get("rects") or [],
            "start_s": round(start_s, 4),
            "speech_end_s": round(speech_end_s, 4),
            "end_s": round(end_s, 4),
        })

    total_s_raw = cursor_samples / SR
    total_s = math.ceil(total_s_raw * FPS) / FPS
    pad_end = max(0, int(round(total_s * SR)) - cursor_samples)
    if pad_end > 0:
        pieces.append(np.zeros(pad_end, dtype=np.float32))
    if timing_items:
        timing_items[-1]["end_s"] = round(total_s, 4)

    raw_concat = work_dir / "voiceover_raw.wav"
    final_wav = work_dir / "voiceover_final.wav"
    sf.write(str(raw_concat), np.concatenate(pieces), SR, subtype="FLOAT")
    loudnorm_wav(raw_concat, final_wav, target_lufs=-14.0)

    total_words = populate_word_alignments_from_wav(final_wav, timing_items)

    timings_obj = {
        "schema_version": "2.5-word",
        "alignment_engine": "torchaudio_mms_fa_ctc_20ms",
        "fps": FPS,
        "sample_rate": SR,
        "chunk_gap_s": args.gap_sec,
        "total_s": round(total_s, 4),
        "chunk_count": len(timing_items),
        "word_count": total_words,
        "voice": "Peter (Peter_voice_mau2_tre_20261116.wav)",
        "items": timing_items,
    }
    timings_path = work_dir / "timings.json"
    timings_path.write_text(json.dumps(timings_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    Path(args.checkpoint_state).parent.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_state).write_text(
        json.dumps({"needs_continuation": False, "row": args.row}, indent=2), encoding="utf-8"
    )

    wav_fid = upsert_file(drive, proj_id, str(final_wav), "voiceover_final.wav", "audio/wav")
    upsert_file(drive, proj_id, str(timings_path), "timings.json", "application/json")

    wav_url = file_view_url(wav_fid)
    update_cells(sheets, {
        f"C{args.row}": folder_url(proj_id),
        f"D{args.row}": wav_url,
        f"B{args.row}": "Html",
        f"G{args.row}": f"🎨 Đã ghép & căn nhịp {total_words} chữ ({total_s:.0f}s, nghỉ {args.gap_sec:.1f}s) — đang dựng HTML Word-Spotlight...",
    })
    notify_bridge_row_html_ready(args.row)
    print(f"[Job 11 Merge] Completed voiceover_final.wav + {total_words} word timings for row {args.row} -> {wav_url}", flush=True)


def run_align_existing(
    drive,
    sheets,
    args,
    row_info: dict,
    proj_id: str,
    proj_name: str,
    root_files: dict,
    chunks_spec: list,
):
    """Fast 0-TTS upgrade: align exact per-word timestamps on existing voiceover_final.wav."""
    work_dir = Path("work") / f"row_{args.row}"
    work_dir.mkdir(parents=True, exist_ok=True)

    voice_fid = extract_drive_id(row_info["voiceover_url"]) or (root_files.get("voiceover_final.wav") or {}).get("id")
    if not voice_fid:
        print(f"[Align] Row {args.row} has no voiceover_final.wav yet; falling back to merge.", flush=True)
        run_merge_final(drive, sheets, args, row_info, proj_id, proj_name, chunks_spec)
        return

    update_cells(sheets, {
        f"G{args.row}": "🎯 Đang căn chỉnh nhịp từng chữ (MMS_FA 20ms) trên file Voiceover sẵn có (0% đọc lại TTS)..."
    })

    final_wav = work_dir / "voiceover_final.wav"
    download_drive_file(drive, voice_fid, str(final_wav))

    if "timings.json" not in root_files:
        run_merge_final(drive, sheets, args, row_info, proj_id, proj_name, chunks_spec)
        return

    timings_local = work_dir / "timings_in.json"
    download_drive_file(drive, root_files["timings.json"]["id"], str(timings_local))
    timings_in = json.loads(timings_local.read_text(encoding="utf-8"))
    timing_items = timings_in.get("items") or []
    total_s = float(timings_in.get("total_s") or 0.0)

    total_words = populate_word_alignments_from_wav(final_wav, timing_items)

    timings_obj = {
        **timings_in,
        "schema_version": "2.5-word",
        "alignment_engine": "torchaudio_mms_fa_ctc_20ms",
        "word_count": total_words,
        "items": timing_items,
    }
    timings_out = work_dir / "timings.json"
    timings_out.write_text(json.dumps(timings_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    upsert_file(drive, proj_id, str(timings_out), "timings.json", "application/json")

    Path(args.checkpoint_state).parent.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_state).write_text(
        json.dumps({"needs_continuation": False, "row": args.row}, indent=2), encoding="utf-8"
    )

    wav_url = row_info["voiceover_url"] or file_view_url(voice_fid)
    update_cells(sheets, {
        f"C{args.row}": folder_url(proj_id),
        f"D{args.row}": wav_url,
        f"E{args.row}": "",
        f"F{args.row}": "",
        f"B{args.row}": "Html",
        f"G{args.row}": f"🎯 Đã căn nhịp {total_words} chữ ({total_s:.0f}s) — đang giao Mushu (#46) dựng HTML Word-Spotlight...",
    })
    notify_bridge_row_html_ready(args.row)
    print(f"[Align] Upgraded timings.json ({total_words} words) for row {args.row} ({proj_name}) -> Status=Html", flush=True)


def main():
    t_job_start = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--project", default="")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--mode", choices=["shard", "merge", "align", "full"], default="full")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=10)
    ap.add_argument("--gap-sec", type=float, default=CHUNK_GAP_S)
    ap.add_argument("--ref-audio", default="peter_ref.wav")
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-minutes", type=float, default=float(os.environ.get("OMNI_SOFT_LIMIT_MIN", "225")))
    ap.add_argument("--checkpoint-state", default="work/checkpoint_state.json")
    args = ap.parse_args()

    drive, sheets = get_services()
    row_info = read_task_row(sheets, args.row)
    proj_id, proj_name = resolve_project_folder(drive, sheets, args.row, args.project, create_if_missing=False)
    root_files = list_folder_files(drive, proj_id)

    if "manifest.json" not in root_files:
        raise RuntimeError(f"Row {args.row}: manifest.json missing in project folder {proj_name} ({proj_id})")

    work_dir = Path("work") / f"row_{args.row}"
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_local = work_dir / "manifest.json"
    download_drive_file(drive, root_files["manifest.json"]["id"], str(manifest_local))
    manifest = json.loads(manifest_local.read_text(encoding="utf-8"))
    chunks_spec = manifest.get("chunks") or []
    if not chunks_spec:
        raise RuntimeError(f"Row {args.row}: manifest has 0 chunks")

    if args.mode == "align":
        run_align_existing(drive, sheets, args, row_info, proj_id, proj_name, root_files, chunks_spec)
        return

    if row_info["voiceover_url"] and "voiceover_final.wav" in root_files and "timings.json" in root_files:
        # Check if timings.json already has v2.5-word alignments; if v2.4, upgrade in-place!
        t_check = work_dir / "timings_check.json"
        download_drive_file(drive, root_files["timings.json"]["id"], str(t_check))
        t_data = json.loads(t_check.read_text(encoding="utf-8"))
        if str(t_data.get("schema_version") or "").startswith("2.5"):
            if row_info["status"] in ("Gdrive", "Ready", "Voiceover"):
                update_cells(sheets, {f"B{args.row}": "Html"})
            print(f"Row {args.row}: voiceover_final.wav + v2.5-word timings.json already complete; skipping.")
            return
        if args.mode == "merge":
            run_align_existing(drive, sheets, args, row_info, proj_id, proj_name, root_files, chunks_spec)
            return

    chunks_drive_id, existing_chunk_files = list_all_chunks_across_folders(drive, proj_id)

    if args.mode == "shard":
        run_shard_synthesis(
            drive, sheets, args, proj_id, proj_name, chunks_spec, chunks_drive_id, existing_chunk_files, t_job_start
        )
    elif args.mode == "merge":
        run_merge_final(drive, sheets, args, row_info, proj_id, proj_name, chunks_spec)
    else:
        run_shard_synthesis(
            drive, sheets, args, proj_id, proj_name, chunks_spec, chunks_drive_id, existing_chunk_files, t_job_start
        )
        run_merge_final(drive, sheets, args, row_info, proj_id, proj_name, chunks_spec)


if __name__ == "__main__":
    main()
