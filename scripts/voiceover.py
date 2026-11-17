#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from gcommon import (
    download_drive_file,
    ensure_folder,
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
CHUNK_GAP_S = 0.50  # Khoảng nghỉ chuẩn 0.5 giây giữa các chunks theo yêu cầu
TAIL_S = 0.50


def detect_lang(text: str) -> str:
    vi_chars = set("àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ")
    low = text.lower()
    if any(ch in vi_chars for ch in low):
        return "vi"
    return "en"


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

        if not re.search(r"[A-Za-z0-9\u00C0-\u1EF9]", text):
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

        # Bắt buộc upload ngay lên Drive trước khi sang câu kế
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

        # Chỉ luồng 0 cập nhật trực tiếp tổng số chunk từ Drive lên cột G để tránh 10 luồng ghi đè nhau
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

    print(f"[Job 11 Merge] All {len(chunks_spec)} chunks verified on Drive! Merging with {args.gap_sec:.2f}s spacing...", flush=True)
    update_cells(sheets, {
        f"G{args.row}": f"🎚 Tiến trình #11 đang ghép {len(chunks_spec)} chunks (cách nhau {args.gap_sec:.1f}s) & chuẩn hóa -14 LUFS..."
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

    timings_obj = {
        "schema_version": "2.4",
        "fps": FPS,
        "sample_rate": SR,
        "chunk_gap_s": args.gap_sec,
        "total_s": round(total_s, 4),
        "chunk_count": len(timing_items),
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
        f"G{args.row}": f"🎨 Đã ghép xong {len(timing_items)} câu ({total_s:.0f}s, nghỉ {args.gap_sec:.1f}s) — đang dựng HTML 16:9...",
    })
    print(f"[Job 11 Merge] Completed voiceover_final.wav for row {args.row}: duration={total_s:.2f}s -> {wav_url}", flush=True)


def main():
    t_job_start = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--project", default="")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--mode", choices=["shard", "merge", "full"], default="full")
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

    if row_info["voiceover_url"] and "voiceover_final.wav" in root_files and "timings.json" in root_files:
        if row_info["status"] in ("Gdrive", "Ready", "Voiceover"):
            update_cells(sheets, {f"B{args.row}": "Html"})
        print(f"Row {args.row}: voiceover_final.wav + timings.json already complete; skipping.")
        return

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
