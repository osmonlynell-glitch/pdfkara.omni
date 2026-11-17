#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from gcommon import (
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
SAME_PAGE_PAUSE_S = 0.18
PAGE_TURN_PAUSE_S = 0.42
TAIL_S = 0.45


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


def main():
    t_job_start = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--project", default="")
    ap.add_argument("--start", type=int, default=0)
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
        if row_info["status"] in ("Gdrive", "Ready"):
            update_cells(sheets, {f"B{args.row}": "Voiceover"})
        print(f"Row {args.row}: voiceover_final.wav + timings.json already complete; skipping.")
        return

    if "manifest.json" not in root_files:
        raise RuntimeError(f"Row {args.row}: manifest.json missing in project folder {proj_name} ({proj_id})")

    work_dir = Path("work") / f"row_{args.row}"
    chunks_local_dir = work_dir / "chunks"
    chunks_local_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_state).parent.mkdir(parents=True, exist_ok=True)

    manifest_local = work_dir / "manifest.json"
    download_drive_file(drive, root_files["manifest.json"]["id"], str(manifest_local))
    manifest = json.loads(manifest_local.read_text(encoding="utf-8"))
    chunks_spec = manifest.get("chunks") or []
    if not chunks_spec:
        raise RuntimeError(f"Row {args.row}: manifest has 0 chunks")

    chunks_drive_id = ensure_folder(drive, proj_id, "chunks")
    existing_chunk_files = list_folder_files(drive, chunks_drive_id)

    missing_indices = []
    for item in chunks_spec:
        idx = int(item["index"])
        cname = f"chunk_{idx:04d}.wav"
        fmeta = existing_chunk_files.get(cname)
        if not fmeta or int(fmeta.get("size") or 0) < 512:
            missing_indices.append(idx)

    print(
        f"Row {args.row} ({proj_name}): total={len(chunks_spec)} chunks, "
        f"already_on_drive={len(chunks_spec) - len(missing_indices)}, remaining={len(missing_indices)}"
    )

    if missing_indices:
        from omnivoice import OmniVoice

        torch.set_num_threads(max(1, os.cpu_count() or 4))
        model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32, asr_device="cpu"
        )
        voice_prompt = model.create_voice_clone_prompt(ref_audio=args.ref_audio, ref_text=None)
        soft_deadline_s = args.max_minutes * 60.0

        spec_by_idx = {int(c["index"]): c for c in chunks_spec}
        for pos, idx in enumerate(missing_indices):
            c_item = spec_by_idx[idx]
            text = str(c_item["text"]).strip()
            lang = c_item.get("lang") or detect_lang(text)
            t0 = time.monotonic()

            audio = model.generate(
                text=text,
                language=lang,
                voice_clone_prompt=voice_prompt,
                speed=args.speed,
                num_step=args.num_step,
            )[0].astype(np.float32)

            peak = float(np.max(np.abs(audio))) or 1.0
            if peak > 0.96:
                audio = audio / peak * 0.96

            cname = f"chunk_{idx:04d}.wav"
            local_chunk = chunks_local_dir / cname
            sf.write(str(local_chunk), audio, SR, subtype="PCM_16")

            # MANDATORY SPEC RULE: Upload immediately to Drive before starting next chunk!
            fid = upsert_file(drive, chunks_drive_id, str(local_chunk), cname, "audio/wav")
            existing_chunk_files[cname] = {"id": fid, "name": cname, "size": local_chunk.stat().st_size}

            done_cnt = len(existing_chunk_files)
            tot_cnt = len(chunks_spec)
            pct = int(round(done_cnt * 100.0 / max(1, tot_cnt)))
            try:
                update_cells(sheets, {
                    f"B{args.row}": "Voiceover",
                    f"G{args.row}": f"🎙 Đang đọc giọng Peter: {done_cnt}/{tot_cnt} câu ({pct}%)",
                })
            except Exception:
                pass

            gen_s = time.monotonic() - t0
            dur_s = len(audio) / SR
            elapsed_min = (time.monotonic() - t_job_start) / 60.0
            print(
                f"[chunk {idx + 1}/{len(chunks_spec)}] {len(text)} chars -> {dur_s:.2f}s audio "
                f"in {gen_s:.1f}s (uploaded {cname}, elapsed={elapsed_min:.1f}m)",
                flush=True,
            )

            remaining_after = missing_indices[pos + 1 :]
            if remaining_after and (time.monotonic() - t_job_start) >= soft_deadline_s:
                ckpt = {
                    "needs_continuation": True,
                    "row": args.row,
                    "pdf_link": row_info["pdf_link"] or "resume",
                    "project_folder_url": folder_url(proj_id),
                    "next_chunk": remaining_after[0],
                    "completed_chunks": len(chunks_spec) - len(remaining_after),
                    "total_chunks": len(chunks_spec),
                }
                Path(args.checkpoint_state).write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
                upsert_file(drive, proj_id, args.checkpoint_state, "checkpoint.json", "application/json")
                print(f"Approaching job timeout ({elapsed_min:.1f}m). Saved checkpoint -> next_chunk={remaining_after[0]}")
                return

    # All chunks exist on Drive -> download any previously generated chunks not yet on local disk
    pieces = [np.zeros(int(round(LEAD_IN_S * SR)), dtype=np.float32)]
    cursor_samples = len(pieces[0])
    timing_items = []

    for pos, item in enumerate(chunks_spec):
        idx = int(item["index"])
        cname = f"chunk_{idx:04d}.wav"
        local_chunk = chunks_local_dir / cname
        if not local_chunk.is_file():
            download_drive_file(drive, existing_chunk_files[cname]["id"], str(local_chunk))

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
            next_page = int(chunks_spec[pos + 1].get("page") or item.get("page") or 1)
            cur_page = int(item.get("page") or 1)
            pause_s = PAGE_TURN_PAUSE_S if next_page != cur_page else SAME_PAGE_PAUSE_S
            pause_samples = int(round(pause_s * SR))
            pieces.append(np.zeros(pause_samples, dtype=np.float32))
            cursor_samples += pause_samples
        else:
            tail_samples = int(round(TAIL_S * SR))
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

    # Pad exact frame boundary @ 30fps
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
        "total_s": round(total_s, 4),
        "chunk_count": len(timing_items),
        "voice": "Peter (Peter_voice_mau2_tre_20261116.wav)",
        "items": timing_items,
    }
    timings_path = work_dir / "timings.json"
    timings_path.write_text(json.dumps(timings_obj, ensure_ascii=False, indent=2), encoding="utf-8")

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
        f"G{args.row}": f"🎨 Đã đọc xong {len(timing_items)} câu ({total_s:.0f}s) — đang dựng HTML 16:9...",
    })
    print(f"Completed voiceover for row {args.row}: duration={total_s:.2f}s -> {wav_url}")


if __name__ == "__main__":
    main()
