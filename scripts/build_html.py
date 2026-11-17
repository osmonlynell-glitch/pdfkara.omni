#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from gcommon import (
    dispatch_github_workflow,
    download_drive_file,
    file_view_url,
    folder_url,
    get_services,
    list_folder_files,
    read_task_row,
    resolve_project_folder,
    update_cells,
    upsert_file,
)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=1920, height=1080" />
  <title>__TITLE__</title>
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      width: 1920px;
      height: 1080px;
      overflow: hidden;
      background: #0b111b;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    #root {
      position: relative;
      width: 1920px;
      height: 1080px;
      overflow: hidden;
      background: radial-gradient(circle at 50% 18%, #172336 0%, #090e17 75%);
    }
    .stage-frame {
      position: absolute;
      left: 240px;
      top: 32px;
      width: 1440px;
      height: 1016px;
      border-radius: 16px;
      overflow: hidden;
      background: #ffffff;
      box-shadow: 0 28px 80px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(255, 255, 255, 0.12);
    }
    #page-mover {
      position: absolute;
      left: 0;
      top: 0;
      width: 1440px;
      will-change: transform;
    }
    #page-img {
      display: block;
      width: 1440px;
      height: auto;
      user-select: none;
    }
    #hl-layer {
      position: absolute;
      inset: 0;
      pointer-events: none;
    }
    .pdf-hl {
      position: absolute;
      background: rgba(255, 226, 36, 0.44);
      mix-blend-mode: multiply;
      border-radius: 5px;
      box-shadow: 0 0 0 2px rgba(245, 178, 0, 0.34);
    }
    .hud-pill {
      position: absolute;
      right: 40px;
      top: 36px;
      padding: 10px 18px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.82);
      border: 1px solid rgba(148, 163, 184, 0.28);
      color: #e2e8f0;
      font-size: 20px;
      font-weight: 600;
      letter-spacing: 0.03em;
    }
    .progress-track {
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: 6px;
      background: rgba(255, 255, 255, 0.12);
    }
    #progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #facc15 0%, #f59e0b 100%);
    }
  </style>
</head>
<body>
  <div id="root" data-composition-id="main" data-start="0" data-duration="__DURATION__" data-width="1920" data-height="1080">
    <div class="stage-frame" id="viewport">
      <div id="page-mover">
        <img id="page-img" src="pages/page_001.png" alt="PDF Page" />
        <div id="hl-layer"></div>
      </div>
    </div>
    <div class="hud-pill" id="page-pill">PAGE 1 / 1</div>
    <div class="progress-track"><div id="progress-bar"></div></div>
  </div>
  <script>
    const DATA = __KARAOKE_JSON__;
    const VIEW_W = 1440;
    const VIEW_H = 1016;
    const FOCUS_Y = VIEW_H * 0.42;

    const pagesByNum = {};
    (DATA.pages || []).forEach(p => { pagesByNum[p.page] = p; });
    const items = DATA.items || [];
    const totalS = Number(DATA.total_s || 1);
    const totalPages = (DATA.pages || []).length || 1;

    const pageImg = document.getElementById("page-img");
    const pageMover = document.getElementById("page-mover");
    const hlLayer = document.getElementById("hl-layer");
    const pagePill = document.getElementById("page-pill");
    const progBar = document.getElementById("progress-bar");

    let currentPage = -1;
    let currentIdx = -2;

    function pageHeightPx(pnum) {
      const pm = pagesByNum[pnum];
      if (!pm || !pm.width || !pm.height) return 2036;
      return VIEW_W * (pm.height / pm.width);
    }

    function itemCenterY(item, pageH) {
      const rects = item.rects || [];
      if (!rects.length) return pageH * 0.35;
      let yMin = 1, yMax = 0;
      for (let i = 0; i < rects.length; i++) {
        if (rects[i][1] < yMin) yMin = rects[i][1];
        if (rects[i][3] > yMax) yMax = rects[i][3];
      }
      return ((yMin + yMax) * 0.5) * pageH;
    }

    function clampScroll(targetY, pageH) {
      const maxScroll = Math.max(0, pageH - VIEW_H);
      const raw = targetY - FOCUS_Y;
      return Math.max(0, Math.min(maxScroll, raw));
    }

    function smoothstep(x) {
      const c = Math.max(0, Math.min(1, x));
      return c * c * (3 - 2 * c);
    }

    function findActiveIndex(t) {
      if (!items.length) return -1;
      for (let i = 0; i < items.length; i++) {
        if (t >= items[i].start_s && t < items[i].end_s) return i;
      }
      return t < items[0].start_s ? 0 : items.length - 1;
    }

    function renderAtTime(t) {
      progBar.style.width = Math.min(100, Math.max(0, (t / totalS) * 100)).toFixed(3) + "%";
      const idx = findActiveIndex(t);
      if (idx < 0) return;
      const cur = items[idx];
      const pnum = cur.page || 1;
      const pMeta = pagesByNum[pnum] || { image: "pages/page_" + String(pnum).padStart(3, "0") + ".png" };
      const pageH = pageHeightPx(pnum);

      if (pnum !== currentPage) {
        currentPage = pnum;
        pageImg.src = pMeta.image;
        pageMover.style.height = pageH.toFixed(1) + "px";
        pagePill.textContent = "PAGE " + pnum + " / " + totalPages;
        currentIdx = -2;
      }

      if (idx !== currentIdx) {
        currentIdx = idx;
        hlLayer.innerHTML = "";
        const padX = 0.004, padY = 0.0025;
        (cur.rects || []).forEach(r => {
          const d = document.createElement("div");
          d.className = "pdf-hl";
          const x0 = Math.max(0, r[0] - padX);
          const y0 = Math.max(0, r[1] - padY);
          const w = Math.min(1 - x0, (r[2] - r[0]) + padX * 2);
          const h = Math.min(1 - y0, (r[3] - r[1]) + padY * 2);
          d.style.left = (x0 * 100).toFixed(4) + "%";
          d.style.top = (y0 * 100).toFixed(4) + "%";
          d.style.width = (w * 100).toFixed(4) + "%";
          d.style.height = (h * 100).toFixed(4) + "%";
          hlLayer.appendChild(d);
        });
      }

      let scrollY = clampScroll(itemCenterY(cur, pageH), pageH);
      if (idx > 0 && items[idx - 1].page === pnum) {
        const prevScroll = clampScroll(itemCenterY(items[idx - 1], pageH), pageH);
        const transDur = Math.min(0.38, Math.max(0.12, (cur.end_s - cur.start_s) * 0.25));
        const alpha = smoothstep((t - cur.start_s) / transDur);
        scrollY = prevScroll + (scrollY - prevScroll) * alpha;
      }
      pageMover.style.transform = "translate3d(0," + (-scrollY).toFixed(2) + "px,0)";
    }

    const tl = gsap.timeline({
      paused: true,
      onUpdate: function () { renderAtTime(this.time()); }
    });
    tl.to({}, { duration: totalS }, 0);
    window.__timelines = window.__timelines || {};
    window.__timelines["main"] = tl;
    tl.seek(0);
    renderAtTime(0);
  </script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--project", default="")
    ap.add_argument("--dispatch-render", action="store_true")
    args = ap.parse_args()

    drive, sheets = get_services()
    row_info = read_task_row(sheets, args.row)
    proj_id, proj_name = resolve_project_folder(drive, sheets, args.row, args.project, create_if_missing=False)
    files = list_folder_files(drive, proj_id)

    if "manifest.json" not in files or "timings.json" not in files:
        raise RuntimeError(f"Row {args.row}: manifest.json or timings.json missing in {proj_name}")

    work_dir = Path("work") / f"row_{args.row}"
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = work_dir / "manifest.json"
    timings_p = work_dir / "timings.json"
    download_drive_file(drive, files["manifest.json"]["id"], str(manifest_p))
    download_drive_file(drive, files["timings.json"]["id"], str(timings_p))

    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    timings = json.loads(timings_p.read_text(encoding="utf-8"))

    payload = {
        "project_name": proj_name,
        "total_s": timings["total_s"],
        "fps": timings.get("fps", 30),
        "pages": manifest.get("pages") or [],
        "items": timings.get("items") or [],
    }
    html_text = (
        HTML_TEMPLATE.replace("__TITLE__", proj_name)
        .replace("__DURATION__", str(timings["total_s"]))
        .replace("__KARAOKE_JSON__", json.dumps(payload, ensure_ascii=False))
    )

    out_html = work_dir / "karaoke_16x9.html"
    out_html.write_text(html_text, encoding="utf-8")
    html_fid = upsert_file(drive, proj_id, str(out_html), "karaoke_16x9.html", "text/html")
    html_url = file_view_url(html_fid)

    update_cells(sheets, {
        f"C{args.row}": folder_url(proj_id),
        f"E{args.row}": html_url,
        f"B{args.row}": "Html",
    })
    print(f"Built karaoke_16x9.html for row {args.row} -> {html_url}")

    if args.dispatch_render:
        owner = os.environ.get("GITHUB_REPOSITORY_OWNER") or os.environ.get("GITHUB_REPOSITORY", "/").split("/")[0]
        if owner:
            dispatch_github_workflow(
                f"{owner}/pdfkara.ren",
                "render.yml",
                {
                    "row": str(args.row),
                    "project_name": folder_url(proj_id),
                    "html_url": html_url,
                },
            )


if __name__ == "__main__":
    main()
