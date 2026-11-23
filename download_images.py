"""
Download Fakeddit images for one split (train / validate / test).

Reads:   multimodal_only_samples/multimodal_{split}.tsv
Writes:  images/{split}/{id}.{ext}
         manifests/{split}.csv  (only successfully-downloaded rows)

Manifest columns: id, image_path, clean_title, label_2way, label_3way, label_6way

Resumable: re-running skips files already on disk.
"""

import argparse
import csv
import io
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

ROOT = Path(__file__).parent
TSV_DIR = ROOT / "multimodal_only_samples"
IMG_DIR = ROOT / "images"
MANIFEST_DIR = ROOT / "manifests"

SPLIT_TO_TSV = {
    "train":    "multimodal_train.tsv",
    "validate": "multimodal_validate.tsv",
    "test":     "multimodal_test_public.tsv",
}

PIL_FMT_TO_EXT = {"JPEG": "jpg", "PNG": "png", "GIF": "gif", "WEBP": "webp", "BMP": "bmp"}

UA = "Mozilla/5.0 (compatible; FakedditFetcher/1.0)"
TIMEOUT = 20
MAX_RETRIES = 3
MAX_BYTES = 15 * 1024 * 1024  # 15 MB cap per image


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


def fetch_one(session: requests.Session, row_id: str, url: str, out_dir: Path) -> tuple[str, str | None, str]:
    """Returns (id, relative_path_or_None, status)."""
    # Resume: any existing file matching this id?
    for ext in PIL_FMT_TO_EXT.values():
        p = out_dir / f"{row_id}.{ext}"
        if p.exists() and p.stat().st_size > 0:
            return row_id, str(p.relative_to(ROOT)), "cached"

    backoff = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=TIMEOUT, stream=True)
            if r.status_code in (403, 404, 410):
                return row_id, None, f"http_{r.status_code}"
            r.raise_for_status()

            buf = io.BytesIO()
            total = 0
            for chunk in r.iter_content(chunk_size=64 * 1024):
                buf.write(chunk)
                total += len(chunk)
                if total > MAX_BYTES:
                    return row_id, None, "too_large"
            buf.seek(0)

            try:
                img = Image.open(buf)
                img.verify()
            except (UnidentifiedImageError, Exception):
                return row_id, None, "not_image"

            buf.seek(0)
            fmt = Image.open(buf).format
            ext = PIL_FMT_TO_EXT.get(fmt, "bin")
            if ext == "bin":
                return row_id, None, "unknown_fmt"

            out_path = out_dir / f"{row_id}.{ext}"
            with open(out_path, "wb") as f:
                f.write(buf.getvalue())
            return row_id, str(out_path.relative_to(ROOT)), "ok"

        except (requests.Timeout, requests.ConnectionError):
            if attempt == MAX_RETRIES - 1:
                return row_id, None, "timeout"
            time.sleep(backoff + random.random())
            backoff *= 2
        except requests.HTTPError as e:
            return row_id, None, f"http_err_{e.response.status_code if e.response else 'x'}"
        except Exception as e:
            return row_id, None, f"err_{type(e).__name__}"

    return row_id, None, "retries_exhausted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLIT_TO_TSV.keys()))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap rows (random sample) for quick experiments.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tsv_path = TSV_DIR / SPLIT_TO_TSV[args.split]
    out_dir = IMG_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    df = df[df["hasImage"] == True]
    df = df[df["image_url"].notna() & (df["image_url"] != "")]
    df = df.drop_duplicates(subset=["id"])
    print(f"[load] {len(df):,} rows with images")

    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=args.seed)
        print(f"[load] sampled to {len(df):,}")

    rows = df[["id", "image_url", "clean_title",
               "2_way_label", "3_way_label", "6_way_label"]].to_dict("records")

    results: dict[str, tuple[str | None, str]] = {}
    status_counts: dict[str, int] = {}

    session = make_session()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch_one, session, r["id"], r["image_url"], out_dir) for r in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=args.split):
            rid, path, status = fut.result()
            results[rid] = (path, status)
            status_counts[status] = status_counts.get(status, 0) + 1

    print("[stats]")
    for k, v in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v:>8,}")

    manifest_path = MANIFEST_DIR / f"{args.split}.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image_path", "clean_title",
                    "label_2way", "label_3way", "label_6way"])
        for r in rows:
            path, status = results[r["id"]]
            if path is None:
                continue
            w.writerow([r["id"], path, r["clean_title"],
                        r["2_way_label"], r["3_way_label"], r["6_way_label"]])

    kept = sum(1 for v in results.values() if v[0] is not None)
    print(f"[manifest] {manifest_path}  ({kept:,} rows)")


if __name__ == "__main__":
    sys.exit(main())
