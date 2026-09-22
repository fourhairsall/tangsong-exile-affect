#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_corpora.py -- obtain the raw corpora used by this project.

None of these texts are redistributed in this repository: they belong to their
respective providers and remain subject to their own licences.  This script
only automates fetching them and arranging them into the directory layout the
analysis pipeline expects:

    data/
      tang/       poet.tang.*.json    全唐詩            (chinese-poetry)
      songshi/    poet.song.*.json    全宋詩            (Book1Q84 / chinese-poetry)
      yuding/     *.json              御定全唐詩        (see docs/CORPUS_FETCH.md)
      song2/      KR4h0143_*.txt      御選宋詩          (kanripo, 四庫全書文淵閣本)
      songchao/   KR4h0157_*.txt      宋詩鈔            (kanripo, 四庫全書文淵閣本)
      qtw/        KR4h0168_*.txt      全唐文            (kanripo)
      songbaijia/ KR4h0167_*.txt      宋百家詩存        (kanripo)
      songyipu/   KR4h0113_*.txt      宋詩紀事          (kanripo)
      ci/         ci.song.*.json      全宋詞            (chinese-poetry; the pipeline
                                                        used a sqlite ci.db build)

Usage
-----
    python tools/fetch_corpora.py --data-dir data

Requires network access and `git` on PATH.  Every step is idempotent: sources
already present in the working directory are skipped.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(tempfile.gettempdir(), "tangsong_corpus_cache")

# --- upstream providers -------------------------------------------------------
POETRY_REPOS = [
    # (clone URL, description). The first is the canonical dataset; the second
    # (Book1Q84) is a mirror that also carries 全宋詩 as poet.song.*.json.
    ("https://github.com/chinese-poetry/chinese-poetry.git",
     "chinese-poetry: 全唐詩 json/, 全宋詞 ci/"),
    ("https://github.com/Book1Q84/Chinese-Poetry-Dataset.git",
     "Book1Q84 mirror: 全唐詩 + 全宋詩 (poet.song.*.json)"),
]

KANRIPO = [
    ("KR4h0143", "song2", "御選宋詩 (清·康熙敕選, 四庫全書文淵閣本)"),
    ("KR4h0157", "songchao", "宋詩鈔 (清·吳之振, 四庫全書文淵閣本)"),
    ("KR4h0168", "qtw", "全唐文 (四庫全書文淵閣本)"),
    ("KR4h0167", "songbaijia", "宋百家詩存 (四庫全書文淵閣本)"),
    ("KR4h0113", "songyipu", "宋詩紀事 (四庫全書文淵閣本)"),
]


def run(cmd, cwd=None):
    print("  $ " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def clone(url, dest):
    if os.path.isdir(dest):
        print("  [skip] already cloned:", dest)
        return
    run(["git", "clone", "--depth", "1", url, dest])


def copy_matching(src_dir, pattern, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    hits = sorted(glob.glob(os.path.join(src_dir, pattern)))
    for h in hits:
        shutil.copy2(h, os.path.join(dst_dir, os.path.basename(h)))
    return len(hits)


def fetch_poetry(cache, data_dir):
    for url, desc in POETRY_REPOS:
        name = os.path.basename(url)[:-4]
        dest = os.path.join(cache, name)
        print("== %s" % desc)
        try:
            clone(url, dest)
        except subprocess.CalledProcessError as e:
            print("  [warn] clone failed:", e)
            continue

        # 全唐詩 / 全宋詩 live in json/ ; 全宋詞 in ci/
        for src_rel, pattern, out in [
            ("json", "poet.tang.*.json", "tang"),
            ("json", "poet.song.*.json", "songshi"),
            ("ci",   "ci.song.*.json",   "ci"),
        ]:
            src = os.path.join(dest, src_rel)
            if os.path.isdir(src):
                n = copy_matching(src, pattern, os.path.join(data_dir, out))
                if n:
                    print("  -> data/%s/ : %d files" % (out, n))


def fetch_kanripo(cache, data_dir):
    for kr, out, desc in KANRIPO:
        dest = os.path.join(cache, kr)
        print("== %s  %s" % (kr, desc))
        try:
            clone("https://github.com/kanripo/%s.git" % kr, dest)
        except subprocess.CalledProcessError as e:
            print("  [warn] clone failed:", e)
            continue
        n = copy_matching(dest, "%s_*.txt" % kr, os.path.join(data_dir, out))
        print("  -> data/%s/ : %d files" % (out, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data",
                    help="target data directory (default: ./data)")
    ap.add_argument("--cache", default=DEFAULT_CACHE,
                    help="where to keep the upstream clones (default: temp dir)")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    fetch_poetry(args.cache, data_dir)
    fetch_kanripo(args.cache, data_dir)

    print()
    print("Done. Layout under %s:" % data_dir)
    for d in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, d)
        if os.path.isdir(p):
            n = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))])
            print("  %-14s %d files" % (d, n))
    print()
    print("NOTE: 御定全唐詩 (data/yuding/) is not mirrored on GitHub; see")
    print("      docs/CORPUS_FETCH.md for how it was obtained. If a source is")
    print("      missing, the pipeline stage that consumes it will be skipped.")


if __name__ == "__main__":
    sys.exit(main())
