#!/usr/bin/env python3
"""Usage: python tools/grade.py [-k test_name] [--reference]"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description="XGBoost教材の自動採点")
    p.add_argument("--reference", action="store_true", help="解答例を採点（CI 用）")
    p.add_argument("-k", "--keyword", help="pytest のテスト名フィルタ")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["XGB_TEXTBOOK_TARGET"] = "solutions.reference" if args.reference else "exercises.student"
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_exercises.py"]
    if args.keyword:
        cmd.extend(["-k", args.keyword])
    result = subprocess.run(cmd, env=env, cwd=root, check=False)
    if result.returncode == 0:
        print("PASS: すべての選択テストに合格しました。")
    else:
        print("NOT YET: 未実装または不正解があります。失敗した項目を確認してください。")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

