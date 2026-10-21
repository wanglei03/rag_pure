"""Stage 07:评估 —— 用数据证明"这套 RAG 管用"。

运行:python stages/stage07_evaluation/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.runner import run_full_eval

if __name__ == "__main__":
    run_full_eval()
