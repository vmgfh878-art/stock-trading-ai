"""AAPL, GOOGL, MSFT Transformer 한 번에 학습"""
import subprocess
import sys

for symbol in ['AAPL', 'GOOGL', 'MSFT']:
    print(f"\n{'='*70}\n{symbol} Transformer 학습\n{'='*70}")
    subprocess.run(
        [sys.executable, 'src/training/train_transformer.py', '--symbol', symbol],
        check=True
    )

print("\n전체 Transformer 학습 완료!")