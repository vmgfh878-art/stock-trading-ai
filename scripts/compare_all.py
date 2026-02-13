"""AAPL, GOOGL, MSFT LSTM vs Transformer 비교 분석"""
import subprocess
import sys

for symbol in ['AAPL', 'GOOGL', 'MSFT']:
    print(f"\n{'='*70}\n{symbol} 비교 분석\n{'='*70}")
    subprocess.run(
        [sys.executable, 'src/utils/analyze_transformer.py', '--symbol', symbol],
        check=True
    )

print("\n전체 비교 완료!")
print("결과 파일:")
for symbol in ['AAPL', 'GOOGL', 'MSFT']:
    print(f"  data/models/{symbol.lower()}_comparison.png")