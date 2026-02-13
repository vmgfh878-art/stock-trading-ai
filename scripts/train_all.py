"""
AAPL, GOOGL, MSFT 세 종목 한 번에 학습
"""
import subprocess
import sys

SYMBOLS = ['AAPL', 'GOOGL', 'MSFT']

if __name__ == "__main__":
    for symbol in SYMBOLS:
        print(f"\n{'='*70}")
        print(f"{symbol} 학습 시작")
        print(f"{'='*70}")

        result = subprocess.run(
            [sys.executable, 'src/training/train_stock.py', '--symbol', symbol],
            check=True
        )

    print("\n" + "=" * 70)
    print("전체 학습 완료: AAPL / GOOGL / MSFT")
    print("=" * 70) 
 
