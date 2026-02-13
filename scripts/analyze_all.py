"""
AAPL, GOOGL, MSFT 세 종목 한 번에 분석
"""
import subprocess
import sys

SYMBOLS = ['AAPL', 'GOOGL', 'MSFT']

if __name__ == "__main__":
    for symbol in SYMBOLS:
        print(f"\n{'='*70}")
        print(f"{symbol} 분석 시작")
        print(f"{'='*70}")

        result = subprocess.run(
            [sys.executable, 'src/utils/analyze_stock.py', '--symbol', symbol],
            check=True
        )

    print("\n" + "=" * 70)
    print("전체 분석 완료: AAPL / GOOGL / MSFT")
    print("다음 파일들을 확인하세요:")
    for symbol in SYMBOLS:
        print(f"  data/models/{symbol.lower()}_analysis.png")
    print("=" * 70) 
