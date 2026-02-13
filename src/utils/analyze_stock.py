"""
범용 주식 분석 및 예측
AAPL, GOOGL, MSFT 어떤 종목이든 동일한 형식으로 분석

사용법:
    python src/utils/analyze_stock.py --symbol AAPL
    python src/utils/analyze_stock.py --symbol GOOGL
    python src/utils/analyze_stock.py --symbol MSFT
"""
import argparse
import os
import sys
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lstm_model import StockLSTM


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────

def load_model(symbol: str):
    """
    학습된 모델, 스케일러, Feature, 데이터 로드
    """
    symbol_lower = symbol.lower()
    device = 'cpu'

    model_path = f'data/models/lstm_{symbol_lower}.pt'
    scaler_path = f'data/models/scaler_{symbol_lower}.pkl'
    features_path = f'data/models/features_{symbol_lower}.pkl'
    data_path = f'data/processed/{symbol_lower}_processed.csv'

    for path in [model_path, scaler_path, features_path, data_path]:
        if not os.path.exists(path):
            print(f"에러: {path} 없음")
            print(f"먼저 train_stock.py --symbol {symbol} 실행 필요")
            sys.exit(1)

    with open(features_path, 'rb') as f:
        features = pickle.load(f)

    model = StockLSTM(
        input_size=len(features),
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

    print(f"[{symbol}] 모델 로드 완료")
    print(f"  데이터: {len(df)}행 | Features: {len(features)}개")
    print(f"  기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")

    return model, scaler, features, df, device


# ──────────────────────────────────────────────
# 예측
# ──────────────────────────────────────────────

def predict_one(model, scaler, features, window_60days, device):
    """
    60일 데이터 -> 60일 후 종가 예측
    """
    close_idx = features.index('Close')

    scaled = scaler.transform(window_60days)
    tensor = torch.FloatTensor(scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        output, _ = model(tensor)

    pred_norm = output.cpu().numpy()[0, 0]
    dummy = np.zeros((1, len(features)))
    dummy[0, close_idx] = pred_norm
    inversed = scaler.inverse_transform(dummy)

    return inversed[0, close_idx]


# ──────────────────────────────────────────────
# 백테스트
# ──────────────────────────────────────────────

def run_backtest(model, scaler, features, df, device):
    """
    과거 예측 vs 실제 비교
    모델 신뢰도 측정
    """
    print("\n  백테스트 실행 중...")

    results = []

    for i in range(60, len(df) - 120, 5):
        window = df.iloc[i:i + 60][features].values

        if len(window) < 60:
            continue
        if i + 120 >= len(df):
            continue

        actual = df.iloc[i + 120]['Close']
        predicted = predict_one(model, scaler, features, window, device)
        current = df.iloc[i + 60]['Close']

        results.append({
            'date': df.iloc[i + 60]['Date'],
            'actual': actual,
            'predicted': predicted,
            'error_pct': abs((predicted - actual) / actual * 100),
            'direction_actual': 1 if actual > current else -1,
            'direction_pred': 1 if predicted > current else -1
        })

    results_df = pd.DataFrame(results)

    avg_error = results_df['error_pct'].mean()
    direction_acc = (
        results_df['direction_actual'] == results_df['direction_pred']
    ).mean() * 100

    print(f"  테스트 횟수:   {len(results_df)}회")
    print(f"  평균 오차율:   {avg_error:.2f}%")
    print(f"  방향 정확도:   {direction_acc:.1f}%")

    return results_df, avg_error, direction_acc


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────

def visualize(symbol, model, scaler, features, df, device, results_df, avg_error, direction_acc):
    """
    전체 분석 결과 시각화
    총 9개 서브플롯 (6행 2열)
    """
    print("\n  그래프 생성 중...")

    fig = plt.figure(figsize=(20, 26))
    gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.45, wspace=0.3)

    recent = df.tail(180).copy()
    dates = recent['Date']

    # ── 1. 가격 + 이동평균 ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, recent['Close'], linewidth=2, label='Close', color='black')
    ax1.plot(dates, recent['SMA_20'], linewidth=1.5, label='SMA 20', color='royalblue', alpha=0.8)
    ax1.plot(dates, recent['SMA_50'], linewidth=1.5, label='SMA 50', color='darkorange', alpha=0.8)
    ax1.plot(dates, recent['SMA_200'], linewidth=1.5, label='SMA 200', color='crimson', alpha=0.8)
    ax1.set_title(f'{symbol} Price + Moving Averages (Recent 180 Days)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ── 2. RSI ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(dates, recent['RSI_14'], linewidth=2, color='purple')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax2.fill_between(dates, recent['RSI_14'], 70,
                     where=(recent['RSI_14'] >= 70), alpha=0.2, color='red')
    ax2.fill_between(dates, recent['RSI_14'], 30,
                     where=(recent['RSI_14'] <= 30), alpha=0.2, color='green')
    ax2.set_title('RSI (14)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── 3. MACD ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(dates, recent['MACD'], linewidth=2, label='MACD', color='royalblue')
    ax3.plot(dates, recent['MACD_Signal'], linewidth=2, label='Signal', color='crimson')
    ax3.bar(dates, recent['MACD_Hist'],
            color=['green' if v >= 0 else 'red' for v in recent['MACD_Hist']],
            alpha=0.5, label='Histogram')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title('MACD', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Value')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── 4. 볼린저 밴드 ──
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(dates, recent['Close'], linewidth=2, label='Close', color='black')
    ax4.plot(dates, recent['BB_Upper'], linewidth=1, label='Upper Band', color='red', alpha=0.7)
    ax4.plot(dates, recent['BB_Mid'], linewidth=1, label='Middle', color='royalblue', alpha=0.7)
    ax4.plot(dates, recent['BB_Lower'], linewidth=1, label='Lower Band', color='green', alpha=0.7)
    ax4.fill_between(dates, recent['BB_Upper'], recent['BB_Lower'], alpha=0.08, color='royalblue')
    ax4.set_title('Bollinger Bands', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Price (USD)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ── 5. 거래량 ──
    ax5 = fig.add_subplot(gs[2, 1])
    colors_vol = ['green' if recent['Close'].iloc[i] >= recent['Close'].iloc[i-1]
                  else 'red' for i in range(len(recent))]
    ax5.bar(dates, recent['Volume'], color=colors_vol, alpha=0.6)
    ax5.plot(dates, recent['Volume_MA20'], linewidth=2, color='royalblue', label='Volume MA20')
    ax5.set_title('Volume', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Volume')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    # ── 6. 백테스트: 예측 vs 실제 ──
    ax6 = fig.add_subplot(gs[3, :])
    ax6.plot(results_df['date'], results_df['actual'],
             linewidth=2, label='Actual Price', color='black')
    ax6.plot(results_df['date'], results_df['predicted'],
             linewidth=2, label='Predicted Price', color='royalblue',
             linestyle='--', alpha=0.8)
    ax6.fill_between(results_df['date'],
                     results_df['actual'],
                     results_df['predicted'],
                     alpha=0.15, color='red', label='Error Range')
    ax6.set_title('Backtest: Predicted vs Actual', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Price (USD)')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # ── 7. 오차 분포 ──
    ax7 = fig.add_subplot(gs[4, 0])
    ax7.hist(results_df['error_pct'], bins=20,
             color='royalblue', alpha=0.7, edgecolor='black')
    ax7.axvline(x=avg_error, color='red', linestyle='--',
                linewidth=2, label=f'Average: {avg_error:.2f}%')
    ax7.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Error (%)')
    ax7.set_ylabel('Count')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # ── 8. 방향 정확도 ──
    ax8 = fig.add_subplot(gs[4, 1])
    wrong_acc = 100 - direction_acc
    ax8.pie([direction_acc, wrong_acc],
            labels=[f'Correct\n{direction_acc:.1f}%', f'Wrong\n{wrong_acc:.1f}%'],
            colors=['green', 'red'],
            autopct='%1.1f%%',
            startangle=90)
    ax8.set_title('Direction Accuracy\n(Up / Down)', fontsize=12, fontweight='bold')

    # ── 9. 60일 후 예측 ──
    ax9 = fig.add_subplot(gs[5, :])

    recent_60_data = df.tail(60)[features].values
    predicted_price = predict_one(model, scaler, features, recent_60_data, device)

    current_price = df['Close'].iloc[-1]
    current_date = df['Date'].iloc[-1]
    prediction_date = current_date + timedelta(days=84)

    change = predicted_price - current_price
    change_pct = (change / current_price) * 100

    upper_bound = predicted_price * (1 + avg_error / 100)
    lower_bound = predicted_price * (1 - avg_error / 100)

    recent_120 = df.tail(120)
    ax9.plot(recent_120['Date'], recent_120['Close'],
             linewidth=2, label='Actual (Recent 120 Days)', color='black')
    ax9.axhline(y=current_price, color='green', linestyle='--',
                alpha=0.7, linewidth=2, label=f'Current: ${current_price:.2f}')
    ax9.plot([current_date, prediction_date],
             [current_price, predicted_price],
             'r--', linewidth=1.5, alpha=0.5)
    ax9.scatter([prediction_date], [predicted_price],
                s=300, c='red', marker='*', zorder=5,
                label=f'Predicted: ${predicted_price:.2f}')
    ax9.errorbar([prediction_date], [predicted_price],
                 yerr=[[predicted_price - lower_bound], [upper_bound - predicted_price]],
                 fmt='none', color='red', capsize=10, linewidth=2,
                 label=f'Range: ${lower_bound:.0f} ~ ${upper_bound:.0f}')

    ax9.set_title(
        f'{symbol} 60-Day Prediction  |  '
        f'Current: ${current_price:.2f}  ->  '
        f'Predicted: ${predicted_price:.2f} ({change_pct:+.2f}%)',
        fontsize=12, fontweight='bold'
    )
    ax9.set_ylabel('Price (USD)')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)

    # 전체 제목
    fig.suptitle(
        f'{symbol} LSTM Analysis Report  |  Generated: {datetime.now().strftime("%Y-%m-%d")}',
        fontsize=16, fontweight='bold', y=1.005
    )

    # 저장
    output_path = f'data/models/{symbol.lower()}_analysis.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()

    print(f"  저장: {output_path}")

    return predicted_price, current_price, change_pct, lower_bound, upper_bound


# ──────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────

def print_report(symbol, current_price, predicted_price, change_pct,
                 lower_bound, upper_bound, avg_error, direction_acc):
    """
    최종 분석 결과 출력
    """
    change = predicted_price - current_price

    if change_pct >= 10:
        signal = "강력 매수"
    elif change_pct >= 5:
        signal = "매수"
    elif change_pct >= -5:
        signal = "보유"
    elif change_pct >= -10:
        signal = "매도"
    else:
        signal = "강력 매도"

    print("\n" + "=" * 70)
    print(f"{symbol} 분석 결과")
    print("=" * 70)

    print(f"\n현재 종가:        ${current_price:,.2f}")
    print(f"60일 후 예측:     ${predicted_price:,.2f}")
    print(f"예상 변화:        ${change:+.2f} ({change_pct:+.2f}%)")
    print(f"예측 범위:        ${lower_bound:,.2f} ~ ${upper_bound:,.2f}")

    print(f"\n모델 신뢰도:")
    print(f"  평균 오차율:    {avg_error:.2f}%")
    print(f"  방향 정확도:    {direction_acc:.1f}%")

    print(f"\n투자 신호:        {signal}")

    print(f"\n투자 시뮬레이션:")
    for amount in [1000000, 5000000, 10000000]:
        shares = amount / current_price
        future_value = shares * predicted_price
        profit = future_value - amount
        print(f"  {amount:,}원 ->  {future_value:,.0f}원  ({profit:+,.0f}원)")

    print("\n" + "=" * 70)
    print("주의: 예측은 참고용입니다. 투자 결정은 신중하게!")
    print("=" * 70 + "\n")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main(symbol: str):
    print("\n" + "=" * 70)
    print(f"{symbol} LSTM 분석 시작")
    print("=" * 70)

    # 1. 모델 로드
    model, scaler, features, df, device = load_model(symbol)

    # 2. 백테스트
    results_df, avg_error, direction_acc = run_backtest(
        model, scaler, features, df, device
    )

    # 3. 시각화
    predicted_price, current_price, change_pct, lower_bound, upper_bound = visualize(
        symbol, model, scaler, features, df, device,
        results_df, avg_error, direction_acc
    )

    # 4. 결과 출력
    print_report(
        symbol, current_price, predicted_price, change_pct,
        lower_bound, upper_bound, avg_error, direction_acc
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='주식 LSTM 분석')
    parser.add_argument('--symbol', type=str, required=True,
                        help='종목 코드 (예: AAPL, GOOGL, MSFT)')
    args = parser.parse_args()

    main(args.symbol.upper())