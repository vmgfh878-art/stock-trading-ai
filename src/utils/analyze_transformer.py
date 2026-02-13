"""
Transformer 분석 및 LSTM과 비교
LSTM과 완전히 동일한 형식의 그래프 출력
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
from models.transformer_model import StockTransformer
from models.lstm_model import StockLSTM


def load_models(symbol):
    """
    Transformer + LSTM 둘 다 로드
    """
    symbol_lower = symbol.lower()
    device = 'cpu'

    # Transformer 로드
    with open(f'data/models/features_transformer_{symbol_lower}.pkl', 'rb') as f:
        features_tf = pickle.load(f)
    with open(f'data/models/scaler_transformer_{symbol_lower}.pkl', 'rb') as f:
        scaler_tf = pickle.load(f)

    transformer = StockTransformer(
        input_size=len(features_tf),
        d_model=128, nhead=8,
        num_layers=3, dropout=0.1
    ).to(device)
    transformer.load_state_dict(
        torch.load(f'data/models/transformer_{symbol_lower}.pt', map_location=device)
    )
    transformer.eval()

    # LSTM 로드
    with open(f'data/models/features_{symbol_lower}.pkl', 'rb') as f:
        features_lstm = pickle.load(f)
    with open(f'data/models/scaler_{symbol_lower}.pkl', 'rb') as f:
        scaler_lstm = pickle.load(f)

    lstm = StockLSTM(
        input_size=len(features_lstm),
        hidden_size=128, num_layers=2, dropout=0.2
    ).to(device)
    lstm.load_state_dict(
        torch.load(f'data/models/lstm_{symbol_lower}.pt', map_location=device)
    )
    lstm.eval()

    df = pd.read_csv(f'data/processed/{symbol_lower}_processed.csv')
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

    print(f"[{symbol}] 모델 로드 완료")
    print(f"  Transformer: 시퀀스 504일 | Features {len(features_tf)}개")
    print(f"  LSTM:        시퀀스  60일 | Features {len(features_lstm)}개")

    return transformer, scaler_tf, features_tf, lstm, scaler_lstm, features_lstm, df, device


def predict_transformer(transformer, scaler, features, window, device):
    """Transformer 예측 (504일 입력)"""
    close_idx = features.index('Close')
    scaled = scaler.transform(window)
    tensor = torch.FloatTensor(scaled).unsqueeze(0).to(device)
    with torch.no_grad():
        output = transformer(tensor)
    pred_norm = output.cpu().numpy()[0, 0]
    dummy = np.zeros((1, len(features)))
    dummy[0, close_idx] = pred_norm
    return scaler.inverse_transform(dummy)[0, close_idx]


def predict_lstm(lstm, scaler, features, window, device):
    """LSTM 예측 (60일 입력)"""
    close_idx = features.index('Close')
    scaled = scaler.transform(window)
    tensor = torch.FloatTensor(scaled).unsqueeze(0).to(device)
    with torch.no_grad():
        output, _ = lstm(tensor)
    pred_norm = output.cpu().numpy()[0, 0]
    dummy = np.zeros((1, len(features)))
    dummy[0, close_idx] = pred_norm
    return scaler.inverse_transform(dummy)[0, close_idx]


def run_backtest(transformer, scaler_tf, features_tf,
                 lstm, scaler_lstm, features_lstm, df, device):
    """
    두 모델 동시 백테스트
    같은 기간에 대해 둘 다 예측하고 비교
    """
    print("\n  백테스트 실행 중...")

    results = []

    # 504일 이상 되는 시점부터 시작
    start_idx = max(504, 60)

    for i in range(start_idx, len(df) - 120, 5):
        # Transformer: 504일 입력
        if i < 504:
            continue
        window_tf = df.iloc[i-504:i][features_tf].values
        if len(window_tf) < 504:
            continue

        # LSTM: 60일 입력
        window_lstm = df.iloc[i-60:i][features_lstm].values
        if len(window_lstm) < 60:
            continue

        # 60일 후 실제값
        if i + 120 >= len(df):
            continue

        actual = df.iloc[i + 120]['Close']
        current = df.iloc[i]['Close']

        pred_tf = predict_transformer(transformer, scaler_tf, features_tf, window_tf, device)
        pred_lstm = predict_lstm(lstm, scaler_lstm, features_lstm, window_lstm, device)

        results.append({
            'date': df.iloc[i]['Date'],
            'actual': actual,
            'pred_transformer': pred_tf,
            'pred_lstm': pred_lstm,
            'error_tf': abs((pred_tf - actual) / actual * 100),
            'error_lstm': abs((pred_lstm - actual) / actual * 100),
            'dir_actual': 1 if actual > current else -1,
            'dir_tf': 1 if pred_tf > current else -1,
            'dir_lstm': 1 if pred_lstm > current else -1,
        })

    results_df = pd.DataFrame(results)

    avg_err_tf = results_df['error_tf'].mean()
    avg_err_lstm = results_df['error_lstm'].mean()
    dir_acc_tf = (results_df['dir_actual'] == results_df['dir_tf']).mean() * 100
    dir_acc_lstm = (results_df['dir_actual'] == results_df['dir_lstm']).mean() * 100

    print(f"\n  백테스트 결과 ({len(results_df)}회):")
    print(f"  {'':20} {'Transformer':>15} {'LSTM':>15}")
    print(f"  {'평균 오차율':20} {avg_err_tf:>14.2f}% {avg_err_lstm:>14.2f}%")
    print(f"  {'방향 정확도':20} {dir_acc_tf:>14.1f}% {dir_acc_lstm:>14.1f}%")

    return results_df, avg_err_tf, avg_err_lstm, dir_acc_tf, dir_acc_lstm


def visualize(symbol, transformer, scaler_tf, features_tf,
              lstm, scaler_lstm, features_lstm, df, device,
              results_df, avg_err_tf, avg_err_lstm, dir_acc_tf, dir_acc_lstm):
    """
    LSTM 분석과 동일한 형식 + 비교 추가
    """
    print("\n  그래프 생성 중...")

    fig = plt.figure(figsize=(20, 28))
    gs = gridspec.GridSpec(7, 2, figure=fig, hspace=0.45, wspace=0.3)

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
    ax2.fill_between(dates, recent['RSI_14'], 70,
                     where=(recent['RSI_14'] >= 70), alpha=0.2, color='red')
    ax2.fill_between(dates, recent['RSI_14'], 30,
                     where=(recent['RSI_14'] <= 30), alpha=0.2, color='green')
    ax2.set_title('RSI (14) - Overbought / Oversold', fontsize=12, fontweight='bold')
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
    ax3.set_title('MACD - Trend Signal', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── 4. 볼린저 밴드 ──
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(dates, recent['Close'], linewidth=2, label='Close', color='black')
    ax4.plot(dates, recent['BB_Upper'], linewidth=1, label='Upper', color='red', alpha=0.7)
    ax4.plot(dates, recent['BB_Mid'], linewidth=1, label='Middle', color='royalblue', alpha=0.7)
    ax4.plot(dates, recent['BB_Lower'], linewidth=1, label='Lower', color='green', alpha=0.7)
    ax4.fill_between(dates, recent['BB_Upper'], recent['BB_Lower'], alpha=0.08, color='royalblue')
    ax4.set_title('Bollinger Bands - Volatility', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Price (USD)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # ── 5. 거래량 ──
    ax5 = fig.add_subplot(gs[2, 1])
    colors_vol = ['green' if recent['Close'].iloc[i] >= recent['Close'].iloc[i-1]
                  else 'red' for i in range(len(recent))]
    ax5.bar(dates, recent['Volume'], color=colors_vol, alpha=0.6)
    ax5.plot(dates, recent['Volume_MA20'], linewidth=2, color='royalblue', label='Volume MA20')
    ax5.set_title('Volume - Trading Activity', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    # ── 6. 백테스트: 두 모델 비교 ──
    ax6 = fig.add_subplot(gs[3, :])
    ax6.plot(results_df['date'], results_df['actual'],
             linewidth=2, label='Actual Price', color='black')
    ax6.plot(results_df['date'], results_df['pred_transformer'],
             linewidth=2, label=f'Transformer (err: {avg_err_tf:.2f}%)',
             color='royalblue', linestyle='--', alpha=0.8)
    ax6.plot(results_df['date'], results_df['pred_lstm'],
             linewidth=2, label=f'LSTM (err: {avg_err_lstm:.2f}%)',
             color='darkorange', linestyle=':', alpha=0.8)
    ax6.set_title('Backtest: Transformer vs LSTM vs Actual', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Price (USD)')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # ── 7. 오차 분포 비교 ──
    ax7 = fig.add_subplot(gs[4, 0])
    ax7.hist(results_df['error_tf'], bins=20, alpha=0.6,
             color='royalblue', label=f'Transformer (avg: {avg_err_tf:.2f}%)', edgecolor='black')
    ax7.hist(results_df['error_lstm'], bins=20, alpha=0.6,
             color='darkorange', label=f'LSTM (avg: {avg_err_lstm:.2f}%)', edgecolor='black')
    ax7.axvline(x=avg_err_tf, color='royalblue', linestyle='--', linewidth=2)
    ax7.axvline(x=avg_err_lstm, color='darkorange', linestyle='--', linewidth=2)
    ax7.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Error (%)')
    ax7.set_ylabel('Count')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # ── 8. 방향 정확도 비교 ──
    ax8 = fig.add_subplot(gs[4, 1])
    models = ['Transformer', 'LSTM']
    accs = [dir_acc_tf, dir_acc_lstm]
    colors_bar = ['royalblue', 'darkorange']
    bars = ax8.bar(models, accs, color=colors_bar, alpha=0.8, edgecolor='black')
    ax8.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
    ax8.set_title('Direction Accuracy (Up / Down)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Accuracy (%)')
    ax8.set_ylim(0, 100)
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accs):
        ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')

    # ── 9. Transformer 60일 예측 ──
    ax9 = fig.add_subplot(gs[5, :])

    recent_504 = df.tail(504)[features_tf].values
    pred_tf = predict_transformer(transformer, scaler_tf, features_tf, recent_504, device)

    recent_60 = df.tail(60)[features_lstm].values
    pred_lstm = predict_lstm(lstm, scaler_lstm, features_lstm, recent_60, device)

    current_price = df['Close'].iloc[-1]
    current_date = df['Date'].iloc[-1]
    pred_date = current_date + timedelta(days=84)

    change_tf = (pred_tf - current_price) / current_price * 100
    change_lstm = (pred_lstm - current_price) / current_price * 100

    upper_tf = pred_tf * (1 + avg_err_tf / 100)
    lower_tf = pred_tf * (1 - avg_err_tf / 100)

    upper_lstm = pred_lstm * (1 + avg_err_lstm / 100)
    lower_lstm = pred_lstm * (1 - avg_err_lstm / 100)

    recent_120 = df.tail(120)
    ax9.plot(recent_120['Date'], recent_120['Close'],
             linewidth=2, label='Actual (Recent 120 Days)', color='black')
    ax9.axhline(y=current_price, color='green', linestyle='--',
                alpha=0.7, linewidth=2, label=f'Current: ${current_price:.2f}')

    # Transformer 예측
    ax9.plot([current_date, pred_date], [current_price, pred_tf],
             'b--', linewidth=1.5, alpha=0.5)
    ax9.scatter([pred_date], [pred_tf], s=300, c='royalblue',
                marker='*', zorder=5,
                label=f'Transformer: ${pred_tf:.2f} ({change_tf:+.2f}%)')
    ax9.errorbar([pred_date], [pred_tf],
                 yerr=[[pred_tf - lower_tf], [upper_tf - pred_tf]],
                 fmt='none', color='royalblue', capsize=8, linewidth=2)

    # LSTM 예측
    ax9.plot([current_date, pred_date + timedelta(days=3)],
             [current_price, pred_lstm],
             'orange', linestyle='--', linewidth=1.5, alpha=0.5)
    ax9.scatter([pred_date + timedelta(days=3)], [pred_lstm],
                s=300, c='darkorange', marker='*', zorder=5,
                label=f'LSTM: ${pred_lstm:.2f} ({change_lstm:+.2f}%)')
    ax9.errorbar([pred_date + timedelta(days=3)], [pred_lstm],
                 yerr=[[pred_lstm - lower_lstm], [upper_lstm - pred_lstm]],
                 fmt='none', color='darkorange', capsize=8, linewidth=2)

    ax9.set_title(f'{symbol} 60-Day Prediction | Transformer vs LSTM',
                  fontsize=12, fontweight='bold')
    ax9.set_ylabel('Price (USD)')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)

    # ── 10. 최종 비교 요약 ──
    ax10 = fig.add_subplot(gs[6, :])
    ax10.axis('off')

    winner_err = "Transformer" if avg_err_tf < avg_err_lstm else "LSTM"
    winner_dir = "Transformer" if dir_acc_tf > dir_acc_lstm else "LSTM"

    summary = (
        f"{'':10}{'Transformer':>20}{'LSTM':>20}{'Winner':>15}\n"
        f"{'─'*65}\n"
        f"{'Sequence':10}{'504 days (2yr)':>20}{'60 days':>20}{'':>15}\n"
        f"{'Avg Error':10}{avg_err_tf:>19.2f}%{avg_err_lstm:>19.2f}%{winner_err:>15}\n"
        f"{'Direction':10}{dir_acc_tf:>19.1f}%{dir_acc_lstm:>19.1f}%{winner_dir:>15}\n"
        f"{'─'*65}\n"
        f"{'Prediction':10}{pred_tf:>19.2f}  {pred_lstm:>19.2f}\n"
        f"{'Change':10}{change_tf:>+19.2f}%{change_lstm:>+19.2f}%\n"
    )

    ax10.text(0.05, 0.95, summary,
              transform=ax10.transAxes,
              fontsize=12, verticalalignment='top',
              fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    fig.suptitle(
        f'{symbol} LSTM vs Transformer Analysis Report  |  '
        f'{datetime.now().strftime("%Y-%m-%d")}',
        fontsize=16, fontweight='bold', y=1.005
    )

    output_path = f'data/models/{symbol.lower()}_comparison.png'
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  저장: {output_path}")

    return pred_tf, pred_lstm, current_price, change_tf, change_lstm


def print_report(symbol, pred_tf, pred_lstm, current_price,
                 change_tf, change_lstm, avg_err_tf, avg_err_lstm,
                 dir_acc_tf, dir_acc_lstm):
    """최종 결과 출력"""

    def get_signal(change_pct):
        if change_pct >= 10: return "강력 매수"
        elif change_pct >= 5: return "매수"
        elif change_pct >= -5: return "보유"
        elif change_pct >= -10: return "매도"
        else: return "강력 매도"

    print("\n" + "=" * 70)
    print(f"{symbol} 최종 분석 결과")
    print("=" * 70)

    print(f"\n현재 종가:          ${current_price:,.2f}")

    print(f"\n{'':22} {'Transformer':>15} {'LSTM':>15}")
    print(f"{'─' * 55}")
    print(f"{'시퀀스 길이':22} {'504일 (2년)':>15} {'60일':>15}")
    print(f"{'60일 후 예측':22} ${pred_tf:>14,.2f} ${pred_lstm:>14,.2f}")
    print(f"{'예상 변화':22} {change_tf:>+14.2f}% {change_lstm:>+14.2f}%")
    print(f"{'평균 오차율':22} {avg_err_tf:>14.2f}% {avg_err_lstm:>14.2f}%")
    print(f"{'방향 정확도':22} {dir_acc_tf:>14.1f}% {dir_acc_lstm:>14.1f}%")
    print(f"{'투자 신호':22} {get_signal(change_tf):>15} {get_signal(change_lstm):>15}")

    winner_err = "Transformer" if avg_err_tf < avg_err_lstm else "LSTM"
    winner_dir = "Transformer" if dir_acc_tf > dir_acc_lstm else "LSTM"
    print(f"\n오차율 우승:        {winner_err}")
    print(f"방향 정확도 우승:   {winner_dir}")

    print("\n" + "=" * 70)
    print("주의: 예측은 참고용입니다. 투자 결정은 신중하게!")
    print("=" * 70 + "\n")


def main(symbol: str):
    print("\n" + "=" * 70)
    print(f"{symbol} LSTM vs Transformer 비교 분석")
    print("=" * 70)

    transformer, scaler_tf, features_tf, lstm, scaler_lstm, features_lstm, df, device = load_models(symbol)

    results_df, avg_err_tf, avg_err_lstm, dir_acc_tf, dir_acc_lstm = run_backtest(
        transformer, scaler_tf, features_tf,
        lstm, scaler_lstm, features_lstm, df, device
    )

    pred_tf, pred_lstm, current_price, change_tf, change_lstm = visualize(
        symbol, transformer, scaler_tf, features_tf,
        lstm, scaler_lstm, features_lstm, df, device,
        results_df, avg_err_tf, avg_err_lstm, dir_acc_tf, dir_acc_lstm
    )

    print_report(
        symbol, pred_tf, pred_lstm, current_price,
        change_tf, change_lstm, avg_err_tf, avg_err_lstm,
        dir_acc_tf, dir_acc_lstm
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    args = parser.parse_args()
    main(args.symbol.upper())