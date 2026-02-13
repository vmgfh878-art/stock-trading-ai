"""
범용 주식 예측 모델 학습
AAPL, GOOGL, MSFT 어떤 종목이든 동일한 방식으로 학습

사용법:
    python src/training/train_stock.py --symbol AAPL
    python src/training/train_stock.py --symbol GOOGL
    python src/training/train_stock.py --symbol MSFT
"""
import argparse
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lstm_model import StockLSTM


# ──────────────────────────────────────────────
# 데이터 수집
# ──────────────────────────────────────────────

def download_data(symbol: str) -> pd.DataFrame:
    """
    야후 파이낸스에서 최신 데이터 다운로드
    2015년부터 오늘까지 (최대한 많은 데이터)
    """
    print(f"[{symbol}] 데이터 다운로드 중...")

    ticker = yf.Ticker(symbol)
    df = ticker.history(start='2015-01-01', interval='1d')

    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # 타임존 제거
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    df['Symbol'] = symbol

    print(f"  완료: {len(df)}행")
    print(f"  기간: {df['Date'].min().strftime('%Y-%m-%d')} ~ {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  최신 종가: ${df['Close'].iloc[-1]:.2f}")

    return df


# ──────────────────────────────────────────────
# 기술적 지표
# ──────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    기술적 지표 추가
    min_periods=1 로 결측치 최소화
    """
    print(f"  기술적 지표 계산 중...")

    # 이동평균
    for p in [5, 10, 20, 50, 200]:
        df[f'SMA_{p}'] = df['Close'].rolling(p, min_periods=1).mean()
        df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    df['RSI_14'] = 100 - (100 / (1 + avg_gain / avg_loss))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20, min_periods=1).mean()
    std = df['Close'].rolling(20, min_periods=1).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * std
    df['BB_Lower'] = df['BB_Mid'] - 2 * std
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

    # 거래량
    df['Volume_MA20'] = df['Volume'].rolling(20, min_periods=1).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    price_change = df['Close'].diff()
    obv = (price_change > 0).astype(int) - (price_change < 0).astype(int)
    df['OBV'] = (obv * df['Volume']).cumsum()

    # 가격 파생
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Loc'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    df['Price_Range'] = df['High'] - df['Low']

    df = df.bfill().ffill().fillna(0)

    print(f"  완료: {len(df.columns)}개 컬럼")
    return df


# ──────────────────────────────────────────────
# 시퀀스 생성
# ──────────────────────────────────────────────

def create_sequences(
    df: pd.DataFrame,
    sequence_length: int = 60,
    forecast_horizon: int = 60
):
    """
    시퀀스 생성 및 정규화

    입력: 과거 60일 데이터
    출력: 60일 후 종가

    Returns:
        X_train, y_train, X_val, y_val, scaler, features
    """
    print(f"  시퀀스 생성 중... (입력 {sequence_length}일 -> {forecast_horizon}일 후 예측)")

    exclude = ['Date', 'Symbol']
    features = [c for c in df.columns if c not in exclude]

    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    close_idx = features.index('Close')

    X, y = [], []
    total = len(scaled) - sequence_length - forecast_horizon + 1

    for i in range(total):
        X.append(scaled[i:i + sequence_length])
        future_idx = i + sequence_length + forecast_horizon - 1
        y.append(scaled[future_idx, close_idx])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"  X_train: {X_train.shape} | X_val: {X_val.shape}")

    return X_train, y_train, X_val, y_val, scaler, features


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────

def train(
    symbol: str,
    X_train, y_train,
    X_val, y_val
):
    """
    모델 학습
    Early Stopping 포함
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  학습 시작 (Device: {device})")

    input_size = X_train.shape[2]

    model = StockLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ).to(device)

    print(f"  파라미터: {sum(p.numel() for p in model.parameters()):,}개")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        ),
        batch_size=32, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        ),
        batch_size=32, shuffle=False
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    for epoch in range(100):
        start = time.time()

        # 학습
        model.train()
        total_train = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output, _ = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train += loss.item()

        # 검증
        model.eval()
        total_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output, _ = model(X_batch)
                total_val += criterion(output.squeeze(), y_batch).item()

        avg_train = total_train / len(train_loader)
        avg_val = total_val / len(val_loader)
        elapsed = time.time() - start

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(f"  Epoch [{epoch+1}/100] "
              f"Train: {avg_train:.6f} "
              f"Val: {avg_val:.6f} "
              f"({elapsed:.1f}s)")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs('data/models', exist_ok=True)
            torch.save(
                model.state_dict(),
                f'data/models/lstm_{symbol.lower()}.pt'
            )
            print(f"  -> 저장 (Val: {avg_val:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early Stopping (Epoch {epoch+1})")
                break

    print(f"\n  학습 완료. 최고 Val Loss: {best_val_loss:.6f}")
    return model, train_losses, val_losses


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main(symbol: str):
    print("\n" + "=" * 70)
    print(f"{symbol} LSTM 모델 학습")
    print("=" * 70)

    # 1. 데이터 수집
    df = download_data(symbol)

    # 2. 기술적 지표
    df = add_indicators(df)

    # 3. 저장
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(f'data/processed/{symbol.lower()}_processed.csv', index=False)
    print(f"\n  데이터 저장: data/processed/{symbol.lower()}_processed.csv")

    # 4. 시퀀스 생성
    X_train, y_train, X_val, y_val, scaler, features = create_sequences(df)

    # 5. 스케일러 + Feature 저장
    with open(f'data/models/scaler_{symbol.lower()}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'data/models/features_{symbol.lower()}.pkl', 'wb') as f:
        pickle.dump(features, f)

    print(f"  스케일러 저장: data/models/scaler_{symbol.lower()}.pkl")

    # 6. 학습
    model, train_losses, val_losses = train(
        symbol, X_train, y_train, X_val, y_val
    )

    # 7. 학습 곡선 저장
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{symbol} Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        f'data/models/{symbol.lower()}_training_history.png',
        dpi=200, bbox_inches='tight'
    )
    plt.close()

    print(f"  학습 곡선 저장: data/models/{symbol.lower()}_training_history.png")
    print(f"\n[{symbol}] 모든 학습 완료!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='주식 LSTM 모델 학습')
    parser.add_argument('--symbol', type=str, required=True,
                        help='종목 코드 (예: AAPL, GOOGL, MSFT)')
    args = parser.parse_args()

    main(args.symbol.upper()) 
