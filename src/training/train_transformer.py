"""
Transformer 모델 학습
시퀀스 길이 504일 (2년치)

사용법:
    python src/training/train_transformer.py --symbol AAPL
    python src/training/train_transformer.py --symbol GOOGL
    python src/training/train_transformer.py --symbol MSFT
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer_model import StockTransformer


def create_sequences(df, sequence_length=504, forecast_horizon=60):
    """
    시퀀스 생성

    LSTM:        60일 -> 60일 후 예측
    Transformer: 504일(2년) -> 60일 후 예측

    504일 = 약 2년치 영업일
    2년 패턴 (계절성, 실적 사이클 등) 학습 가능
    """
    print(f"  시퀀스 생성 (입력 {sequence_length}일 -> {forecast_horizon}일 후 예측)")

    exclude = ['Date', 'Symbol']
    features = [c for c in df.columns if c not in exclude]

    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    close_idx = features.index('Close')

    X, y = [], []
    total = len(scaled) - sequence_length - forecast_horizon + 1

    if total <= 0:
        print(f"  에러: 데이터 부족")
        print(f"  필요: {sequence_length + forecast_horizon}일")
        print(f"  보유: {len(scaled)}일")
        sys.exit(1)

    for i in range(total):
        X.append(scaled[i:i + sequence_length])
        future_idx = i + sequence_length + forecast_horizon - 1
        y.append(scaled[future_idx, close_idx])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")

    return X_train, y_train, X_val, y_val, scaler, features


def train(symbol, X_train, y_train, X_val, y_val):
    """
    Transformer 학습
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    input_size = X_train.shape[2]

    model = StockTransformer(
        input_size=input_size,
        d_model=128,
        nhead=8,
        num_layers=3,
        dropout=0.1,
        dim_feedforward=256
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터: {total_params:,}개")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 학습률 스케줄러
    # 학습 진행될수록 학습률 감소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 배치 사이즈 16 (시퀀스가 길어서 메모리 고려)
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        ),
        batch_size=16, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        ),
        batch_size=16, shuffle=False
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    print(f"\n  학습 시작\n")

    for epoch in range(100):
        start = time.time()

        # 학습
        model.train()
        total_train = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
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
                output = model(X_batch)
                total_val += criterion(output.squeeze(), y_batch).item()

        avg_train = total_train / len(train_loader)
        avg_val = total_val / len(val_loader)
        elapsed = time.time() - start

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        # 학습률 조절
        scheduler.step(avg_val)

        print(f"  Epoch [{epoch+1}/100] "
              f"Train: {avg_train:.6f} "
              f"Val: {avg_val:.6f} "
              f"({elapsed:.1f}s)")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs('data/models', exist_ok=True)
            torch.save(
                model.state_dict(),
                f'data/models/transformer_{symbol.lower()}.pt'
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


def main(symbol: str):
    print("\n" + "=" * 70)
    print(f"{symbol} Transformer 모델 학습 (시퀀스 504일)")
    print("=" * 70)

    # 기존 전처리 데이터 사용
    data_path = f'data/processed/{symbol.lower()}_processed.csv'

    if not os.path.exists(data_path):
        print(f"에러: {data_path} 없음")
        print(f"먼저 train_stock.py --symbol {symbol} 실행")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    print(f"\n데이터: {len(df)}행")

    # 시퀀스 생성 (504일)
    X_train, y_train, X_val, y_val, scaler, features = create_sequences(
        df, sequence_length=504, forecast_horizon=60
    )

    # 스케일러, Feature 저장
    with open(f'data/models/scaler_transformer_{symbol.lower()}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'data/models/features_transformer_{symbol.lower()}.pkl', 'wb') as f:
        pickle.dump(features, f)

    # 학습
    model, train_losses, val_losses = train(
        symbol, X_train, y_train, X_val, y_val
    )

    # 학습 곡선
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{symbol} Transformer Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        f'data/models/transformer_{symbol.lower()}_training.png',
        dpi=200, bbox_inches='tight'
    )
    plt.close()

    print(f"\n[{symbol}] Transformer 학습 완료!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    args = parser.parse_args()
    main(args.symbol.upper())