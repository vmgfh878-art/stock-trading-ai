"""
LSTM 모델 학습 파이프라인

학습 프로세스:
1. 데이터 로드 및 전처리
2. DataLoader 생성 (배치 단위 처리)
3. 모델, 손실함수, 옵티마이저 초기화
4. 학습 루프 (Epoch 반복)
5. 검증 및 Early Stopping
6. 모델 저장
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import time
from typing import Dict, Tuple
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import StockLSTM
from preprocessing.sequence_generator import SequenceGenerator


class EarlyStopping:
    """
    Early Stopping 구현
    
    목적: 과적합 방지
    - Validation Loss가 개선 안 되면 학습 중단
    - 최적의 모델 저장
    
    예시:
    Epoch 1: val_loss = 0.05
    Epoch 2: val_loss = 0.04 (개선) -> 모델 저장
    Epoch 3: val_loss = 0.045 (악화)
    Epoch 4: val_loss = 0.046 (악화)
    Epoch 5: val_loss = 0.047 (악화)
    -> patience=3 이면 여기서 학습 중단
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: 몇 epoch 동안 개선 없으면 중단?
            min_delta: 이 값 이상 개선되어야 "개선"으로 간주
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        검증 손실 체크
        
        Returns:
            True: 학습 중단해야 함
            False: 계속 학습
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop


class LSTMTrainer:
    """
    LSTM 모델 학습 클래스
    
    역할:
    1. 학습 루프 관리
    2. 손실 추적
    3. 모델 저장
    4. 학습 기록
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        """
        Args:
            model: LSTM 모델
            device: 'cpu' or 'cuda'
            learning_rate: 학습률
            batch_size: 배치 크기
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        
        # 손실 함수: MSE (Mean Squared Error)
        # 회귀 문제이므로 MSE 사용
        self.criterion = nn.MSELoss()
        
        # 옵티마이저: Adam
        # Adam = Adaptive Moment Estimation
        # 학습률 자동 조절, 빠른 수렴
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
        
    def create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """
        DataLoader 생성
        
        DataLoader의 역할:
        - 배치 단위로 데이터 제공
        - 메모리 효율적
        - 멀티 프로세싱 지원
        
        Args:
            X_train, y_train: 학습 데이터
            X_val, y_val: 검증 데이터
            
        Returns:
            train_loader, val_loader
        """
        # NumPy -> Torch Tensor 변환
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # TensorDataset: X와 y를 묶음
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # DataLoader: 배치 단위로 제공
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 매 epoch마다 섞기 (일반화 도움)
            drop_last=True  # 마지막 불완전한 배치 버리기
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 검증은 섞지 않음
            drop_last=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        1 Epoch 학습
        
        과정:
        1. 모델을 학습 모드로
        2. 배치별로 순전파
        3. 손실 계산
        4. 역전파
        5. 가중치 업데이트
        
        Returns:
            평균 손실
        """
        self.model.train()  # 학습 모드 (Dropout 활성화)
        total_loss = 0.0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # 데이터를 device로 이동 (GPU 사용 시)
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # 그래디언트 초기화
            # PyTorch는 그래디언트 누적되므로 매번 초기화 필요
            self.optimizer.zero_grad()
            
            # 순전파
            outputs, _ = self.model(X_batch)
            
            # 손실 계산
            # outputs: (batch_size, 1)
            # y_batch: (batch_size,)
            # squeeze로 차원 맞추기
            loss = self.criterion(outputs.squeeze(), y_batch)
            
            # 역전파
            # 각 파라미터의 그래디언트 계산
            loss.backward()
            
            # 그래디언트 클리핑 (폭발 방지)
            # 그래디언트가 너무 크면 학습 불안정
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 가중치 업데이트
            # w = w - lr × gradient
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # 평균 손실 반환
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        검증 (Validation)
        
        학습 데이터 외에 별도 데이터로 성능 평가
        과적합 여부 확인
        
        Returns:
            평균 검증 손실
        """
        self.model.eval()  # 평가 모드 (Dropout 비활성화)
        total_loss = 0.0
        
        # 그래디언트 계산 안 함 (메모리 절약, 속도 향상)
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs, _ = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_path: str = 'data/models/lstm_model.pt'
    ) -> Dict:
        """
        전체 학습 루프
        
        Args:
            train_loader: 학습 데이터
            val_loader: 검증 데이터
            epochs: 최대 학습 횟수
            early_stopping_patience: Early Stopping patience
            save_path: 모델 저장 경로
            
        Returns:
            학습 기록 딕셔너리
        """
        print("=" * 70)
        print("학습 시작")
        print("=" * 70)
        
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 학습
            train_loss = self.train_epoch(train_loader)
            
            # 검증
            val_loss = self.validate(val_loader)
            
            # 기록
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 경과 시간
            epoch_time = time.time() - start_time
            
            # 출력
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.6f} "
                  f"Val Loss: {val_loss:.6f} "
                  f"Time: {epoch_time:.2f}s")
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> 모델 저장 (Val Loss: {val_loss:.6f})")
            
            # Early Stopping 체크
            if early_stopping(val_loss):
                print(f"\nEarly Stopping! (Epoch {epoch+1})")
                break
        
        print("=" * 70)
        print("학습 완료")
        print(f"최고 Val Loss: {best_val_loss:.6f}")
        print("=" * 70)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def plot_losses(self, save_path: str = 'data/models/training_history.png'):
        """
        학습 곡선 시각화
        
        Train Loss vs Val Loss 그래프
        - 두 선이 비슷: 정상 학습
        - Val Loss >> Train Loss: 과적합
        - 둘 다 감소 안 함: 학습 안 됨
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"학습 곡선 저장: {save_path}")
        plt.close()


# 전체 학습 파이프라인
if __name__ == "__main__":
    print("=" * 70)
    print("LSTM 주식 예측 모델 학습")
    print("=" * 70)
    
    # 1. 데이터 준비
    print("\n1. 데이터 로드")
    data_path = "data/processed/aapl_with_indicators.csv"
    df = pd.read_csv(data_path)
    print(f"   데이터 크기: {df.shape}")
    
    # 2. 시퀀스 생성
    print("\n2. 시퀀스 생성")
    generator = SequenceGenerator(
        sequence_length=60,
        target_column='Close'
    )
    
    X_train, y_train, X_val, y_val = generator.prepare_data(
        df,
        validation_split=0.2
    )
    
    generator.save_scaler()
    
    # 3. 모델 생성
    print("\n3. 모델 초기화")
    input_size = X_train.shape[2]  # Feature 개수
    
    model = StockLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    print(f"   Input Size: {input_size}")
    print(f"   파라미터 개수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Trainer 생성
    print("\n4. Trainer 초기화")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    trainer = LSTMTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        batch_size=32
    )
    
    # 5. DataLoader 생성
    train_loader, val_loader = trainer.create_dataloaders(
        X_train, y_train, X_val, y_val
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # 6. 학습 시작
    print("\n5. 학습 시작")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=10,
        save_path='data/models/lstm_model.pt'
    )
    
    # 7. 학습 곡선 저장
    trainer.plot_losses()
    
    print("\n전체 프로세스 완료!")