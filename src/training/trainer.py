"""
LSTM 모델 학습 파이프라인
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
    Early Stopping
    검증 손실이 개선 안 되면 학습 중단
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
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
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        batch_size: int = 32
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        
        self.criterion = nn.MSELoss()
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        
        self.train_losses = []
        self.val_losses = []
        
    def create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs, _ = self.model(X_batch)
            
            loss = self.criterion(outputs.squeeze(), y_batch)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        
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
        print("=" * 70)
        print("학습 시작")
        print("=" * 70)
        
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_loader)
            
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.6f} "
                  f"Val Loss: {val_loss:.6f} "
                  f"Time: {epoch_time:.2f}s")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> 모델 저장 (Val Loss: {val_loss:.6f})")
            
            if early_stopping(val_loss):
                print(f"\nEarly Stopping (Epoch {epoch+1})")
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


if __name__ == "__main__":
    print("=" * 70)
    print("LSTM 주식 예측 모델 학습")
    print("=" * 70)
    
    print("\n1. 데이터 로드")
    data_path = "data/processed/aapl_with_indicators.csv"
    df = pd.read_csv(data_path)
    print(f"   데이터 크기: {df.shape}")
    
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
    
    print("\n3. 모델 초기화")
    input_size = X_train.shape[2]
    
    model = StockLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    print(f"   Input Size: {input_size}")
    print(f"   파라미터 개수: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n4. Trainer 초기화")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    trainer = LSTMTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        batch_size=32
    )
    
    train_loader, val_loader = trainer.create_dataloaders(
        X_train, y_train, X_val, y_val
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    print("\n5. 학습 시작")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=10,
        save_path='data/models/lstm_model.pt'
    )
    
    trainer.plot_losses()
    
    print("\n전체 프로세스 완료!")