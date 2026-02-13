"""
Transformer 기반 주식 예측 모델

LSTM과 차이:
- 504일(2년) 시퀀스 전체를 한 번에 처리
- Attention으로 중요한 날짜 스스로 판단
- 2015년 데이터에서 2년치 패턴 단위로 학습
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    위치 정보 주입

    Transformer는 순서 개념이 없어서
    "이게 몇 번째 날인지" 직접 알려줘야 함

    sin/cos으로 각 위치마다 고유한 값 생성:
    Day 1:   [sin(1), cos(1), sin(1/100), ...]
    Day 252: [sin(252), cos(252), ...]
    Day 504: [sin(504), cos(504), ...]
    """
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class StockTransformer(nn.Module):
    """
    주식 예측용 Transformer

    구조:
    Input (batch, 504일, 32 features)
      -> Linear (32 -> 128)
      -> Positional Encoding
      -> Transformer Encoder x 3층
         (각 층마다 Multi-Head Attention 수행)
      -> Global Average Pooling
      -> FC (128 -> 64 -> 1)
    Output (batch, 1)
    """

    def __init__(
        self,
        input_size: int = 32,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        dim_feedforward: int = 256
    ):
        """
        Args:
            input_size:      Feature 개수 (32)
            d_model:         내부 차원 (128)
            nhead:           Attention Head 수
                             d_model / nhead = 128 / 8 = 16
                             8개 관점에서 동시에 Attention
            num_layers:      Encoder 층 수 (3)
            dropout:         드롭아웃
            dim_feedforward: FFN 차원 (256)
        """
        super().__init__()

        # 32 -> 128 차원 변환
        self.input_projection = nn.Linear(input_size, d_model)

        # 위치 정보 인코딩
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # 예측값 출력
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 504, 32)
        Returns:
            (batch, 1)
        """
        # (batch, 504, 32) -> (batch, 504, 128)
        x = self.input_projection(x)

        # 위치 정보 추가
        x = self.pos_encoding(x)

        # Transformer: 504일 전체를 한 번에 처리
        # 각 날짜가 다른 모든 날짜에 Attention
        x = self.transformer_encoder(x)

        # 504일 평균 -> (batch, 128)
        x = x.mean(dim=1)

        # (batch, 128) -> (batch, 1)
        return self.fc(x) 
