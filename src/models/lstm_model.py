"""
PyTorch LSTM 모델

협업 포인트:
- ML 엔지니어: 이 모델 학습 및 튜닝
- 백엔드: 학습된 .pt 파일 로드해서 API에서 예측
- 프론트: API 호출 → 예측 결과 시각화
"""
import torch
import torch.nn as nn
from typing import Tuple


class StockLSTM(nn.Module):
    """
    주식 가격 예측용 LSTM 모델
    
    구조:
    Input → LSTM Layer 1 → Dropout → LSTM Layer 2 → Dropout → FC → Output
    
    딥러닝 이론:
    - 2개 LSTM Layer: 더 복잡한 패턴 학습
    - Dropout: 과적합 방지 (랜덤하게 뉴런 끄기)
    - FC (Fully Connected): 최종 예측값 생성
    """
    
    def __init__(
        self,
        input_size: int,      # Feature 개수 (32)
        hidden_size: int,     # LSTM hidden unit 개수 (128)
        num_layers: int,      # LSTM layer 개수 (2)
        dropout: float,       # Dropout 비율 (0.2)
        output_size: int = 1  # 출력 개수 (1, 종가만 예측)
    ):
        """
        Args:
            input_size: 입력 Feature 개수
            hidden_size: LSTM hidden state 크기
            num_layers: LSTM 레이어 개수
            dropout: Dropout 비율 (0~1)
            output_size: 출력 크기 (기본 1, 종가 예측)
        """
        super(StockLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        """
        batch_first=True: 
        - 입력 shape: (batch_size, sequence_length, input_size)
        - False면: (sequence_length, batch_size, input_size)
        - batch_first=True가 직관적!
        
        dropout:
        - LSTM layer 사이에만 적용
        - 마지막 layer에는 적용 안 됨
        
        bidirectional:
        - False: 과거→미래 방향만
        - True: 과거→미래, 미래→과거 양방향
        - 주식은 미래 데이터 못 쓰니까 False!
        """
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout Layer
        """
        과적합(Overfitting) 방지:
        - 학습 시: 랜덤하게 20% 뉴런 비활성화
        - 테스트 시: 모든 뉴런 사용
        - 더 일반화된 모델 학습
        """
        self.dropout = nn.Dropout(dropout)
        
        # Fully Connected Layer
        """
        LSTM 출력 → 최종 예측값
        - 입력: hidden_size (128)
        - 출력: 1 (예측 종가)
        """
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        순전파 (Forward Pass)
        
        데이터 흐름:
        1. 입력 x → LSTM
        2. LSTM 출력 → Dropout
        3. Dropout 출력 → FC Layer
        4. FC 출력 = 최종 예측
        
        Args:
            x: 입력 텐서 (batch_size, sequence_length, input_size)
               예: (32, 60, 32)
            hidden: 초기 hidden state (선택, 보통 None)
            
        Returns:
            output: 예측값 (batch_size, 1)
            hidden: 마지막 hidden state (나중에 연속 예측 시 사용)
        """
        # LSTM 통과
        """
        lstm_out shape: (batch_size, sequence_length, hidden_size)
                        (32, 60, 128)
        
        각 타임스텝마다 출력이 나오지만,
        우리는 마지막 타임스텝(60일째)만 사용!
        """
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # 마지막 타임스텝의 출력만 추출
        """
        lstm_out[:, -1, :]:
        - [:, -1, :] = 모든 배치의 마지막 타임스텝
        - shape: (batch_size, hidden_size)
        -         (32, 128)
        """
        last_output = lstm_out[:, -1, :]
        
        # Dropout 적용
        """
        과적합 방지
        학습 시에만 작동 (model.train())
        """
        dropped = self.dropout(last_output)
        
        # Fully Connected Layer → 최종 예측
        """
        (32, 128) → (32, 1)
        각 샘플마다 1개 예측값
        """
        output = self.fc(dropped)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hidden state 초기화
        
        LSTM은 2개 state 필요:
        1. h_0: hidden state (단기 기억)
        2. c_0: cell state (장기 기억)
        
        Args:
            batch_size: 배치 크기
            device: 'cpu' or 'cuda'
            
        Returns:
            (h_0, c_0): 초기화된 hidden states
        """
        h_0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size
        ).to(device)
        
        c_0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size
        ).to(device)
        
        return (h_0, c_0)


# 모델 테스트
if __name__ == "__main__":
    print("="*60)
    print(" LSTM 모델 테스트")
    print("="*60)
    
    # 하이퍼파라미터
    input_size = 32      # Feature 개수
    hidden_size = 128    # LSTM hidden units
    num_layers = 2       # LSTM layers
    dropout = 0.2        # Dropout 비율
    sequence_length = 60 # 입력 시퀀스 길이
    batch_size = 32      # 배치 크기
    
    print(f"\n 모델 설정:")
    print(f"   Input Size: {input_size}")
    print(f"   Hidden Size: {hidden_size}")
    print(f"   Num Layers: {num_layers}")
    print(f"   Dropout: {dropout}")
    
    # 모델 생성
    model = StockLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    print(f"\n  모델 구조:")
    print(model)
    
    # 파라미터 개수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n파라미터:")
    print(f"   전체: {total_params:,}")
    print(f"   학습 가능: {trainable_params:,}")
    
    # 더미 데이터로 테스트
    print(f"\n 순전파 테스트:")
    dummy_input = torch.randn(batch_size, sequence_length, input_size)
    print(f"   입력 shape: {dummy_input.shape}")
    
    # 순전파
    output, hidden = model(dummy_input)
    
    print(f"   출력 shape: {output.shape}")
    print(f"   Hidden state shape: {hidden[0].shape}")
    print(f"   Cell state shape: {hidden[1].shape}")
    
    # 예측값 확인
    print(f"\n예측값 샘플 (정규화된 값):")
    print(f"   {output[:5].squeeze().detach().numpy()}")
    
    print(f"\n모델 테스트 완료!")
    print("="*60) 
 
