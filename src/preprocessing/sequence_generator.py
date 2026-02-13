"""
LSTM을 위한 시퀀스 데이터 생성

협업 포인트:
- ML 엔지니어: 이 모듈로 학습 데이터 생성
- 백엔드: 실시간 예측 시 최근 60일 데이터를 시퀀스로 변환
"""
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class SequenceGenerator:
    """
    시계열 데이터를 LSTM 입력 형태로 변환
    
    핵심 개념:
    1. Sliding Window: 과거 N일 → 미래 1일 예측
    2. Normalization: 0~1 범위로 스케일링 (LSTM 학습 효율 UP)
    3. Train/Val Split: 시계열은 셔플 금지! 순서 유지
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        target_column: str = 'Close',
        feature_columns: List[str] = None
    ):
        """
        Args:
            sequence_length: 입력 시퀀스 길이 (과거 몇 일?)
            target_column: 예측할 타겟 (보통 'Close')
            feature_columns: 사용할 Feature들 (None이면 전부 사용)
        """
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        데이터 전처리 및 시퀀스 생성
        
        프로세스:
        1. Feature 선택
        2. 결측치 제거
        3. 정규화 (MinMaxScaler)
        4. 시퀀스 생성
        5. Train/Val 분할
        
        Args:
            df: 전처리된 DataFrame
            validation_split: 검증 데이터 비율
            
        Returns:
            X_train, y_train, X_val, y_val
        """
        print("\n" + "="*60)
        print("📊 시퀀스 데이터 생성 시작")
        print("="*60)
        
        # 1. Feature 선택
        if self.feature_columns is None:
            # 날짜, 심볼 제외한 모든 수치형 컬럼
            exclude = ['Date', 'Symbol']
            self.feature_columns = [
                col for col in df.columns 
                if col not in exclude
            ]
        
        print(f"\n📌 사용할 Features: {len(self.feature_columns)}개")
        print(f"   - {', '.join(self.feature_columns[:5])}...")
        
        # 2. 필요한 컬럼만 추출
        df_features = df[self.feature_columns].copy()
        
        # 3. 결측치 제거
        print(f"\n🧹 결측치 처리:")
        print(f"   처리 전: {df_features.shape}")
        df_features = df_features.dropna()
        print(f"   처리 후: {df_features.shape}")
        
        if len(df_features) < self.sequence_length + 1:
            raise ValueError(
                f"데이터가 부족합니다! "
                f"최소 {self.sequence_length + 1}행 필요"
            )
        
        # 4. 정규화 (0~1 범위)
        print(f"\n📏 데이터 정규화 (MinMaxScaler):")
        print(f"   예: Close {df_features['Close'].min():.2f}~{df_features['Close'].max():.2f}")
        
        scaled_data = self.scaler.fit_transform(df_features)
        
        print(f"   → 0~1 범위로 변환")
        
        # 5. 시퀀스 생성
        X, y = self._create_sequences(scaled_data, df_features)
        
        print(f"\n🔢 시퀀스 생성 완료:")
        print(f"   X shape: {X.shape} (샘플, 타임스텝, Features)")
        print(f"   y shape: {y.shape} (샘플,)")
        
        # 6. Train/Val 분할 (시계열은 순서 유지!)
        split_idx = int(len(X) * (1 - validation_split))
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        print(f"\n✂️  Train/Val 분할:")
        print(f"   Train: {X_train.shape[0]} 샘플")
        print(f"   Val:   {X_val.shape[0]} 샘플")
        
        print("\n" + "="*60)
        print("✅ 데이터 준비 완료!")
        print("="*60)
        
        return X_train, y_train, X_val, y_val
    
    def _create_sequences(
        self,
        data: np.ndarray,
        df_original: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding Window로 시퀀스 생성
        
        예시 (sequence_length=3):
        데이터: [1, 2, 3, 4, 5]
        
        Seq 1: [1, 2, 3] → 4
        Seq 2: [2, 3, 4] → 5
        
        딥러닝 관점:
        - 더 긴 sequence_length = 더 많은 과거 정보
        - 하지만 너무 길면: 학습 느림, 기울기 소실 위험
        - 보통 30~60일이 적절
        """
        X = []  # 입력 시퀀스
        y = []  # 타겟 (다음 날 종가)
        
        # Close 컬럼의 인덱스 찾기
        target_idx = list(df_original.columns).index(self.target_column)
        
        for i in range(len(data) - self.sequence_length):
            # 입력: i일 ~ i+sequence_length일
            X.append(data[i:i + self.sequence_length])
            
            # 타겟: i+sequence_length일의 종가
            y.append(data[i + self.sequence_length, target_idx])
        
        return np.array(X), np.array(y)
    
    def save_scaler(self, filepath: str = "data/models/scaler.pkl"):
        """
        스케일러 저장
        
        협업 중요!
        - 학습 시: 스케일러 저장
        - 예측 시: 같은 스케일러로 변환 필수!
        - 안 그러면 예측 값이 이상해짐
        
        백엔드 연동:
        - API에서 이 스케일러 로드해서 실시간 데이터 변환
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 스케일러 저장: {filepath}")
    
    def load_scaler(self, filepath: str = "data/models/scaler.pkl"):
        """스케일러 로드"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"📂 스케일러 로드: {filepath}")
    
    def inverse_transform_prediction(self, pred: np.ndarray) -> np.ndarray:
        """
        정규화된 예측값을 원래 스케일로 복원
        
        중요!
        - LSTM 출력: 0~1 범위
        - 실제 가격으로 변환 필요
        
        예: 0.75 → $182.50
        """
        # pred가 1D면 2D로 변환
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        
        # 전체 Feature 개수만큼 더미 컬럼 생성
        dummy = np.zeros((len(pred), len(self.feature_columns)))
        
        # Close 컬럼 위치에 예측값 넣기
        target_idx = self.feature_columns.index(self.target_column)
        dummy[:, target_idx] = pred.flatten()
        
        # 역변환
        inversed = self.scaler.inverse_transform(dummy)
        
        # Close 컬럼만 반환
        return inversed[:, target_idx]

# 테스트
if __name__ == "__main__":
    import pandas as pd
    
    print("🚀 프로그램 시작!")
    
    # 전처리된 데이터 로드
    data_path = "data/processed/aapl_with_indicators.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ 파일 없음: {data_path}")
        exit(1)
    
    df = pd.read_csv(data_path)
    print(f"📂 데이터 로드: {data_path}")
    print(f"   크기: {df.shape}")
    
    print("\n🔧 SequenceGenerator 생성 중...")
    
    # 시퀀스 생성
    generator = SequenceGenerator(
        sequence_length=60,
        target_column='Close'
    )
    
    print("✅ Generator 생성 완료")
    print("\n📊 prepare_data 실행 중...")
    
    try:
        X_train, y_train, X_val, y_val = generator.prepare_data(
            df,
            validation_split=0.2
        )
        
        print("\n✅ prepare_data 완료!")
        
        # 스케일러 저장
        generator.save_scaler()
        
        # 결과 확인
        print(f"\n📊 최종 데이터 shape:")
        print(f"   X_train: {X_train.shape}")
        print(f"   y_train: {y_train.shape}")
        print(f"   X_val: {X_val.shape}")
        print(f"   y_val: {y_val.shape}")
        
        print(f"\n🎉 시퀀스 생성 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 에러 발생:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc() 
