"""
기술적 지표 계산 모듈
"""
import pandas as pd
import numpy as np
import ta


class TechnicalIndicators:
    """기술적 지표 계산"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def add_moving_averages(self) -> pd.DataFrame:
        """이동평균 추가"""
        for period in [5, 10, 20, 50, 200]:
            self.df[f'SMA_{period}'] = self.df['Close'].rolling(window=period).mean()
            self.df[f'EMA_{period}'] = self.df['Close'].ewm(span=period, adjust=False).mean()
        return self.df
    
    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        """RSI 추가"""
        self.df['RSI_14'] = ta.momentum.RSIIndicator(
            close=self.df['Close'],
            window=period
        ).rsi()
        return self.df
    
    def add_macd(self) -> pd.DataFrame:
        """MACD 추가"""
        macd = ta.trend.MACD(close=self.df['Close'])
        self.df['MACD'] = macd.macd()
        self.df['MACD_Signal'] = macd.macd_signal()
        self.df['MACD_Hist'] = macd.macd_diff()
        return self.df
    
    def add_bollinger_bands(self, period: int = 20) -> pd.DataFrame:
        """볼린저 밴드 추가"""
        bb = ta.volatility.BollingerBands(close=self.df['Close'], window=period)
        self.df[f'BB_High_{period}'] = bb.bollinger_hband()
        self.df[f'BB_Low_{period}'] = bb.bollinger_lband()
        self.df[f'BB_Mid_{period}'] = bb.bollinger_mavg()
        self.df[f'BB_Width_{period}'] = bb.bollinger_wband()
        return self.df
    
    def add_volume_indicators(self) -> pd.DataFrame:
        """거래량 지표 추가"""
        self.df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            close=self.df['Close'],
            volume=self.df['Volume']
        ).on_balance_volume()
        
        self.df['Volume_SMA_20'] = self.df['Volume'].rolling(window=20).mean()
        self.df['Volume_Ratio'] = self.df['Volume'] / self.df['Volume_SMA_20']
        return self.df
    
    def add_price_features(self) -> pd.DataFrame:
        """가격 파생 특징 추가"""
        self.df['Daily_Return'] = self.df['Close'].pct_change() * 100
        self.df['High_Low_Ratio'] = self.df['High'] / self.df['Low']
        self.df['Close_Loc'] = (
            (self.df['Close'] - self.df['Low']) / 
            (self.df['High'] - self.df['Low'])
        )
        self.df['Price_Range'] = self.df['High'] - self.df['Low']
        return self.df
    
    def add_all(self) -> pd.DataFrame:
        """모든 지표 추가"""
        print("📊 기술적 지표 계산 시작...")
        
        self.add_moving_averages()
        print("  ✓ 이동평균")
        
        self.add_rsi()
        print("  ✓ RSI")
        
        self.add_macd()
        print("  ✓ MACD")
        
        self.add_bollinger_bands()
        print("  ✓ 볼린저 밴드")
        
        self.add_volume_indicators()
        print("  ✓ 거래량 지표")
        
        self.add_price_features()
        print("  ✓ 가격 파생 특징")
        
        print(f"✅ 완료! 총 {len(self.df.columns)} 개 컬럼")
        
        return self.df


if __name__ == "__main__":
    from glob import glob
    import os
    
    # 데이터 로드
    csv_files = glob("data/raw/*.csv")
    if not csv_files:
        print("❌ CSV 파일 없음!")
        exit(1)
    
    latest = max(csv_files, key=os.path.getctime)
    df = pd.read_csv(latest)
    
    print(f"📂 원본: {latest}")
    print(f"   크기: {df.shape}")
    
    # AAPL만
    aapl = df[df['Symbol'] == 'AAPL'].copy()
    aapl = aapl.sort_values('Date').reset_index(drop=True)
    
    print(f"\n🍎 AAPL: {aapl.shape}")
    
    # 지표 추가
    ti = TechnicalIndicators(aapl)
    result = ti.add_all()
    
    print(f"\n📊 처리 후: {result.shape}")
    
    # 결측치
    nulls = result.isnull().sum()
    print(f"\n결측치:\n{nulls[nulls > 0]}")
    
    # 저장
    output = "data/processed/aapl_with_indicators.csv"
    os.makedirs("data/processed", exist_ok=True)
    result.to_csv(output, index=False)
    print(f"\n💾 저장: {output}")