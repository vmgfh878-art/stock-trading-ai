"""
Yahoo Finance 데이터 수집
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List
import os


class YahooFinanceCollector:
    """야후 파이낸스 데이터 수집기"""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str, interval: str = "1d"):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        
    def fetch_data(self, symbol: str) -> pd.DataFrame:
        """단일 종목 데이터 수집"""
        print(f"📊 {symbol} 데이터 수집 중...")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval=self.interval
            )
            
            df.reset_index(inplace=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df['Symbol'] = symbol
            
            print(f"✅ {symbol}: {len(df)} 행 수집 완료")
            return df
            
        except Exception as e:
            print(f"❌ {symbol} 수집 실패: {e}")
            return pd.DataFrame()
    
    def fetch_all(self) -> pd.DataFrame:
        """모든 종목 데이터 수집"""
        all_data = []
        
        for symbol in self.symbols:
            df = self.fetch_data(symbol)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = None):
        """CSV 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_data_{timestamp}.csv"
        
        filepath = os.path.join("data", "raw", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.to_csv(filepath, index=False)
        print(f"저장: {filepath}")
        return filepath


if __name__ == "__main__":
    print("="*60)
    print("주식 데이터 수집 시작")
    print("="*60)
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    
    collector = YahooFinanceCollector(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    df = collector.fetch_all()
    
    if not df.empty:
        print("\n" + "="*60)
        print("수집 결과")
        print("="*60)
        print(f"총 {len(df)} 행")
        print(f"\n처음 5행:")
        print(df.head())
        
        collector.save_to_csv(df)
    else:
        print("수집 실패") 
