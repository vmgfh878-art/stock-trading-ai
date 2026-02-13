# Stock Trading AI

LSTM + Transformer 딥러닝 기반 주식 가격 예측 시스템
미국 주요 종목(AAPL, GOOGL, MSFT)의 **60일 후 종가를 예측**하고 두 모델의 성능을 비교합니다.

---

## 프로젝트 구조

```
stock-trading-ai/
│
├── src/
│   ├── models/
│   │   ├── lstm_model.py           # LSTM 모델
│   │   └── transformer_model.py    # Transformer 모델
│   │
│   ├── training/
│   │   ├── train_stock.py          # LSTM 학습
│   │   └── train_transformer.py    # Transformer 학습
│   │
│   ├── utils/
│   │   ├── analyze_stock.py        # LSTM 분석 및 시각화
│   │   └── analyze_transformer.py  # LSTM vs Transformer 비교
│   │
│   ├── preprocessing/
│   │   ├── technical_indicators.py # 기술적 지표 32개 계산
│   │   └── sequence_generator.py   # 시퀀스 생성
│   │
│   └── data_collection/
│       └── yahoo_finance.py        # 야후 파이낸스 데이터 수집
│
├── scripts/
│   ├── train_all.py                # LSTM 3종목 한 번에 학습
│   ├── analyze_all.py              # LSTM 3종목 한 번에 분석
│   ├── train_transformer_all.py    # Transformer 3종목 학습
│   └── compare_all.py              # LSTM vs Transformer 비교
│
├── data/
│   ├── processed/                  # 전처리된 데이터 (.csv)
│   └── models/                     # 학습된 모델, 스케일러, 분석 그래프
│
├── README.md
└── requirements.txt
```

---

## 시스템 흐름

```
야후 파이낸스 API
      ↓
원본 주가 데이터 (OHLCV)
      ↓
32개 기술적 지표 계산
(SMA, EMA, RSI, MACD, Bollinger Bands 등)
      ↓
시퀀스 생성
  LSTM:        과거 60일  → 60일 후 예측
  Transformer: 과거 504일 → 60일 후 예측
      ↓
모델 학습 + 백테스트
      ↓
성능 비교 + 시각화
```

---

## 사용 기술

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.14 |
| 딥러닝 | PyTorch |
| 데이터 수집 | yfinance API |
| 데이터 처리 | pandas, numpy, scikit-learn |
| 시각화 | matplotlib |
| 버전 관리 | Git / GitHub |

---

## 모델 구조 비교

| 항목 | LSTM | Transformer |
|------|------|-------------|
| 시퀀스 길이 | 60일 | 504일 (2년) |
| 처리 방식 | 순차 처리 | 전체 동시 처리 |
| 핵심 기술 | Gate (Forget/Input/Output) | Multi-Head Attention |
| 장기 패턴 | 제한적 | 강함 |
| 학습 속도 | 빠름 | 느림 |
| 파라미터 | 약 150,000개 | 약 200,000개 |

---

## 기술적 지표 (32개 Feature)

| 분류 | 지표 |
|------|------|
| 이동평균 | SMA 5/10/20/50/200, EMA 5/10/20/50/200 |
| 모멘텀 | RSI(14), MACD, MACD Signal, MACD Histogram |
| 변동성 | Bollinger Bands Upper/Mid/Lower/Width |
| 거래량 | Volume MA20, Volume Ratio, OBV |
| 가격 파생 | Daily Return, High-Low Ratio, Close Location |

---

## 분석 결과

### LSTM 단독 결과

| 종목 | 현재가 | 60일 후 예측 | 오차율 | 방향 정확도 |
|------|--------|-------------|--------|------------|
| AAPL | $261.73 | $225.54 (-13.83%) | 11.25% | 63.5% |
| GOOGL | $309.00 | $266.47 (-13.76%) | 10.83% | 65.6% |
| MSFT | $401.84 | $377.19 (-6.13%) | 9.62% | 71.9% |

### LSTM vs Transformer 비교

| 종목 | LSTM 오차 | TF 오차 | LSTM 방향 | TF 방향 | 승자 |
|------|-----------|---------|-----------|---------|------|
| AAPL | 12.73% | 17.46% | 72.4% | 47.1% | **LSTM** |
| GOOGL | 13.54% | 13.76% | 73.3% | 69.0% | **LSTM** |
| MSFT | 9.99% | **9.92%** | 76.8% | **83.0%** | **Transformer** |

> 종목마다 최적 모델이 다름 → 단일 모델보다 앙상블이 더 효과적일 가능성

---

## 실행 방법

### 환경 설정

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### LSTM 학습 및 분석

```bash
# 개별 종목 학습
python src/training/train_stock.py --symbol AAPL

# 3종목 한 번에 학습
python scripts/train_all.py

# 3종목 한 번에 분석
python scripts/analyze_all.py
```

### Transformer 학습 및 비교

```bash
# Transformer 학습
python scripts/train_transformer_all.py

# LSTM vs Transformer 비교
python scripts/compare_all.py
```

### 결과 파일

```
data/models/
├── aapl_analysis.png        # AAPL LSTM 분석
├── googl_analysis.png       # GOOGL LSTM 분석
├── msft_analysis.png        # MSFT LSTM 분석
├── aapl_comparison.png      # AAPL LSTM vs Transformer
├── googl_comparison.png     # GOOGL LSTM vs Transformer
└── msft_comparison.png      # MSFT LSTM vs Transformer
```

---

## 주요 발견

```
1. LSTM이 대부분 종목에서 안정적
   짧은 시퀀스(60일)가 최근 패턴에 더 민감하게 반응

2. Transformer는 MSFT에서 강세
   긴 시퀀스(504일)가 MSFT의 장기 패턴에 잘 맞음

3. 두 모델 모두 방향 정확도 65~83%
   동전 던지기(50%)보다 유의미하게 높음
   단, 정확한 가격 예측은 오차 10~17%로 한계 존재

4. 단기 예측 한계 확인
   뉴스, 실적, 금리 등 외부 요인 반영 불가
   기술적 패턴만으로는 단기 예측에 한계
```

---

## 주의사항

- 과거 데이터 기반 예측이며 미래를 보장하지 않습니다
- 뉴스, 실적, 거시경제 등 외부 요인은 반영되지 않습니다
- 실제 투자 결정은 본인 판단과 책임 하에 진행하세요

---

## 개발자

vmgfh878-art | [GitHub](https://github.com/vmgfh878-art/stock-trading-ai)
