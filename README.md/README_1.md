# Stock Trading AI

LSTM 딥러닝 기반 주식 가격 예측 시스템
AAPL, GOOGL, MSFT 3개 종목에 대해 **60일 후 종가를 예측**하고 투자 신호를 제공합니다.

> **현재 개발 중**: Transformer 모델 추가 작업 진행 중 (LSTM vs Transformer 성능 비교 예정)

---

## 프로젝트 구조

```
stock-trading-ai/
│
├── src/
│   ├── models/
│   │   ├── lstm_model.py           # LSTM 모델 구조
│   │   └── transformer_model.py    # Transformer 모델 구조 (개발 중)
│   │
│   ├── training/
│   │   ├── train_stock.py          # LSTM 학습 (범용)
│   │   └── train_transformer.py    # Transformer 학습 (개발 중)
│   │
│   ├── utils/
│   │   ├── analyze_stock.py        # LSTM 분석 및 시각화
│   │   └── analyze_transformer.py  # 모델 비교 분석 (개발 중)
│   │
│   ├── preprocessing/
│   │   ├── technical_indicators.py # 기술적 지표 계산
│   │   └── sequence_generator.py   # 시퀀스 생성
│   │
│   └── data_collection/
│       └── yahoo_finance.py        # 야후 파이낸스 데이터 수집
│
├── scripts/
│   ├── train_all.py                # LSTM 3종목 한 번에 학습
│   ├── analyze_all.py              # LSTM 3종목 한 번에 분석
│   ├── train_transformer_all.py    # Transformer 3종목 학습 (개발 중)
│   └── compare_all.py              # LSTM vs Transformer 비교 (개발 중)
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
(SMA, EMA, RSI, MACD, Bollinger Bands, Volume 등)
      ↓
시퀀스 생성
(과거 60일 → 60일 후 종가)
      ↓
LSTM 학습
      ↓
백테스트 (과거 예측 정확도 측정)
      ↓
60일 후 예측 + 시각화 (9개 그래프)
```

---

## 사용 기술

| 분류 | 기술 |
|------|------|
| 언어 | Python 3.14 |
| 딥러닝 | PyTorch (LSTM, Transformer) |
| 데이터 수집 | yfinance API |
| 데이터 처리 | pandas, numpy, scikit-learn |
| 시각화 | matplotlib |
| 버전 관리 | Git / GitHub |

---

## 기술적 지표 (32개 Feature)

| 분류 | 지표 |
|------|------|
| 이동평균 | SMA 5/10/20/50/200, EMA 5/10/20/50/200 |
| 모멘텀 | RSI (14), MACD, MACD Signal, MACD Histogram |
| 변동성 | Bollinger Bands (Upper / Mid / Lower / Width) |
| 거래량 | Volume MA20, Volume Ratio, OBV |
| 가격 파생 | Daily Return, High-Low Ratio, Close Location, Price Range |

---

## LSTM 모델 구조

```
Input  (batch, 60일, 32 features)
  → LSTM Layer 1  (hidden: 128)
  → Dropout (0.2)
  → LSTM Layer 2  (hidden: 128)
  → 마지막 타임스텝
  → Dropout (0.2)
  → Fully Connected
Output (batch, 1)  ← 60일 후 종가
```

| 항목 | 값 |
|------|-----|
| 파라미터 수 | 약 150,000개 |
| 손실 함수 | MSE |
| 옵티마이저 | Adam (lr=0.001) |
| Early Stopping | patience=15 |
| 시퀀스 길이 | 60일 |

---

## Transformer 모델 구조 (개발 중)

```
Input  (batch, 504일, 32 features)   ← 2년치 시퀀스
  → Linear Projection  (32 → 128)
  → Positional Encoding
  → Transformer Encoder × 3층
     (Multi-Head Attention, nhead=8)
  → Global Average Pooling
  → FC (128 → 64 → 1)
Output (batch, 1)  ← 60일 후 종가
```

| 항목 | LSTM | Transformer |
|------|------|-------------|
| 시퀀스 길이 | 60일 | 504일 (2년) |
| 처리 방식 | 순차 처리 | 전체 동시 처리 |
| 장기 패턴 학습 | 제한적 | 강함 |
| 학습 속도 | 빠름 | 느림 |
| 상태 | 완료 | 개발 중 |

---

## 현재 결과 (LSTM)

| 종목 | 현재가 | 60일 후 예측 | 예상 변화 | 오차율 | 방향 정확도 |
|------|--------|-------------|----------|--------|------------|
| AAPL | $261.73 | $225.54 | -13.83% | 11.25% | 63.5% |
| GOOGL | $309.00 | $266.47 | -13.76% | 10.83% | 65.6% |
| MSFT | $401.84 | $377.19 | -6.13% | 9.62% | 71.9% |

> 방향 정확도 65~72% = 동전 던지기(50%)보다 유의미하게 높음

---

## 실행 방법

### 환경 설정

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### LSTM 학습

```bash
# 개별 종목
python src/training/train_stock.py --symbol AAPL
python src/training/train_stock.py --symbol GOOGL
python src/training/train_stock.py --symbol MSFT

# 3종목 한 번에
python scripts/train_all.py
```

### LSTM 분석

```bash
# 개별 종목
python src/utils/analyze_stock.py --symbol AAPL

# 3종목 한 번에
python scripts/analyze_all.py
```

### 결과 파일

```
data/models/
├── aapl_analysis.png     # AAPL 분석 그래프
├── googl_analysis.png    # GOOGL 분석 그래프
└── msft_analysis.png     # MSFT 분석 그래프
```

---

## 개발 로드맵

- [x] 데이터 수집 파이프라인 (yfinance API)
- [x] 기술적 지표 32개 계산
- [x] LSTM 모델 학습 및 백테스트
- [x] 시각화 (9개 그래프)
- [x] AAPL / GOOGL / MSFT 분석 완료
- [ ] Transformer 모델 학습 (진행 중)
- [ ] LSTM vs Transformer 성능 비교
- [ ] FastAPI 백엔드 구축
- [ ] React 프론트엔드 연동
- [ ] 자동화 (스케줄러 + 알림)

---

## 주의사항

- 이 시스템은 **과거 데이터 기반 예측**이며 100% 정확하지 않습니다
- 뉴스, 실적 발표, 거시경제 지표 등 **외부 요인은 반영되지 않습니다**
- 실제 투자 결정은 본인의 판단과 책임 하에 진행하세요

---

## 개발자

vmgfh878-art | [GitHub](https://github.com/vmgfh878-art/stock-trading-ai)
