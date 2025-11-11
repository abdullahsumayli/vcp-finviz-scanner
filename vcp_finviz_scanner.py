import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from finvizfinance.screener.technical import Technical  # :contentReference[oaicite:0]{index=0}

# ---------- إعدادات عامة ----------
MAX_LONG = 5       # عدد مرشحي الشراء
MAX_SHORT = 5      # عدد مرشحي الشورت
LOOKBACK_DAYS = 90 # فترة تحليل الـ VCP التقريبية

# ---------- دوال مساعدة ----------

def download_history(ticker, days=LOOKBACK_DAYS):
    data = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
    if data is None or data.empty or len(data) < 40:
        return None
    return data

def vcp_long_score(df: pd.DataFrame) -> float:
    """
    VCP تقريبي للشراء:
    - ترند صاعد في آخر 8 أسابيع.
    - تقلص تذبذب السعر.
    - تقلص نسبي في الفوليوم.
    يرجع درجة (كلما أعلى أفضل). 0 يعني استبعاد.
    """
    closes = df["Close"]
    highs = df["High"]
    lows = df["Low"]
    vols = df["Volume"].astype(float)

    if len(closes) < 60:
        return 0.0

    # ترند صاعد بسيط
    close_now = float(closes.iloc[-1])
    close_40 = float(closes.iloc[-40])
    
    if close_now <= close_40:
        return 0.0

    # تقسيم آخر 60 شمعة إلى 4 مقاطع لقياس تقلص التذبذب
    recent = df.iloc[-60:]
    ranges = (recent["High"] - recent["Low"]) / recent["Close"]
    chunks = np.array_split(ranges.values, 4)
    avg_ranges = [np.mean(c) for c in chunks]

    # يجب أن تكون كل مرحلة تذبذبها أقل من السابقة (تقلص)
    if not all(avg_ranges[i] < avg_ranges[i-1] for i in range(1, len(avg_ranges))):
        return 0.0

    # تقلص نسبي في الفوليوم
    vol_chunks = np.array_split(recent["Volume"].values, 4)
    avg_vol = [np.mean(c) for c in vol_chunks]
    if not (avg_vol[-1] <= avg_vol[0] * 0.9):
        return 0.0

    # درجة بسيطة تعكس قوة الانكماش والترند
    contraction_strength = (avg_ranges[0] - avg_ranges[-1]) / avg_ranges[0]
    trend_strength = (close_now / close_40) - 1
    score = max(0.0, contraction_strength * 0.7 + trend_strength * 0.3)
    return float(score)

def vcp_short_score(df: pd.DataFrame) -> float:
    """
    VCP معكوس للشورت:
    - ترند هابط في آخر 8 أسابيع.
    - تقلص ارتدادات ثم كسر محتمل.
    """
    closes = df["Close"]
    highs = df["High"]
    lows = df["Low"]
    vols = df["Volume"].astype(float)

    if len(closes) < 60:
        return 0.0

    # ترند هابط
    close_now = float(closes.iloc[-1])
    close_40 = float(closes.iloc[-40])
    
    if close_now >= close_40:
        return 0.0

    recent = df.iloc[-60:]
    # نقيس تذبذب الارتدادات لأعلى
    ranges = (recent["High"] - recent["Low"]) / recent["Close"]
    chunks = np.array_split(ranges.values, 4)
    avg_ranges = [np.mean(c) for c in chunks]

    # تقلص التذبذب ثم عودة ضغط بيعي
    if not all(avg_ranges[i] < avg_ranges[i-1] for i in range(1, len(avg_ranges))):
        return 0.0

    # فوليوم لا يزيد بقوة في الارتدادات الأخيرة (إشارة ضعف)
    vol_chunks = np.array_split(recent["Volume"].values, 4)
    avg_vol = [np.mean(c) for c in vol_chunks]
    # نسمح بثبات أو ارتفاع بسيط فقط
    if avg_vol[-1] > avg_vol[0] * 1.2:
        return 0.0

    drop_strength = (close_40 / close_now) - 1
    contraction_strength = (avg_ranges[0] - avg_ranges[-1]) / avg_ranges[0]
    score = max(0.0, contraction_strength * 0.5 + drop_strength * 0.5)
    return float(score)

# ---------- سحب مرشحين من Finviz ----------

def get_finviz_candidates_long():
    """
    Stage 2 تقريبياً:
    - أسهم أمريكية كبيرة/متوسطة
    - سيولة جيدة
    - السعر فوق متوسطات 50 و200
    """
    tech = Technical()
    filters = {
        "Index": "S&P 500",                 # يمكنك تغييرها لـ: S&P 500 | S&P 400 | S&P 600 | NASDAQ 100
        "Average Volume": "Over 500K",
        "Price": "Over $20",
        "50-Day Simple Moving Average": "Price above SMA50",
        "200-Day Simple Moving Average": "Price above SMA200"
    }
    tech.set_filter(filters_dict=filters)
    df = tech.screener_view()
    tickers = df.get("Ticker", pd.Series()).dropna().unique().tolist()
    return tickers

def get_finviz_candidates_short():
    """
    Stage 4 تقريبياً:
    - تحت متوسط 50 و200
    - سيولة كافية للشورت
    """
    tech = Technical()
    filters = {
        "Index": "S&P 500",
        "Average Volume": "Over 500K",
        "Price": "Over $20",
        "50-Day Simple Moving Average": "Price below SMA50",
        "200-Day Simple Moving Average": "Price below SMA200"
    }
    tech.set_filter(filters_dict=filters)
    df = tech.screener_view()
    tickers = df.get("Ticker", pd.Series()).dropna().unique().tolist()
    return tickers

# ---------- المنطق الرئيسي ----------

def scan():
    long_cands = get_finviz_candidates_long()
    short_cands = get_finviz_candidates_short()

    long_scores = []
    short_scores = []

    for t in long_cands:
        hist = download_history(t)
        if hist is None:
            continue
        s = vcp_long_score(hist)
        if s > 0:
            long_scores.append((t, s))

    for t in short_cands:
        hist = download_history(t)
        if hist is None:
            continue
        s = vcp_short_score(hist)
        if s > 0:
            short_scores.append((t, s))

    long_scores = sorted(long_scores, key=lambda x: x[1], reverse=True)[:MAX_LONG]
    short_scores = sorted(short_scores, key=lambda x: x[1], reverse=True)[:MAX_SHORT]

    print("\n=== مرشحي الشراء (Stage 2 + VCP تقريبي) ===")
    if not long_scores:
        print("لا يوجد مرشحين متطابقين اليوم وفق هذه القواعد.")
    else:
        for t, s in long_scores:
            print(f"{t}  | Score: {s:.3f}")

    print("\n=== مرشحي الشورت (Stage 4 + Breakdown تقريبي) ===")
    if not short_scores:
        print("لا يوجد مرشحين متطابقين اليوم وفق هذه القواعد.")
    else:
        for t, s in short_scores:
            print(f"{t}  | Score: {s:.3f}")

if __name__ == "__main__":
    scan()
