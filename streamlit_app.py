import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from finvizfinance.screener.technical import Technical

import streamlit as st

# ---------------- إعدادات عامة ----------------

LOOKBACK_DAYS = 90

# ---------------- دوال مساعدة ----------------

@st.cache_data(show_spinner=False)
def download_history(ticker, days=LOOKBACK_DAYS):
    data = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
    if data is None or data.empty or len(data) < 40:
        return None
    return data

def vcp_long_score(df: pd.DataFrame) -> float:
    closes = df["Close"]
    if len(closes) < 60:
        return 0.0

    # ترند صاعد 40 يوم تقريبًا
    if closes.iloc[-1] <= closes.iloc[-40]:
        return 0.0

    recent = df.iloc[-60:]
    ranges = (recent["High"] - recent["Low"]) / recent["Close"]
    chunks = np.array_split(ranges.values, 4)
    avg_ranges = [float(np.mean(c)) for c in chunks]

    # تقلص التذبذب
    if not all(avg_ranges[i] < avg_ranges[i-1] for i in range(1, len(avg_ranges))):
        return 0.0

    # تقلص الفوليوم
    vol_chunks = np.array_split(recent["Volume"].values, 4)
    avg_vol = [float(np.mean(c)) for c in vol_chunks]
    if not (avg_vol[-1] <= avg_vol[0] * 0.9):
        return 0.0

    contraction_strength = (avg_ranges[0] - avg_ranges[-1]) / avg_ranges[0]
    trend_strength = (closes.iloc[-1] / closes.iloc[-40]) - 1
    score = max(0.0, contraction_strength * 0.7 + trend_strength * 0.3)
    return float(score)

def vcp_short_score(df: pd.DataFrame) -> float:
    closes = df["Close"]
    if len(closes) < 60:
        return 0.0

    # ترند هابط
    if closes.iloc[-1] >= closes.iloc[-40]:
        return 0.0

    recent = df.iloc[-60:]
    ranges = (recent["High"] - recent["Low"]) / recent["Close"]
    chunks = np.array_split(ranges.values, 4)
    avg_ranges = [float(np.mean(c)) for c in chunks]

    if not all(avg_ranges[i] < avg_ranges[i-1] for i in range(1, len(avg_ranges))):
        return 0.0

    vol_chunks = np.array_split(recent["Volume"].values, 4)
    avg_vol = [float(np.mean(c)) for c in vol_chunks]
    if avg_vol[-1] > avg_vol[0] * 1.2:
        return 0.0

    drop_strength = (closes.iloc[-40] / closes.iloc[-1]) - 1
    contraction_strength = (avg_ranges[0] - avg_ranges[-1]) / avg_ranges[0]
    score = max(0.0, contraction_strength * 0.5 + drop_strength * 0.5)
    return float(score)

@st.cache_data(show_spinner=False)
def get_finviz_candidates_long(index_filter: str, min_price: str, min_avg_vol: str):
    tech = Technical()
    filters = {
        "Index": index_filter,
        "Price": min_price,                 # مثال: "Over $20"
        "Average Volume": min_avg_vol,      # مثال: "Over 500K"
        "50-Day Simple Moving Average": "Price above SMA50",
        "200-Day Simple Moving Average": "Price above SMA200",
    }
    tech.set_filter(filters_dict=filters)
    df = tech.screener_view()
    if df is None or df.empty:
        return []
    return df["Ticker"].dropna().unique().tolist()

@st.cache_data(show_spinner=False)
def get_finviz_candidates_short(index_filter: str, min_price: str, min_avg_vol: str):
    tech = Technical()
    filters = {
        "Index": index_filter,
        "Price": min_price,
        "Average Volume": min_avg_vol,
        "50-Day Simple Moving Average": "Price below SMA50",
        "200-Day Simple Moving Average": "Price below SMA200",
    }
    tech.set_filter(filters_dict=filters)
    df = tech.screener_view()
    if df is None or df.empty:
        return []
    return df["Ticker"].dropna().unique().tolist()

def scan(max_long: int, max_short: int,
         index_filter: str, min_price: str, min_avg_vol: str):
    long_cands = get_finviz_candidates_long(index_filter, min_price, min_avg_vol)
    short_cands = get_finviz_candidates_short(index_filter, min_price, min_avg_vol)

    long_rows = []
    short_rows = []

    for t in long_cands:
        hist = download_history(t)
        if hist is None:
            continue
        s = vcp_long_score(hist)
        if s > 0:
            last = float(hist["Close"].iloc[-1])
            atr = float((hist["High"] - hist["Low"]).tail(14).mean())
            long_rows.append({"Ticker": t, "Price": round(last, 2),
                              "Score": round(s, 3), "ATR14": round(atr, 2)})

    for t in short_cands:
        hist = download_history(t)
        if hist is None:
            continue
        s = vcp_short_score(hist)
        if s > 0:
            last = float(hist["Close"].iloc[-1])
            atr = float((hist["High"] - hist["Low"]).tail(14).mean())
            short_rows.append({"Ticker": t, "Price": round(last, 2),
                               "Score": round(s, 3), "ATR14": round(atr, 2)})

    long_df = pd.DataFrame(long_rows).sort_values("Score", ascending=False).head(max_long)
    short_df = pd.DataFrame(short_rows).sort_values("Score", ascending=False).head(max_short)

    return long_df, short_df

# ---------------- واجهة Streamlit ----------------

st.set_page_config(page_title="VCP Scanner", layout="wide")

st.title("VCP Scanner - Minervini Style")
st.caption("أداة تجريبية لاختيار مرشحي شراء وشورت وفق VCP + Stage2/4 (لا تعتبر توصية استثمارية).")

col1, col2, col3, col4 = st.columns(4)

with col1:
    max_long = st.number_input("عدد مرشحي الشراء", min_value=1, max_value=30, value=5, step=1)
with col2:
    max_short = st.number_input("عدد مرشحي الشورت", min_value=1, max_value=30, value=5, step=1)
with col3:
    index_filter = st.selectbox(
        "المؤشر",
        ["S&P 500", "S&P 400", "S&P 600", "NASDAQ 100", "Any"],
        index=0
    )
with col4:
    min_price = st.selectbox(
        "السعر",
        ["Over $10", "Over $20", "Over $30", "Over $50"],
        index=1
    )

min_avg_vol = "Over 500K"

if st.button("تشغيل الفلتر الآن", type="primary"):
    with st.spinner("جاري الفحص..."):
        long_df, short_df = scan(max_long, max_short, index_filter, min_price, min_avg_vol)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("مرشحي الشراء (Stage 2 + VCP تقريبي)")
        if long_df.empty:
            st.write("لا يوجد مرشحين متوافقين مع القواعس الحالية.")
        else:
            st.dataframe(long_df, use_container_width=True)

    with c2:
        st.subheader("مرشحي الشورت (Stage 4 + VCP معكوس)")
        if short_df.empty:
            st.write("لا يوجد مرشحين متوافقين مع القواعس الحالية.")
        else:
            st.dataframe(short_df, use_container_width=True)

st.markdown(
    """
    **تنبيه مهم:** هذه الأداة تعليمية فقط. 
    قبل أي تنفيذ، راجع الشارت يدويًا، تأكد من الاختراق/الكسر والفوليوم، وحدد وقف خسارة واضح.
    """
)
