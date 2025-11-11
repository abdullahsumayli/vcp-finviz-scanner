import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from finvizfinance.screener.technical import Technical
import streamlit as st

# ================= إعداد عام =================

LOOKBACK_DAYS = 90

st.set_page_config(
    page_title="VCP Scanner - Minervini Style",
    layout="wide"
)

# ================ أدوات مساعدة ================

def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    return s.dropna()

@st.cache_data(show_spinner=False)
def download_history(ticker: str, days: int = LOOKBACK_DAYS):
    data = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
    if data is None or data.empty:
        return None
    data = data.dropna()
    if len(data) < 60:
        return None
    return data

def vcp_long_score(df: pd.DataFrame) -> float:
    closes = get_series(df, "Close")
    highs = get_series(df, "High")
    lows = get_series(df, "Low")
    vols = get_series(df, "Volume")

    n = min(len(closes), len(highs), len(lows), len(vols))
    if n < 60:
        return 0.0

    closes = closes.iloc[-n:]
    highs = highs.iloc[-n:]
    lows = lows.iloc[-n:]
    vols = vols.iloc[-n:]

    # ترند صاعد
    if float(closes.iloc[-1]) <= float(closes.iloc[-40]):
        return 0.0

    recent_h = highs.iloc[-60:]
    recent_l = lows.iloc[-60:]
    recent_c = closes.iloc[-60:]
    recent_v = vols.iloc[-60:]

    ranges = (recent_h - recent_l) / recent_c
    chunks = np.array_split(ranges.values, 4)
    avg_ranges = [float(np.mean(c)) for c in chunks]

    # تقلص تذبذب
    if not all(avg_ranges[i] < avg_ranges[i - 1] for i in range(1, len(avg_ranges))):
        return 0.0

    # تقلص فوليوم
    vol_chunks = np.array_split(recent_v.values, 4)
    avg_vol = [float(np.mean(c)) for c in vol_chunks]
    if not (avg_vol[-1] <= avg_vol[0] * 0.9):
        return 0.0

    contraction_strength = (avg_ranges[0] - avg_ranges[-1]) / avg_ranges[0]
    trend_strength = (float(closes.iloc[-1]) / float(closes.iloc[-40])) - 1.0
    score = max(0.0, contraction_strength * 0.7 + trend_strength * 0.3)
    return float(score)

def vcp_short_score(df: pd.DataFrame) -> float:
    closes = get_series(df, "Close")
    highs = get_series(df, "High")
    lows = get_series(df, "Low")
    vols = get_series(df, "Volume")

    n = min(len(closes), len(highs), len(lows), len(vols))
    if n < 60:
        return 0.0

    closes = closes.iloc[-n:]
    highs = highs.iloc[-n:]
    lows = lows.iloc[-n:]
    vols = vols.iloc[-n:]

    # ترند هابط
    if float(closes.iloc[-1]) >= float(closes.iloc[-40]):
        return 0.0

    recent_h = highs.iloc[-60:]
    recent_l = lows.iloc[-60:]
    recent_c = closes.iloc[-60:]
    recent_v = vols.iloc[-60:]

    ranges = (recent_h - recent_l) / recent_c
    chunks = np.array_split(ranges.values, 4)
    avg_ranges = [float(np.mean(c)) for c in chunks]

    if not all(avg_ranges[i] < avg_ranges[i - 1] for i in range(1, len(avg_ranges))):
        return 0.0

    vol_chunks = np.array_split(recent_v.values, 4)
    avg_vol = [float(np.mean(c)) for c in vol_chunks]
    if avg_vol[-1] > avg_vol[0] * 1.2:
        return 0.0

    drop_strength = (float(closes.iloc[-40]) / float(closes.iloc[-1])) - 1.0
    contraction_strength = (avg_ranges[0] - avg_ranges[-1]) / avg_ranges[0]
    score = max(0.0, contraction_strength * 0.5 + drop_strength * 0.5)
    return float(score)

@st.cache_data(show_spinner=False)
def get_finviz_candidates_long(index_filter: str, min_price: str, min_avg_vol: str):
    tech = Technical()
    filters = {
        "Index": index_filter,
        "Price": min_price,
        "Average Volume": min_avg_vol,
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

def build_rows(ticker_list, mode: str):
    rows = []
    for t in ticker_list:
        hist = download_history(t)
        if hist is None:
            continue

        if mode == "long":
            s = vcp_long_score(hist)
        else:
            s = vcp_short_score(hist)

        if s <= 0:
            continue

        closes = get_series(hist, "Close")
        if len(closes) == 0:
            continue

        last = float(closes.iloc[-1])

        # أداء آخر 20 يوم كنسبة
        if len(closes) >= 20:
            chg20 = (last / float(closes.iloc[-20]) - 1.0) * 100.0
        else:
            chg20 = np.nan

        atr = float((get_series(hist, "High").tail(14) -
                     get_series(hist, "Low").tail(14)).mean())

        # sparkline من آخر 20 إغلاق
        spark_data = closes.tail(20).reset_index(drop=True)

        rows.append({
            "Ticker": t,
            "Price": round(last, 2),
            "Score": round(s, 3),
            "Chg20%": round(chg20, 2) if not np.isnan(chg20) else None,
            "ATR14": round(atr, 2),
            "Spark": spark_data,
        })
    return rows

def scan(max_long: int, max_short: int,
         index_filter: str, min_price: str, min_avg_vol: str):
    max_long = max(1, int(max_long))
    max_short = max(1, int(max_short))

    long_cands = get_finviz_candidates_long(index_filter, min_price, min_avg_vol)
    short_cands = get_finviz_candidates_short(index_filter, min_price, min_avg_vol)

    long_rows = build_rows(long_cands, "long")
    short_rows = build_rows(short_cands, "short")

    long_df = pd.DataFrame(long_rows).sort_values("Score", ascending=False).head(max_long)
    short_df = pd.DataFrame(short_rows).sort_values("Score", ascending=False).head(max_short)

    return long_df, short_df

# ================ تنسيق بصري ================

st.markdown(
    """
    <style>
    body {background-color: #0f1117;}
    .block-container {padding-top: 1.5rem; padding-bottom: 1.5rem;}
    .metric-label {font-size: 0.8rem;}
    .metric-value {font-size: 1.2rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================ واجهة المستخدم ================

st.title("VCP Scanner · Minervini Style")
st.caption("مرشحات تعليمية مبنية على Stage 2 / Stage 4 + VCP تقريبي. ليست توصية استثمارية.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    max_long = st.number_input("عدد مرشحي الشراء", min_value=1, max_value=30, value=5, step=1)
with col2:
    max_short = st.number_input("عدد مرشحي الشورت", min_value=1, max_value=30, value=5, step=1)
with col3:
    index_filter = st.selectbox(
        "المؤشر",
        ["S&P 500", "S&P 400", "S&P 600", "NASDAQ 100", "Any"],
        index=0,
    )
with col4:
    min_price = st.selectbox(
        "فلتر السعر",
        ["Over $10", "Over $20", "Over $30", "Over $50"],
        index=1,
    )

min_avg_vol = "Over 500K"

run = st.button("تشغيل الفلتر الآن", type="primary")

if run:
    with st.spinner("Scanning with Finviz + YF..."):
        long_df, short_df = scan(max_long, max_short, index_filter, min_price, min_avg_vol)

    # كروت ملخص
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("عدد مرشحي الشراء", len(long_df))
    with m2:
        st.metric("أعلى Score شراء", f"{long_df['Score'].max():.3f}" if not long_df.empty else "-")
    with m3:
        st.metric("عدد مرشحي الشورت", len(short_df))
    with m4:
        st.metric("أعلى Score شورت", f"{short_df['Score'].max():.3f}" if not short_df.empty else "-")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("مرشحي الشراء (Stage 2 + VCP)")
        if long_df.empty:
            st.info("لا يوجد مرشحين ضمن الشروط الحالية.")
        else:
            # عرض جدول بدون عمود spark الخام
            st.dataframe(
                long_df[["Ticker", "Price", "Score", "Chg20%", "ATR14"]],
                use_container_width=True,
                height=260,
            )
            # سبـاركلين لأعلى 3
            top3 = long_df.head(3)
            st.caption("Trend snapshot لأعلى 3 Scores:")
            for _, row in top3.iterrows():
                st.text(f"{row['Ticker']}  | Score {row['Score']}")
                st.line_chart(row["Spark"], height=80)

    with c2:
        st.subheader("مرشحي الشورت (Stage 4 + VCP معكوس)")
        if short_df.empty:
            st.info("لا يوجد مرشحين ضمن الشروط الحالية.")
        else:
            st.dataframe(
                short_df[["Ticker", "Price", "Score", "Chg20%", "ATR14"]],
                use_container_width=True,
                height=260,
            )
            st.caption("Trend snapshot لأعلى 3 Scores (شورت):")
            top3s = short_df.head(3)
            for _, row in top3s.iterrows():
                st.text(f"{row['Ticker']}  | Score {row['Score']}")
                st.line_chart(row["Spark"], height=80)

    st.markdown(
        """
        **ملاحظات استخدام:**
        - استخدم هذه القائمة كنقطة بداية.
        - افتح الشارت اليومي والساعة لكل سهم.
        - تأكد من اختراق/كسر واضح + حجم مرتفع + وقف خسارة منطقي قبل أي قرار.
        """
    )
