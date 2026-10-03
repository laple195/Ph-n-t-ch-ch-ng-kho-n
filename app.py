"""
==================================================================
 PHAN TICH XU HUONG CO PHIEU MY / NHAT / FOREX - 1 FILE DUY NHAT
 Chay local:      streamlit run app.py
 Hoac deploy mien phi len Streamlit Community Cloud (xem README).
==================================================================
"""
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# PHAN 1: CHI BAO KY THUAT (indicators)
# ------------------------------------------------------------------
"""
Các hàm tính chỉ báo kỹ thuật (technical indicators).
Tất cả nhận vào một DataFrame có cột: Open, High, Low, Close, Volume
và trả về Series hoặc DataFrame chỉ báo tương ứng.
"""


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low = df["High"], df["Low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_val = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / atr_val.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx_val.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    hl2 = (df["High"] + df["Low"]) / 2
    atr_val = atr(df, period)
    upperband = hl2 + multiplier * atr_val
    lowerband = hl2 - multiplier * atr_val
    close = df["Close"]

    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    trend = pd.Series(1, index=df.index)  # 1 = up, -1 = down

    for i in range(1, len(df)):
        if close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = min(upperband.iloc[i], final_upper.iloc[i - 1]) \
                if close.iloc[i] > final_upper.iloc[i - 1] else upperband.iloc[i]
        else:
            final_upper.iloc[i] = min(upperband.iloc[i], final_upper.iloc[i - 1])

        if close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = max(lowerband.iloc[i], final_lower.iloc[i - 1]) \
                if close.iloc[i] < final_lower.iloc[i - 1] else lowerband.iloc[i]
        else:
            final_lower.iloc[i] = max(lowerband.iloc[i], final_lower.iloc[i - 1])

        if close.iloc[i] > final_upper.iloc[i - 1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    st_line = pd.Series(np.where(trend == 1, final_lower, final_upper), index=df.index)
    return st_line, trend


# ------------------------------------------------------------------
# PHAN 2: CAC PHUONG PHAP GIAO DICH (strategies)
# ------------------------------------------------------------------
"""
Mỗi chiến lược nhận vào DataFrame giá (Open, High, Low, Close, Volume)
và trả về một pd.Series vị thế (position):
   1  = giữ lệnh mua (long)
  -1  = giữ lệnh bán (short)
   0  = đứng ngoài (flat)
Vị thế tại ngày i được coi là quyết định dựa trên dữ liệu đã đóng cửa
tới ngày i (không nhìn tương lai) và được áp dụng cho lợi nhuận ngày i+1.
"""


def strat_ma_crossover(df, fast=20, slow=50):
    f = ema(df["Close"], fast)
    s = ema(df["Close"], slow)
    pos = pd.Series(np.where(f > s, 1, -1), index=df.index)
    pos[f.isna() | s.isna()] = 0
    return pos


def strat_macd(df):
    macd_line, signal_line, _ = macd(df["Close"])
    pos = pd.Series(np.where(macd_line > signal_line, 1, -1), index=df.index)
    pos[macd_line.isna() | signal_line.isna()] = 0
    return pos


def strat_rsi_reversion(df, period=14, lower=30, upper=70):
    r = rsi(df["Close"], period)
    pos = pd.Series(0, index=df.index)
    state = 0
    r_vals = r.values
    for i in range(len(r_vals)):
        if np.isnan(r_vals[i]):
            pos.iloc[i] = 0
            continue
        if r_vals[i] < lower:
            state = 1
        elif r_vals[i] > upper:
            state = -1
        elif lower <= r_vals[i] <= upper and (state == 1 and r_vals[i] > 50):
            state = 0
        elif lower <= r_vals[i] <= upper and (state == -1 and r_vals[i] < 50):
            state = 0
        pos.iloc[i] = state
    return pos


def strat_bollinger_breakout(df, period=20, num_std=2.0):
    upper, mid, lower = bollinger_bands(df["Close"], period, num_std)
    close = df["Close"]
    pos = pd.Series(0, index=df.index)
    state = 0
    for i in range(len(df)):
        if np.isnan(upper.iloc[i]):
            continue
        if close.iloc[i] > upper.iloc[i]:
            state = 1
        elif close.iloc[i] < lower.iloc[i]:
            state = -1
        pos.iloc[i] = state
    return pos


def strat_adx_trend_filter(df, adx_period=14, ma_period=100, adx_threshold=25):
    adx_val, _, _ = adx(df, adx_period)
    ma = sma(df["Close"], ma_period)
    close = df["Close"]
    trend_dir = np.where(close > ma, 1, -1)
    pos = pd.Series(np.where(adx_val > adx_threshold, trend_dir, 0), index=df.index)
    pos[ma.isna()] = 0
    return pos


def strat_supertrend(df, period=10, multiplier=3.0):
    _, trend = supertrend(df, period, multiplier)
    return trend.copy()


STRATEGIES = {
    "MA Crossover (EMA20/50)": strat_ma_crossover,
    "MACD": strat_macd,
    "RSI Mean-Reversion": strat_rsi_reversion,
    "Bollinger Breakout": strat_bollinger_breakout,
    "ADX + MA Trend Filter": strat_adx_trend_filter,
    "Supertrend": strat_supertrend,
}


# ------------------------------------------------------------------
# PHAN 3: BACKTEST ENGINE
# ------------------------------------------------------------------
"""
Chạy backtest cho một chuỗi vị thế (position series) trên dữ liệu giá,
tính các chỉ số hiệu suất: win rate, profit factor, expectancy, tổng lợi
nhuận, max drawdown. Không dùng win rate làm tiêu chí chọn chiến lược
chính — dùng profit factor (kết hợp cả tỷ lệ thắng lẫn độ lớn thắng/thua).
"""


def _segment_trades(position: pd.Series, strat_returns: pd.Series):
    """Gộp các ngày có cùng vị thế (khác 0) liên tiếp thành từng 'trade',
    trả về danh sách lợi nhuận (%) của mỗi trade."""
    trades = []
    current_sign = 0
    current_ret = 1.0
    for pos_val, r in zip(position.values, strat_returns.values):
        sign = np.sign(pos_val)
        if sign == 0:
            if current_sign != 0:
                trades.append(current_ret - 1.0)
                current_sign = 0
                current_ret = 1.0
            continue
        if sign != current_sign:
            if current_sign != 0:
                trades.append(current_ret - 1.0)
            current_sign = sign
            current_ret = 1.0
        if not np.isnan(r):
            current_ret *= (1 + r)
    if current_sign != 0:
        trades.append(current_ret - 1.0)
    return trades


def run_backtest(df: pd.DataFrame, position: pd.Series, min_trades: int = 5):
    close = df["Close"]
    daily_ret = close.pct_change()
    strat_ret = position.shift(1).fillna(0) * daily_ret
    strat_ret = strat_ret.fillna(0)

    equity = (1 + strat_ret).cumprod()
    total_return = equity.iloc[-1] - 1 if len(equity) else 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min() if len(drawdown) else 0.0

    trades = _segment_trades(position, strat_ret)
    n_trades = len(trades)

    if n_trades == 0:
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "n_trades": 0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "expectancy": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "valid": False,
            "equity_curve": equity,
        }

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate = len(wins) / n_trades
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf
    expectancy = np.mean(trades)
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    # annualized Sharpe xấp xỉ (giả định 252 phiên/năm, lãi suất phi rủi ro = 0)
    if strat_ret.std() > 0:
        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe": sharpe,
        "valid": n_trades >= min_trades,
        "equity_curve": equity,
    }


def rank_strategies(results: dict, min_trades: int = 5):
    """Sắp xếp các chiến lược theo profit factor giảm dần, chỉ xét những
    chiến lược có đủ số lượng trade tối thiểu để tránh nhiễu thống kê."""
    scored = []
    for name, res in results.items():
        if res["n_trades"] >= min_trades and np.isfinite(res.get("profit_factor", np.nan)):
            scored.append((name, res))
    scored.sort(key=lambda kv: kv[1]["profit_factor"], reverse=True)
    return scored


# ------------------------------------------------------------------
# PHAN 4: LAY DU LIEU (yfinance)
# ------------------------------------------------------------------
"""
Lấy dữ liệu giá lịch sử bằng yfinance (miễn phí, dữ liệu có độ trễ
~15-20 phút với dữ liệu intraday). Hỗ trợ 3 loại thị trường:
  US  -> cổ phiếu Mỹ, ví dụ: AAPL, TSLA
  JP  -> cổ phiếu Nhật, tự thêm hậu tố .T, ví dụ: 7203 -> 7203.T
  FX  -> forex, tự thêm hậu tố =X, ví dụ: USDJPY -> USDJPY=X
"""

MARKET_LABELS = {
    "US": "Cổ phiếu Mỹ",
    "JP": "Cổ phiếu Nhật (TSE)",
    "FX": "Forex",
}

COMMON_FX_PAIRS = ["USDJPY", "EURJPY", "GBPJPY", "EURUSD", "GBPUSD", "AUDJPY"]


def normalize_ticker(market: str, raw: str) -> str:
    raw = raw.strip().upper().replace(" ", "")
    if market == "JP":
        return raw if raw.endswith(".T") else f"{raw}.T"
    if market == "FX":
        return raw if raw.endswith("=X") else f"{raw}=X"
    return raw  # US: giữ nguyên


def fetch_history(market: str, raw_ticker: str, period: str = "2y", interval: str = "1d"):
    """Trả về (ticker_chuan_hoa, DataFrame OHLCV). Ném lỗi ValueError nếu
    không tìm thấy dữ liệu (mã sai hoặc hết hạn mức API)."""
    ticker = normalize_ticker(market, raw_ticker)
    data = yf.download(
        ticker, period=period, interval=interval,
        progress=False, auto_adjust=True, multi_level_index=False,
    )
    if data is None or data.empty:
        raise ValueError(
            f"Không lấy được dữ liệu cho mã '{ticker}'. "
            f"Kiểm tra lại mã, hoặc thử lại sau (Yahoo Finance có thể giới hạn tần suất)."
        )
    data = data.dropna(subset=["Close"])
    return ticker, data


def fetch_latest_quote(ticker: str):
    """Lấy giá gần nhất + % thay đổi trong ngày (dữ liệu có độ trễ)."""
    t = yf.Ticker(ticker)
    fast = t.fast_info
    try:
        price = fast["last_price"]
        prev_close = fast["previous_close"]
        change_pct = (price - prev_close) / prev_close * 100 if prev_close else None
        return price, change_pct
    except Exception:
        return None, None


# ------------------------------------------------------------------
# PHAN 5: GIAO DIEN STREAMLIT
# ------------------------------------------------------------------
"""
Ứng dụng phân tích xu hướng cổ phiếu Mỹ / Nhật / Forex.
Chạy: streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Phân tích xu hướng", layout="wide")

st.title("📈 Phân tích xu hướng tự động")
st.caption(
    "Nhập mã cổ phiếu/forex — hệ thống sẽ tự backtest nhiều phương pháp phân tích "
    "kỹ thuật trên lịch sử giá của chính mã đó và chọn ra phương pháp có hiệu quả "
    "tốt nhất (theo profit factor) để đưa ra tín hiệu xu hướng hiện tại."
)
st.info(
    "⚠️ Dữ liệu lấy miễn phí từ Yahoo Finance, có độ trễ ~15-20 phút với dữ liệu "
    "trong ngày, không phải giá tick-by-tick tuyệt đối. Đây là công cụ hỗ trợ học "
    "tập/tham khảo, không phải khuyến nghị đầu tư.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Thiết lập")
    market = st.radio(
        "Thị trường",
        options=["US", "JP", "FX"],
        format_func=lambda m: MARKET_LABELS[m],
    )

    if market == "FX":
        preset = st.selectbox("Cặp forex phổ biến", ["(Tự nhập)"] + COMMON_FX_PAIRS)
        if preset == "(Tự nhập)":
            raw_ticker = st.text_input("Nhập cặp forex (vd: USDJPY)", value="USDJPY")
        else:
            raw_ticker = preset
    elif market == "JP":
        raw_ticker = st.text_input("Mã cổ phiếu Nhật (vd: 7203 cho Toyota)", value="7203")
    else:
        raw_ticker = st.text_input("Mã cổ phiếu Mỹ (vd: AAPL)", value="AAPL")

    period = st.selectbox("Khoảng dữ liệu backtest", ["6mo", "1y", "2y", "5y"], index=2)
    min_trades = st.slider("Số lệnh tối thiểu để xét 1 phương pháp hợp lệ", 3, 30, 8)
    run_btn = st.button("🔍 Phân tích", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None

if run_btn:
    with st.spinner("Đang tải dữ liệu và backtest các phương pháp..."):
        try:
            ticker, df = fetch_history(market, raw_ticker, period=period)
            results = {}
            positions = {}
            for name, fn in STRATEGIES.items():
                pos = fn(df)
                res = run_backtest(df, pos, min_trades=min_trades)
                results[name] = res
                positions[name] = pos

            ranked = rank_strategies(results, min_trades=min_trades)
            if not ranked:
                # không có chiến lược nào đủ số lệnh tối thiểu -> nới lỏng, chọn PF cao nhất
                ranked = rank_strategies(results, min_trades=1)

            st.session_state.result = {
                "ticker": ticker,
                "df": df,
                "results": results,
                "positions": positions,
                "ranked": ranked,
            }
        except Exception as e:
            st.session_state.result = None
            st.error(str(e))

res_state = st.session_state.result
if res_state:
    ticker = res_state["ticker"]
    df = res_state["df"]
    results = res_state["results"]
    positions = res_state["positions"]
    ranked = res_state["ranked"]

    if not ranked:
        st.warning("Không có phương pháp nào tạo đủ tín hiệu để đánh giá. Thử khoảng dữ liệu dài hơn.")
    else:
        best_name, best_res = ranked[0]
        best_pos = positions[best_name]
        last_pos = best_pos.iloc[-1]
        trend_label = "🟢 TĂNG (Long)" if last_pos > 0 else ("🔴 GIẢM (Short)" if last_pos < 0 else "⚪ ĐI NGANG")

        price, chg = fetch_latest_quote(ticker)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            f"Giá gần nhất ({ticker})",
            f"{price:,.2f}" if price else f"{df['Close'].iloc[-1]:,.2f}",
            f"{chg:+.2f}%" if chg is not None else None,
        )
        c2.metric("Xu hướng hiện tại", trend_label)
        c3.metric("Phương pháp tốt nhất", best_name)
        c4.metric("Profit Factor (lịch sử)", f"{best_res['profit_factor']:.2f}")

        st.markdown(
            f"**Diễn giải:** Trên dữ liệu {period} gần nhất, phương pháp **{best_name}** cho kết quả "
            f"tốt nhất với {best_res['n_trades']} lệnh, tỷ lệ thắng {best_res['win_rate']*100:.1f}%, "
            f"profit factor {best_res['profit_factor']:.2f}, tổng lợi nhuận backtest "
            f"{best_res['total_return']*100:+.1f}%, drawdown tối đa {best_res['max_drawdown']*100:.1f}%. "
            f"Tín hiệu hiện tại: **{trend_label}**."
        )

        # ---- Bảng so sánh tất cả phương pháp ----
        st.subheader("So sánh các phương pháp đã backtest")
        rows = []
        for name, r in results.items():
            rows.append({
                "Phương pháp": name,
                "Số lệnh": r["n_trades"],
                "Tỷ lệ thắng": f"{r['win_rate']*100:.1f}%" if r["n_trades"] else "-",
                "Profit Factor": f"{r['profit_factor']:.2f}" if np.isfinite(r.get("profit_factor", np.nan)) else "-",
                "Expectancy/lệnh": f"{r['expectancy']*100:.2f}%" if r["n_trades"] else "-",
                "Tổng lợi nhuận": f"{r['total_return']*100:+.1f}%",
                "Max Drawdown": f"{r['max_drawdown']*100:.1f}%",
                "Đủ điều kiện (≥ số lệnh tối thiểu)": "✅" if r["n_trades"] >= min_trades else "❌",
            })
        comp_df = pd.DataFrame(rows).sort_values(
            "Profit Factor", key=lambda c: pd.to_numeric(c, errors="coerce"), ascending=False
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # ---- Biểu đồ giá + tín hiệu ----
        st.subheader(f"Biểu đồ giá & tín hiệu — {best_name}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Giá", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ))

        # overlay chỉ báo tuỳ theo phương pháp tốt nhất
        if best_name.startswith("MA Crossover"):
            fig.add_trace(go.Scatter(x=df.index, y=ema(df["Close"], 20), name="EMA20", line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=ema(df["Close"], 50), name="EMA50", line=dict(width=1)))
        elif best_name == "Bollinger Breakout":
            up, mid, low = bollinger_bands(df["Close"])
            fig.add_trace(go.Scatter(x=df.index, y=up, name="BB Upper", line=dict(width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=df.index, y=low, name="BB Lower", line=dict(width=1, dash="dot")))
        elif best_name == "Supertrend":
            st_line, _ = supertrend(df)
            fig.add_trace(go.Scatter(x=df.index, y=st_line, name="Supertrend", line=dict(width=1.5)))
        elif best_name.startswith("ADX"):
            fig.add_trace(go.Scatter(x=df.index, y=sma(df["Close"], 100), name="SMA100", line=dict(width=1)))

        # đánh dấu điểm vào lệnh (khi vị thế đổi dấu)
        change_points = best_pos[best_pos.diff().fillna(best_pos) != 0]
        buys = change_points[change_points > 0]
        sells = change_points[change_points < 0]
        fig.add_trace(go.Scatter(
            x=buys.index, y=df.loc[buys.index, "Low"] * 0.98, mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=10), name="Vào Long",
        ))
        fig.add_trace(go.Scatter(
            x=sells.index, y=df.loc[sells.index, "High"] * 1.02, mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=10), name="Vào Short",
        ))

        fig.update_layout(height=550, xaxis_rangeslider_visible=False, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Đường cong lợi nhuận (equity curve) của phương pháp tốt nhất"):
            eq = best_res["equity_curve"]
            fig2 = go.Figure(go.Scatter(x=eq.index, y=eq.values, name="Equity"))
            fig2.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.write("👈 Nhập mã và bấm **Phân tích** để bắt đầu.")

