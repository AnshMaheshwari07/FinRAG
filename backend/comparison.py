import yfinance as yf
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def flatten_yf_df(df):
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns]
    return df

def fetch_data(df: pd.DataFrame):
    df = flatten_yf_df(df)
    close_cols = [col for col in df.columns if col.startswith("Close_")]
    df = df[["Date"] + close_cols]
    df_melt = df.melt(id_vars="Date", value_vars=close_cols, var_name="Ticker", value_name="Close")
    df_melt["Ticker"] = df_melt["Ticker"].str.replace("Close_", "")
    fig = px.line(
        df_melt,
        title="Stock Prices Line Chart",
        x="Date",
        y="Close",
        color="Ticker",
        labels={"Date": "Date", "Close": "Closing Price"}
    )
    return fig.to_json()

def return_30day(data: pd.DataFrame, ticker: list[str]):
    data = flatten_yf_df(data)
    close_cols = [col for col in data.columns if col.startswith("Close_")]
    recent = data.iloc[-1][close_cols]
    past = data.iloc[max(0, len(data) - 31)][close_cols]
    returns = ((recent - past) / past) * 100
    returns = returns.sort_values(ascending=False).reset_index()
    returns.columns = ["Ticker", "30 Day Returns %"]
    returns["Ticker"] = returns["Ticker"].str.replace("Close_", "")
    fig = px.bar(
        returns,
        x="Ticker",
        y="30 Day Returns %",
        title=f"Last 30 Day Returns for {ticker}",
        labels={"Ticker": "Ticker", "30 Day Returns %": "Returns (%)"},
    )
    return fig.to_json()

def rolling_average(df: pd.DataFrame, start_date: str, end_date: str):
    df = flatten_yf_df(df)
    df["Date"] = pd.to_datetime(df["Date"])
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    window = (end - start).days
    if window < 1:
        raise ValueError("Window size must be at least 1 day.")
    close_cols = [col for col in df.columns if col.startswith("Close_")]
    df = df[["Date"] + close_cols]
    df_melt = df.melt(id_vars=["Date"], value_vars=close_cols, var_name="Ticker", value_name="Close")
    df_melt["Ticker"] = df_melt["Ticker"].str.replace("Close_", "")
    df_melt["Rolling_Avg"] = df_melt.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    fig = px.line(
        df_melt,
        title=f"Rolling average for {window} days",
        x="Date",
        y="Rolling_Avg",
        color="Ticker",
        labels={"Date": "Date", "Rolling_Avg": "Rolling Average"},
    )
    return fig.to_json()

def create_candlestick_chart(df: pd.DataFrame, ticker: str, start: str, end: str):
    df = flatten_yf_df(df)
    df = df[["Date", f"Open_{ticker}", f"High_{ticker}", f"Low_{ticker}", f"Close_{ticker}"]]
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df[f"Open_{ticker}"],
        high=df[f"High_{ticker}"],
        low=df[f"Low_{ticker}"],
        close=df[f"Close_{ticker}"],
    )])
    fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_rangeslider_visible=False,xaxis_title="Date",yaxis_title="Price(USD)")
    return fig.to_json()


def areaa_chart(data: pd.DataFrame):
    data = data.reset_index()
    
    # Try to detect proper datetime column
    datetime_col = "Date" if "Date" in data.columns else data.columns[0]  # fallback

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[datetime_col],     # Use datetime for x-axis
        y=data["Close"],          # Close price for y-axis
        fill="tozeroy",
        name="Close Price",
        mode="lines",             # Ensure it draws a smooth line
        line=dict(color="skyblue")
    ))

    fig.update_layout(
        title="Area Chart of Closing Price",
        xaxis_title="Date",
        yaxis_title="Close Price",
        template="plotly_white"
    )

    return fig.to_json()

def ticker_info(ticker: list[str]):
    info = {}
    for t in ticker:
        try:
            stock = yf.Ticker(t)
            info[t] = {
                "name": stock.info.get("longName", "N/A"),
                "sector": stock.info.get("sector", "N/A"),
                "industry": stock.info.get("industry", "N/A"),
                "country": stock.info.get("country", "N/A"),
                "market_cap": stock.info.get("marketCap", "N/A"),
                "website": stock.info.get("website", "N/A"),
                "info": stock.info.get("longBusinessSummary", "N/A"),
            }
        except Exception as e:
            info[t] = {"error": f"Failed to fetch data for {t}: {str(e)}"}
    return info
