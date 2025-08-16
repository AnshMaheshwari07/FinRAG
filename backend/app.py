from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from query_processor import query_agent
from pydantic import BaseModel  
import yfinance as yf
from comparison import create_candlestick_chart, fetch_data, rolling_average, ticker_info,return_30day,areaa_chart
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent,AgentType
from langchain.tools import Tool
from websearch import Search

import os
from dotenv import load_dotenv
load_dotenv()



if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("Missing GROQ_API_KEY. Set it in environment variables or .env file.")

app=FastAPI()

llm=ChatGroq(
    model="Gemma2-9b-It",
    api_key=os.getenv("GROQ_API_KEY")
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    input_query:str

class createChart(BaseModel):
    tickers:list[str]
    start_date:str
    end_date:str
    interval:str="1d" 
    chart:str="Line chart"

@app.post("/query")
def chat_query(request:ChatRequest):
    response=query_agent(request.input_query)
    return response

@app.post("/chart")
def create_allcharts(request:createChart):
    data=yf.download(
        tickers=request.tickers,
        start=request.start_date,
        end=request.end_date,
        interval=request.interval,
        progress=False  
    )
    data=data.reset_index()
    
    if(len(request.tickers)==0):
        return {"error":"Atleast one ticker is required."}
    results={}

    info=ticker_info(request.tickers)
    results["info"]=info

    line_chart=fetch_data(data)
    results["line_chart"]=line_chart

    rolling_avg=rolling_average(data,request.start_date,request.end_date)
    results["rolling_average"]=rolling_avg

    area_chart=areaa_chart(data)
    results["area_chart"]=area_chart

    if(len(request.tickers)>1):
        returns=return_30day(data,request.tickers)
        results["30_day_returns"]=returns

    else:
        candlestick_chart=create_candlestick_chart(data,request.tickers[0],request.start_date,request.end_date)
        results["candlestick_chart"]=candlestick_chart
    
    return results



def get_stock(ticker:str,start_date:str,end_date:str):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df.reset_index()

    if df.empty:
        return {"error": "No data found for given date range."}

    min_price = df["Low"].min()
    max_price = df["High"].max()
    start_price = df.iloc[0]["Open"]
    end_price = df.iloc[-1]["Close"]
    pct_change = ((end_price - start_price) / start_price) * 100
    volatility = df["Close"].std()

    return {
        "ticker": ticker,
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2),
        "percent_change": round(pct_change, 2),
        "volatility": round(volatility, 2)
    }

@app.post("/summary")
def create_summary(request:createChart):
    if not request.tickers:
        return {"summary": "No ticker provided."}

    summary_data = get_stock(request.tickers[0], request.start_date, request.end_date)

    if "error" in summary_data:
        return {"summary": summary_data["error"]}

    prompt=f"""
    You are a financial analyst. Hers's data for {summary_data["ticker"]} from {request.start_date} to {request.end_date}.
    
    - Start price: ${summary_data["start_price"]}
    - End price: ${summary_data["end_price"]}
    - Min price: ${summary_data["min_price"]}
    - Max price: ${summary_data["max_price"]}
    - Percent change: {summary_data["percent_change"]}%
    - Volatility: {summary_data["volatility"]}

    Based on these details , First tell about the chart {request.chart} and then in continuation craft a good detailed summary telling these information and trend if possible and also try to start each point in a new line.
    At end just stop after disclaimer do not write let me know if you have another questions.
    """

    try:
        result=llm.invoke(prompt)
        print(result.content)
        return {"summary":result.content}
    except Exception as e:
        return {"summary":f"failed:{str(e)}"}