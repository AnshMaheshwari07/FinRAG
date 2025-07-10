from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from query_processor import query_agent
from pydantic import BaseModel  
import yfinance as yf
from comparison import create_candlestick_chart, fetch_data, rolling_average, ticker_info,return_30day
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent,AgentType

app=FastAPI()

llm=ChatGroq(
    model="Gemma2-9b-It"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
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

class createSummary(BaseModel):
    parsed:str
    ##see what i am thinking is we can pass max,min,name in this pydantic class and then pass them into prompt which llm can use to create summary

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

    if(len(request.tickers)>1):
        returns=return_30day(data,request.tickers)
        results["30_day_returns"]=returns

    else:
        candlestick_chart=create_candlestick_chart(data,request.tickers[0],request.start_date,request.end_date)
        results["candlestick_chart"]=candlestick_chart
    
    return results

@app.post("/summary")
def create_summary(request:createSummary):
    agent=initialize_agent(
        llm,
        tools=[],
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        return_intermediate_steps=True,
        max_iterations=3,
    )
    prompt=f"""
    You are a financial analyst.You have a great understanding of how stock works and how to analyze different charts.
    Given the following chart data, provide a summary of the stock performance and trends.
    {request.parsed}

    In the summary you must include:
    Include:
    - Price trend over time
    - Notable fluctuations
    - Start and end values
    - Observed volatility

    You must give a good crafted detailed summary through which user will able to know each and every nuances
    of the stock performance which is shown in the chart.
    """

    return agent.invoke({"input":prompt})["output"]