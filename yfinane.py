import os
import json
import yfinance as yf
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
load_dotenv()

class FinanceTool:
    def __init__(self):
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
        
        self.llm = ChatGroq(
            model="Gemma2-9b-It",
            temperature=0.1
        )

        self.response_schemas = [
            ResponseSchema(name="ticker", description="Ticker symbol for the stock extracted from query"),
            ResponseSchema(name="start_date", description="Start date for the stock data in YYYY-MM-DD format"),
            ResponseSchema(name="end_date", description="End date for the stock data in YYYY-MM-DD format"),
        ]

        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)

        self.prompt = PromptTemplate(
            template="""
            You are a financial assistant. Extract these fields from the user query:
            - ticker (e.g., AAPL)
            - start_date (YYYY-MM-DD)
            - end_date (YYYY-MM-DD)

            If start_date or end_date are missing, assume the last 30 days.
            
            {format_instructions}

            Query: {query}
            """,
            input_variables=["query"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},

        )

        self.summary_prompt=PromptTemplate(
            template="""
            
            You are a financial assistant. Based on the stock data provided.
            {data}

            Produced a detailed summary of stock data including:
            Opening price on first day 
            Closing Price on last day
            Highest price and date on which it occured
            Lowest price and date on which it occured
            Average price over the period
            Total volume traded over the period
            then you must provide a recommendation to buy sell from given {data} but give
            user a disclaimer that this is not a financial advice as I am a LLM and all details should 
            be in new lines.
            The summary should be in a narrative format, not a list.
            """,
            input_variables=["data"],
        )

        self.summary_chain= LLMChain(
            llm=self.llm,
            prompt=self.summary_prompt,
            verbose=False
        )
        self.extract_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            output_parser=self.output_parser,
            verbose=True
        )

    def get_stock_data(self, query: str) -> dict:
        """
        Extracts the ticker symbol and date range from the query,
        fetches stock data from Yahoo Finance, calculates moving averages,
        and returns recommendation.
        """
        params = self.extract_chain.run(query=query)
        print("Extracted parameters:", params)
        df = yf.Ticker(params["ticker"]).history(
            start=params["start_date"],
            end=params["end_date"]
        )
       
        records = df.reset_index().to_dict(orient="records")
        if df.empty:
            return json.dumps({"error": "No data found for the given ticker and date range."})

        df=df.reset_index()
        df["Date"]=df["Date"].astype(str)
        
        opening_price=df.iloc[0]["Open"]
        closing_price=df.iloc[-1]["Close"]
        max_price=df["High"].max()
        max_price_date=df.loc[df["High"].idxmax()]["Date"]
        min_price=df["Low"].min()
        min_price_date=df.loc[df["Low"].idxmin()]["Date"]
        avg_price=df["Close"].mean()
        total_volume=df["Volume"].sum()

        summary_input = f"""
        Opening price on first day: {opening_price}
        Closing price on last day: {closing_price}
        Highest price: {max_price} on {max_price_date}
        Lowest price: {min_price} on {min_price_date}
        Average price: {avg_price}
        Total volume traded: {total_volume}
        All these details should be in new lines.
        Provide recommendation also like buy/sell/hold but give user a disclaimer that this is not a financial advice as I am a LLM and all details should be in new lines.
        """
        narrative=self.summary_chain.run(data=summary_input)
        result= {
            "summary":narrative
        }
        return result
