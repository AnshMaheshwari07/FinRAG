from langchain_tavily import TavilySearch
import os
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
tavily_instance=TavilySearch(
        max_results=3,
        topic="general",
        include_answer=True,
        include_raw_content=False,# skip full HTML
        include_images=False,     # skip image list
        time_range="day"
    )
class Search:
    @staticmethod
    def tavily(query:str)->str:
        try:
            result = tavily_instance.invoke({"query": query})
            print("tavily returns ",result)
            return result
        except Exception as e:
            return f"Error during Tavily search: {e}"

    