import React, { useState } from "react";
import Plot from "react-plotly.js";
import axios from "axios";

function ChartView() {
  const [chartData, setChartData] = useState({});
  const [tickers, setTickers] = useState("");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [interval, setInterval] = useState("1d");
  const [loading,setLoading]=useState(false);
  const [summary,setSummary]=useState("")


  const handleChartData = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setChartData({})
      const response = await axios.post("http://localhost:8000/chart", {
        tickers: tickers.split(",").map((t) => t.trim().toUpperCase()),
        start_date: start,
        end_date: end,
        interval: interval,
      });
      setChartData(response.data);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching chart data:", error);
      setChartData({
        error: "Failed to fetch chart. Try again later.",
      });
    }
  };

  //pass the parsed data to get summary by llm
  const get_summary=async(parsed)=>{
    try{
        const res=await axios.post("http://localhost:8000/summary",{
            data:parsed.data
    })
    setSummary(res.data.summary)

  }
  catch(err){
    setSummary("Error fetching summary. Please try again later.");
    }
}

  // Utility to render Plotly chart from JSON string
  const renderPlot = (parsed) => {
    try {
      
      return (
        <Plot
          data={parsed.data}
          layout={{ ...parsed.layout, autosize: true }}
          config={{ responsive: true }}
          style={{ width: "100%", height: "100%" }}
        />
      );
    } catch (err) {
      return <p>Error parsing chart.</p>;
    }
  };

    const parsedLineChart = chartData?.line_chart ? JSON.parse(chartData.line_chart) : null;
    console.log(parsedLineChart)
    const parsedRollingAvg = chartData?.rolling_average ? JSON.parse(chartData.rolling_average) : null;
    const parsedReturns = chartData?.["30_day_returns"] ? JSON.parse(chartData["30_day_returns"]) : null;
    const parsedCandle = chartData?.candlestick_chart ? JSON.parse(chartData.candlestick_chart) : null;

  return (
    <div style={{ padding: "20px" }}>
      <h2>Stock Analysis Dashboard</h2>
      <form onSubmit={handleChartData} style={{ marginBottom: "20px" }}>
        <label>Tickers:</label>
        <input
          type="text"
          value={tickers}
          onChange={(e) => setTickers(e.target.value)}
          placeholder="e.g., AAPL,GOOGL"
        />
        <label>Start Date:</label>
        <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
        <label>End Date:</label>
        <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
        <label>Interval:</label>
        <select value={interval} onChange={(e) => setInterval(e.target.value)}>
          <option value="1d">1 day</option>
          <option value="1wk">1 week</option>
          <option value="1mo">1 month</option>
        </select>
        <button type="submit">Generate Info & Charts</button>
      </form>

      {/* Company Info */}
      {loading && <p>Loading chart data...</p>}
      {chartData.info && (
        <div style={{ marginBottom: "20px" }}>
          <h3>Company Info</h3>
          {Object.entries(chartData.info).map(([ticker, data]) => (
            <div key={ticker} style={{ borderBottom: "1px solid #ccc", paddingBottom: "10px" }}>
              <h4>{ticker}</h4>
              <p><strong>Name:</strong> {data.name}</p><br></br>
              <p><strong>Sector:</strong> {data.sector}</p><br></br>
              <p><strong>Industry:</strong> {data.industry}</p><br></br>
              <p><strong>Country:</strong> {data.country}</p><br></br>
              <p><strong>Market Cap:</strong> {data.market_cap} Billion(USD)</p>
              <p><strong>Summary:</strong> {data.info}</p>
            </div>
          ))}
        </div>
      )}

      {/* Line Chart */}
      {chartData?.line_chart && (
        <div style={{ marginBottom: "1000px" }}>
          <h3>Line Chart</h3>

          {renderPlot(parsedLineChart)}

          <div>
            <h3>Analysis and breakdown</h3>
            <button onClick={()=>get_summary(parsedLineChart)}>Generate Analysis of chart</button>
            {summary && (
                <div>
                    <h4>Summary</h4>
                    <p>{summary}</p>
                </div>
            )}
         </div>
        </div>
      )}

      {/* Rolling Average */}
      {chartData?.rolling_average && (
        <div style={{ marginBottom: "400px" }}>
          <h3>Rolling Average</h3>
          {renderPlot(parsedRollingAvg)}
        </div>
      )}

      {/* 30 Day Returns OR Candlestick */}
      {chartData["30_day_returns"] && (
        <div style={{ marginBottom: "400px" }}>
          <h3>30 Day Returns</h3>
          {renderPlot(parsedReturns)}
        </div>
      )}

      {chartData?.candlestick_chart && (
        <div style={{ marginBottom: "400px" }}>
          <h3>Candlestick Chart</h3>
          {renderPlot(parsedCandle)}
        </div>
      )}
    </div>
  );
}

export default ChartView;
