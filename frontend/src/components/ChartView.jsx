import React, { useState, useCallback, useMemo } from "react";
import Plot from "react-plotly.js";
import axios from "axios";
import "./ChartView.css"

function ChartView() {
  const [tickers, setTickers] = useState("");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [interval, setInterval] = useState("1d");
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState({});
  const [summaries, setSummaries] = useState({
    "Line Chart": "",
    "Rolling Average Chart": "",
    "30-day returns": "",
    "Candlestick Chart": "",
    "Area Chart":"",
  });


  const parsedLine = useMemo(() => {
    return chartData.line_chart ? JSON.parse(chartData.line_chart) : null;
  }, [chartData.line_chart]);

  const parsedRolling = useMemo(() => {
    return chartData.rolling_average ? JSON.parse(chartData.rolling_average) : null;
  }, [chartData.rolling_average]);

  const parsedReturns = useMemo(() => {
    return chartData["30_day_returns"] ? JSON.parse(chartData["30_day_returns"]) : null;
  }, [chartData["30_day_returns"]]);

  const parsedCandle = useMemo(() => {
    return chartData.candlestick_chart ? JSON.parse(chartData.candlestick_chart) : null;
  }, [chartData.candlestick_chart]);

  const parsedArea = useMemo(() => {
    return chartData.area_chart ? JSON.parse(chartData.area_chart) : null;
  }, [chartData.area_chart]);

  const handleChartData = useCallback(async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const res = await axios.post("http://localhost:8000/chart", {
        tickers: tickers.split(",").map(t => t.trim().toUpperCase()),
        start_date: start,
        end_date: end,
        interval
      });
      setChartData(res.data);
      // Reset summaries
      setSummaries({
        "Line Chart": "",
        "Rolling Average Chart": "",
        "30-day returns": "",
        "Candlestick Chart": "",
        "Area Chart":""
      });
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [tickers, start, end, interval]);

  console.log("info : ",tickers,start,end);
  const getSummary = useCallback(async (chartType) => {
    setSummaries(prev => ({ ...prev, [chartType]: "Loading…" }));
    if (!tickers || !start || !end) {
    alert("Please fill in the ticker and date range first.");
    return;
}
      
    const { data } = await axios.post("http://localhost:8000/summary", {
      
      tickers: tickers.split(",").map(t => t.trim().toUpperCase()),
      start_date: start,
      end_date: end,
      interval,
      chart: chartType
    });

    setSummaries(prev => ({ ...prev, [chartType]: data.summary }));
  }, [tickers, start, end, interval]);

  const renderPlot = (parsed) => (
    <Plot data={parsed.data} layout={{ ...parsed.layout, autosize: true }} config={{ responsive: true }} style={{ width: "100%", height: "100%" }} />
  );

  return (
  <div className="dashboard-container">
    <h2>Stock Analysis Dashboard</h2>

    <div className="form-container">
      <form onSubmit={handleChartData}>
        <input type="text" value={tickers} onChange={(e) => setTickers(e.target.value)} placeholder="e.g., AAPL, GOOGL" />
        <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
        <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
        <select value={interval} onChange={(e) => setInterval(e.target.value)}>
          <option value="1d">1 day</option>
          <option value="1wk">1 week</option>
          <option value="1mo">1 month</option>
        </select>
        <button type="submit">Generate Info & Charts</button>
      </form>
    </div>

    {loading && <p>Loading chart data...</p>}

    <div className="content-wrapper">
      {/* Left: Company Info */}
      {chartData.info && (
        <div className="company-info">
          <h3>Company Info</h3>
          {Object.entries(chartData.info).map(([ticker, data]) => (
            <div key={ticker}>
              <h4>{ticker}</h4>
              <p><strong>Name:</strong> {data.name}</p>
              <p><strong>Sector:</strong> {data.sector}</p>
              <p><strong>Industry:</strong> {data.industry}</p>
              <p><strong>Country:</strong> {data.country}</p>
              <p><strong>Market Cap:</strong> {data.market_cap} USD</p>
              <p><strong>Summary:</strong> {data.info}</p>
            </div>
          ))}
        </div>
      )}
  
      {/* Right: Chart Sections */}
      <div className="charts-panel">
        {parsedLine && (
          <div className="chart-section">
            <h3>Line Chart</h3>
            <div className="plot-wrapper">{renderPlot(parsedLine)}</div>
            
            <button onClick={() => getSummary("Line Chart")}>Generate Summary</button>
            {summaries["Line Chart"] && <p>{summaries["Line Chart"]}</p>}
          </div>
        )}

        {parsedRolling && (
          <div className="chart-section">
            <h3>Rolling Average</h3>
            <div className="plot-wrapper">{renderPlot(parsedRolling)}</div>
            <button onClick={() => getSummary("Rolling Average Chart")}>Generate Summary</button>
            {summaries["Rolling Average Chart"] && <p>{summaries["Rolling Average Chart"]}</p>}
          </div>
        )}

        {parsedReturns && (
          <div className="chart-section">
            <h3>30-day Returns</h3>
            <div className="plot-wrapper">{renderPlot(parsedReturns)}</div>
            <button onClick={() => getSummary("30-day returns")}>Generate Summary</button>
            {summaries["30-day returns"] && <p>{summaries["30-day returns"]}</p>}
          </div>
        )}

        {parsedCandle && (
          <div className="chart-section">
            <h3>Candlestick Chart</h3>
            <div className="plot-wrapper">{renderPlot(parsedCandle)}</div>
            <button onClick={() => getSummary("Candlestick Chart")}>Generate Summary</button>
            {summaries["Candlestick Chart"] && <p>{summaries["Candlestick Chart"]}</p>}
          </div>
        )}

        {parsedArea && (
          <div className="chart-section">
            <h3>Area Chart</h3>
            <div className="plot-wrapper">{renderPlot(parsedArea)}</div>
            <button onClick={() => getSummary("Area Chart")}>Generate Summary</button>
            {summaries["Area Chart"] && <p>{summaries["Area Chart"]}</p>}
          </div>
        )}
      </div>
    </div>
  </div>
)
}
export default ChartView
