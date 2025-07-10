import React from "react";
import Chatbot from "./components/Chatbot";
import ChartView from "./components/ChartView";

function App(){
    return(
        <div>
            <h1>Welcome to FINRAG</h1>
            <p>FINRAG is a financial research assistant that helps you analyze stock market data and trends.</p>
            <p>Use the chatbot to ask questions about stocks, markets, or trends.</p>
            <p>Use the chart view to visualize stock data.</p>
            <ChartView/>
        </div>
    )
}
export default App;