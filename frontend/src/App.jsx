import React from "react";
import { useNavigate } from "react-router-dom";
import "./App.css";

function App() {
  const navigate = useNavigate();

  return (
    <div className="app-container">
      <h1 className="main-heading">Welcome to <span className="highlight">FINRAG</span></h1>
      <p className="sub-heading">Your one-stop Financial Research Assistant</p>
      <div className="button-container">
        <button className="nav-button" onClick={() => navigate("/Chatbot")}>
          Ask Chatbot for Queries
        </button>
        <button className="nav-button" onClick={() => navigate("/analyze")}>
          Analyze Stocks & View Summaries
        </button>
      </div>
    </div>
  );
}

export default App;
    
