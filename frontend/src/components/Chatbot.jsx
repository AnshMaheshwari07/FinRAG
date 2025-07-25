import React, { useState } from "react";
import axios from "axios";
import "./Chatbot.css";  // ✅ Import the CSS

function Chatbot() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    try {
      setLoading(true);
      const res = await axios.post("http://localhost:8000/query", {
        input_query: query,
      });
      setResponse(res.data);
      setQuery("");
      setLoading(false);
    } catch (error) {
      console.error("Error sending query:", error);
      setResponse({
        final_answer: "An error occurred while processing your query.",
        chain_of_thought: [],
      });
      setLoading(false);
    }
  };

  return (
    <div className="chatbot-container">
      <h1 className="chatbot-title">Ask FINRAG</h1>

      <div className="chatbox">
        <input
          className="chat-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type your question here..."
        />
        <button className="chat-send" onClick={handleSend}>
          Send
        </button>
      </div>

      {loading && <p className="loading-text">Processing your query...</p>}

      {response && (
        <div className="chat-response">
          {response?.chain_of_thought?.length > 0 && (
            <>
              <h3>Chain of Thought</h3>
              <ul>
                {response?.chain_of_thought?.map((step, idx) => (
                  <li key={idx} className="chat-step">
                    <p><strong>Thought:</strong> {step.thought}</p>
                    <p><strong>Action:</strong> {step.action}</p>
                    <p><strong>Observation:</strong> {step.observation}</p>
                  </li>
                ))}
              </ul>
            </>
          )}
          <h2>Final Answer</h2>
          <p className="final-answer">{response?.final_answer}</p>
        </div>
      )}
    </div>
  );
}

export default Chatbot;
