import React,{useState} from "react";
import axios from "axios";


function Chatbot(){
    const [query,setQuery]=useState("")
    const [response,setResponse]=useState("")
    const [loading,setLoading]=useState(false)

    const handleSend=async()=>{
        try{
            setLoading(true)
            const res=await axios.post("http://localhost:8000/query",{input_query:query})
            setResponse(res.data)
            setQuery("")
            setLoading(false)
        }
        catch(error){
            console.error("Error sending query:", error);
            setResponse("An error occurred while processing your query.");
        }
    }

    return(
        <div>
            <h1>Ask FINRAG</h1>
            <input value={query} onChange={(e)=>setQuery(e.target.value)} placeholder="Type your question here..."/>
            <button onClick={handleSend}>Send</button>

            {loading && <p>Processing your query and giving response</p>}
            
            {response && (
                <div>
                    <h3>Chain of Thought</h3>
                    <ul>
                        {response?.chain_of_thought?.map((step,idx)=>(
                            <div key={idx}>

                                <strong>Thought</strong>{step.thought}<br/>
                                <strong>Action</strong>{step.action}<br/>
                                <strong>Observation</strong>{step.observation}<br/>
                            </div>
                        ))}
                        
                    </ul>
                    <h2>Final Answer</h2>
                    <p>{response?.final_answer}</p>
                </div>
            )}
        </div>
    )
    

}
export default Chatbot;