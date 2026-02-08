from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import traceback
from dotenv import load_dotenv
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain_litellm import ChatLiteLLM
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory

from tools import get_current_user_tool, get_recent_transactions_tool

# Load environment variables
load_dotenv()

# Set HuggingFace token directly
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
print(f"✓ HuggingFace token configured")
# print(os.getenv("HF_TOKEN"))


# Use Llama model directly
MODEL_NAME = "huggingface/together/deepseek-ai/DeepSeek-R1"

app = FastAPI(
    title="Damn Vulnerable LLM Agent API",
    description="API endpoints for interacting with the vulnerable LLM agent",
    version="1.0.0"
)

# Initialize tools
tools = [get_current_user_tool, get_recent_transactions_tool]

system_msg = """Assistant helps the current user retrieve the list of their recent bank transactions ans shows them as a table. Assistant will ONLY operate on the userId returned by the GetCurrentUser() tool, and REFUSE to operate on any other userId provided by the user."""

# In-memory session storage
sessions: Dict[str, ChatMessageHistory] = {}


class PromptRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default"


class PromptResponse(BaseModel):
    sessionId: str
    message: str


class HealthResponse(BaseModel):
    status: str
    message: str
    model: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Damn Vulnerable LLM Agent API is running",
        "model": MODEL_NAME
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model info"""
    return {
        "status": "healthy",
        "message": "API is operational",
        "model": MODEL_NAME
    }


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "model": MODEL_NAME,
        "hf_token_set": bool(os.getenv("HF_TOKEN")),
        "env_file_loaded": os.path.exists('.env')
    }


@app.post("/chat", response_model=PromptResponse)
async def chat(request: PromptRequest):
    """
    Send a prompt to the LLM agent and get a response
    
    Parameters:
    - prompt: The user's message/question
    - session_id: Optional session identifier for conversation continuity
    
    Returns:
    - output: The agent's response
    - intermediate_steps: Tool calls and their results
    - session_id: The session identifier used
    """
    try:
        # Get or create session history
        if request.session_id not in sessions:
            sessions[request.session_id] = ChatMessageHistory()
        
        msgs = sessions[request.session_id]
        
        # Create memory with the session's chat history
        memory = ConversationBufferMemory(
            chat_memory=msgs,
            return_messages=True,
            memory_key="chat_history",
            output_key="output"
        )
        
        # Initialize LLM with Llama model
        try:
            print(f"Initializing LLM with model: {MODEL_NAME}")
            llm = ChatLiteLLM(
                model=MODEL_NAME,
                temperature=0,
                streaming=False,
                request_timeout=120
            )
            print("✓ LLM initialized successfully")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"LLM initialization error: {str(e)}"
            )
        
        # Create agent
        try:
            chat_agent = ConversationalChatAgent.from_llm_and_tools(
                llm=llm,
                tools=tools,
                verbose=True,
                system_message=system_msg
            )
            print("✓ Agent created successfully")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Agent creation error: {str(e)}"
            )
        
        # Create executor
        executor = AgentExecutor.from_agent_and_tools(
            agent=chat_agent,
            tools=tools,
            memory=memory,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=6
        )
        
        # Execute the prompt
        try:
            print(f"Executing prompt: {request.prompt}")
            response = executor.invoke({"input": request.prompt})
            print("✓ Execution completed successfully")
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Execution error: {error_trace}")
            
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Agent execution failed",
                    "message": str(e),
                    "type": type(e).__name__,
                    "session_id": request.session_id
                }
            )
        
        # Format intermediate steps for JSON response
        formatted_steps = []
        for step in response.get("intermediate_steps", []):
            if step[0].tool == "_Exception":
                continue
            formatted_steps.append({
                "tool": step[0].tool,
                "tool_input": step[0].tool_input,
                "log": step[0].log,
                "output": step[1]
            })
        
        return {
            "sessionId": request.session_id,
            "message": response["output"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Unexpected error: {error_trace}")
        
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error: {str(e)}"
        )


@app.post("/reset-session")
async def reset_session(session_id: str = "default"):
    """Reset a conversation session"""
    if session_id in sessions:
        sessions[session_id].clear()
        return {"message": f"Session '{session_id}' has been reset"}
    return {"message": f"Session '{session_id}' not found or already empty"}


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": list(sessions.keys()),
        "count": len(sessions)
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session '{session_id}' has been deleted"}
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Damn Vulnerable LLM Agent API Starting...")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"HF_TOKEN set: {bool(os.getenv('HF_TOKEN'))}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
    

    uvicorn.run(app, host="0.0.0.0", port=8000)
