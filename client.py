import streamlit as st
import asyncio
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
load_dotenv()

# Streamlit app title
st.title("PPO Counterfactual Explanation Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize LangGraph agent
if "graph" not in st.session_state:
    async def setup_graph():
        # Initialize MCP client
        client = MultiServerMCPClient({
            "counterfactual_server": {
                "url": "http://localhost:8080/mcp",
                "transport": "streamable_http"
            }
        })
        mcp_tools = await client.get_tools()
        st.session_state["mcp_tools"] = mcp_tools
        
        # Initialize LLM
        llm = ChatOllama(model="qwen3:32b")
        model_with_tools = llm.bind_tools(mcp_tools)
        
        # Define state graph
        def should_continue(state):
            return "tools" if state["messages"][-1].tool_calls else END
        
        def call_model(state):
            response = model_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", ToolNode(mcp_tools))
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", should_continue, ["tools", END])
        builder.add_edge("tools", "call_model")
        return builder.compile()
    
    st.session_state["graph"] = asyncio.run(setup_graph())

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "details" in message:
            with st.expander("Response Details"):
                st.json(message["details"])

# Handle user input
query = st.chat_input("Enter your query (e.g., 'Generate counterfactuals for breast_cancer.csv' or 'Generate 5 counterfactuals for sample index 0'):")
if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    async def run_query():
        graph = st.session_state["graph"]
        result = await graph.ainvoke({
            "messages": [{"role": "user", "content": query}]
        })
        response = result["messages"][-1].content
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response,
            "details": result
        })
        with st.chat_message("assistant"):
            st.markdown(response)
            with st.expander("Response Details"):
                st.json(result)
    
    asyncio.run(run_query())