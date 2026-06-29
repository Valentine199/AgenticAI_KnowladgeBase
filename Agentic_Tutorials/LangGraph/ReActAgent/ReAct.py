from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv("../../LLMBasics/.env")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # use the add_messages reducer during change


@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b

@tool
def subtract(a: int, b: int):
    """This is a subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """This is a multiplication function."""
    return a * b

tools = [add, subtract, multiply]

llm = ChatOllama(model="llama3.2:3b").bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)

graph.add_node("our_agent", model_call)
graph.add_edge(START, "our_agent")

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_conditional_edges("our_agent", should_continue, {
    "continue": "tools",
    "end": END
})

graph.add_edge("tools", "our_agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))


