from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv("../../LLMBasics/.env")


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatOllama(
    model="deepseek-r1:1.5b"
)

def process(state:AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    response.content = response.content.split("</think>")[-1]
    print(f"\nAI: {response.content}")

    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter something: ")
while user_input != "Exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter something: ")
