from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv("../../LLMBasics/.env")

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatOllama(
    model="deepseek-r1:1.5b"
)

def process(state:AgentState) -> AgentState:
    """This Node will process and answer to your input"""
    response = llm.invoke(state['messages'])
    response.content = response.content.split("</think>")[-1]
    state["messages"].append(AIMessage(content=response.content))

    print(f"\nAI: {response.content}")

    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter something: ")
while user_input != "Exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter something: ")

with open("logging.txt", "w") as file:
    file.write("Conversation log:\n")
    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            file.write(f"You: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            file.write(f"AI: {msg.content}\n")
    file.write("Conversation ended")

print("Conversation saved!")