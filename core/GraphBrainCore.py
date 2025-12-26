from __future__ import annotations
import os
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import tools

class AgentState(TypedDict):
    messages: list

class FridayGraph:
    def __init__(self):
        # We define the tools for the LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            api_key=os.getenv("GROQ_API_KEY")
        ).bind_tools([tools.write_code_to_file, tools.read_local_file])

    def call_brain(self, state: AgentState):
        """The thinking step."""
        response = self.llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    def build(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("thought_engine", self.call_brain)
        workflow.set_entry_point("thought_engine")
        workflow.add_edge("thought_engine", END)
        return workflow.compile()
