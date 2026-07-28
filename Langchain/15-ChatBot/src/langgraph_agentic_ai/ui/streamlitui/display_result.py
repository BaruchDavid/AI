import json
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class DisplayResultSteamlit:

    def __init__(self, usecase, graph, user_message):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def display_result(self):
        """
        Displays the result of the LangGraph AgenticAI application in a Streamlit interface.
        It shows the user message, the graph structure, and the response from the LLM.
        """
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message

        if usecase == "Basic Chatbot":

            for event in graph.stream({"messages": ("user", user_message)}):
                print(event.values())
                for value in event.values():
                    print(value["messages"])
                    with st.chat_message("user"):
                        st.write(user_message)
                    with st.chat_message("assistant"):
                        st.write(value["messages"].content)

        if usecase.strip() == "Chatboot with Tools":
            initial_state = {"messages": [user_message]}
            res = graph.invoke(initial_state)
            for messages in res["messages"]:
                if type(messages) == HumanMessage:
                    with st.chat_message("user"):
                        st.write(messages.content)
                elif type(messages) == ToolMessage:
                    with st.chat_message("ai"):
                        st.write("Tool Call Start")
                        st.write(messages.content)
                        st.write("Tool Call End")

                elif type(messages) == AIMessage and messages.content:
                    with st.chat_message("assistant"):
                        st.write(messages.content)
