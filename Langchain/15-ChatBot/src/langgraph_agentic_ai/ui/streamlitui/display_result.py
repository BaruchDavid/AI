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

        if usecase.strip() == "Basic Chatbot":

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
            res = graph.invoke(initial_state)  ##invoke to build the graph with all nodes and edges
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

        if usecase.strip() == "AI News":
            frequency = self.user_message
            with st.spinner("Fetching and summarizing AI news..."):
                result = graph.invoke({"messages": frequency})
                try:
                    AI_NEWS_PATH = f"./AINews/{frequency.lower()}_summary.md"
                    with open(AI_NEWS_PATH, "r", encoding="utf-8") as f:
                        markdown_content = f.read()

                    st.markdown(markdown_content, unsafe_allow_html=True)
                except FileNotFoundError:
                    st.error(f"News Not Generated or File not found : {AI_NEWS_PATH}. Please check the frequency and try again.")
                except Exception as e:
                    st.error(f"An error occurred while reading the news summary: {str(e)}")
