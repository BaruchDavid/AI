from langgraph.graph import StateGraph, START, END
from src.llms.groqllm import GroqLLM
from src.states.blogstate import BlogState
from src.nodes.blog_node import BlogNode


class GraphBuilder:
    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(BlogState)

    def build_topic_graph(self):
        """
        Build a graph to generate blogs based on topic
        """

        blog_node = BlogNode(self.llm)

        # Add nodes to the graph
        self.graph.add_node("title_creation", blog_node.title_creation)
        self.graph.add_node("content_generation", blog_node.content_generation)

        # Add edges to the graph
        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generation")
        self.graph.add_edge("content_generation", END)

        return self.graph

    def build_graph_with_language(self):
        """
        Build a graph to generate blogs based on language
        """

        blog_node = BlogNode(self.llm)

        # Add nodes to the graph
        self.graph.add_node("title_creation", blog_node.title_creation)
        self.graph.add_node("content_generation", blog_node.content_generation)
        ## lege als key-value paar fest, dass 'current_langugage' german ist
        self.graph.add_node("translate_to_german", lambda state: blog_node.translation({**state, "current_language": "german"}))
        ## lege als key-value paar fest, dass 'current_langugage' german ist
        self.graph.add_node("translate_to_french", lambda state: blog_node.translation({**state, "current_language": "french"}))

        self.graph.add_node("route", blog_node.route)

        # Add edges and conditional edges
        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generation")
        self.graph.add_edge("content_generation", "route")

        # decide, which node should be executed based on the current_language in the state
        ## source: der Knoten, von dem die bedingte Kante ausgeht (bei dir: "route")
        ## path: die Funktion, die zur Laufzeit entscheidet, wohin es geht (bei dir: blog_node.route_decision)
        ## path_map: das Dict, das den Rückgabewert von path auf den tatsächlichen Zielknoten abbildet
        self.graph.add_conditional_edges(
                                            source="route",
                                            path=blog_node.route_decision,
                                            path_map={"german": "translate_to_german", "french": "translate_to_french"},
                                        )
        self.graph.add_edge("translate_to_german", END)
        self.graph.add_edge("translate_to_french", END)

        return self.graph

    def setup_graph(self, usecase: str):
        """
        Setup the graph with the initial state
        """
        if usecase == "topic":
            self.build_topic_graph()
        if usecase == "language":
            self.build_graph_with_language()

        return self.graph.compile()


## below code is for the langsmith langgraph studio
llm = GroqLLM().get_llm()

## get the graph
graph_builder = GraphBuilder(llm)
graph = graph_builder.build_topic_graph().compile()
