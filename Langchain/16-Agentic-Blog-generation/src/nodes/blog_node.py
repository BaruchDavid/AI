from src.states.blogstate import BlogState


class BlogNode:
    """Class representin blog node"""

    def __init__(self, llm):
        self.llm = llm

    def title_creation(self, state: BlogState) -> str:
        """
        create the title for the blog
        """

        if "topic" in state and state["topic"]:
            prompt = """
            
                   You are an expert blog content writer. Use Markdown formatting. Generate a blog title for the {topic}. 
                   This t itle should be crated and SEO friendly
                   
                   """

            system_message = prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog": {"title": response.content}}

    def content_generation(self, state: BlogState) -> str:
        """
        Generate the content for the blog based on the title
        """

        if "topic" in state and state["topic"]:
            prompt = """
            
                   You are an expert blog content writer. Use Markdown formatting. Generate a detailed blog content with detailed breakdown for the for the {topic}. 
                   This content should be crated and SEO friendly
                   
                   """

            system_message = prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog": {"title": state["blog"]["title"], "content": response.content}}
