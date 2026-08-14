from src.states.blogstate import BlogState
from langchain_core.messages import SystemMessage, HumanMessage
from src.states.blogstate import Blog


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

    def translation(self, state: BlogState) -> str:
        """
        Translate the blog content to the specified language
        """
        blog_content = state["blog"]["content"]
        language = state["current_language"]

        prompt = f"""

                You are an expert translator. Use Markdown formatting. Translate the following blog content to {language}
                - Maintain the original tone, style, and formatting of the content.
                - Ensure that the translated content is culturally appropriate and relevant for the target audience.
                - Adapt cultural references, idioms, and examples to resonate with the target audience.

                ORIGINAL CONTENT:
                {blog_content}

                """

        response = self.llm.invoke(prompt)
        return {"blog": {"title": state["blog"]["title"], "content": response.content}}

    def route(self, state: BlogState):
        """set current language, which will be passed by lambda function to the translation node"""
        return {"current_language": state["current_language"]}

    def route_decision(self, state: BlogState):
        if state["current_language"] == "german":
            return "german"
        elif state["current_language"] == "french":
            return "french"
        else:
            return state["current_language"]
