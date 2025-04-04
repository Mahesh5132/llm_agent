from langchain.llms import LlamaCpp
from pydantic import BaseModel
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import Tool
import requests
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType
from langchain.agents import AgentExecutor

llm = LlamaCpp(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=256,
    verbose=True
)


class ToolCall(BaseModel):
    action: str
    action_input: str

  # or any LLM you're using

# Underlying parser
parser = PydanticOutputParser(pydantic_object=ToolCall)


# Wrap with OutputFixingParser
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)


def search_web(query):
    response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
    return response.json()["Abstract"] or response.json()["RelatedTopics"][0]["Text"]

search_tool = Tool(
    name="WebSearch",
    func=search_web,
    description="Use this tool to search for real-world information."
)
def summarize_text(text):
    prompt = f"Summarize the following text in one paragraph:\n{text}"
    return llm(prompt)

summarization_tool = Tool(
    name="Summarizer",
    func=summarize_text,
    description="Use this tool to summarize long pieces of text."
)


custom_prompt = PromptTemplate(
    input_variables=["input"],
    template="""You are a helpful AI assistant with access to tools.

You must follow the format below exactly when you want to use a tool:

Available tools:
- WebSearch: Use this to search for real-world information.
- Summarizer: Use this to summarize long pieces of text.

Example 1 (search the web):

Example 2 (summarize text):

Respond to the following input:
{input}
"""
)



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools=[search_tool, summarization_tool]

agent = initialize_agent(
    tools= tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_kwargs={"prompt": custom_prompt} 
)


agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True  
)