from langchain.llms import LlamaCpp
from pydantic import BaseModel
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.tools import Tool
import requests
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory

# Load the LLM model
llm = LlamaCpp(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=256,
    verbose=True
)

# Define a Pydantic model for tool calls
class ToolCall(BaseModel):
    action: str
    action_input: str

# Create parsers to handle tool outputs
parser = PydanticOutputParser(pydantic_object=ToolCall)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

# Define a web search tool
def search_web(query):
    response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
    data = response.json()
    return data.get("Abstract") or (data.get("RelatedTopics")[0]["Text"] if data.get("RelatedTopics") else "No results found.")

search_tool = Tool(
    name="WebSearch",
    func=search_web,
    description="Use this tool to search for real-world information."
)

# Define a summarization tool
def summarize_text(text):
    prompt = f"Summarize the following text in one paragraph:\n{text}"
    return llm(prompt)

summarization_tool = Tool(
    name="Summarizer",
    func=summarize_text,
    description="Use this tool to summarize long pieces of text."
)

# Define the custom prompt
template = """You are a helpful AI assistant with access to tools.

You must follow the format below exactly when you want to use a tool:

Available tools:
- WebSearch: Use this to search for real-world information.
- Summarizer: Use this to summarize long pieces of text.

Respond to the following input:
{input}
"""
custom_prompt = PromptTemplate(input_variables=["input"], template=template)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the tools
tools = [search_tool, summarization_tool]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_kwargs={"prompt": custom_prompt} 
)

# Create an agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True  
)

# Example usage
if __name__ == "__main__":
    user_input = "Summarize this article: OpenAI released a new model."
    response = agent_executor.invoke({"input": user_input})
    print(response)
