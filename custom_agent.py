from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.agents.agent import AgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_core.runnables import Runnable
from langchain.memory import ConversationBufferMemory
from langchain.agents import StructuredChatAgent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.llms import LlamaCpp
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import LlamaCpp
import requests
import os

n_threads = 4

llm = LlamaCpp(
    model_path="models/deepseek-llm-7b-chat.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=256,
    n_ctx=2048,
    use_mlock=True,
    use_mmap=False,
)

def search_web(query):
    response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
    return response.json()["Abstract"] or response.json()["RelatedTopics"][0]["Text"]

search_tool = Tool(
    name="WebSearch",
    func=search_web,
  description="Use this tool ONLY when you need to look up fresh or external information that is NOT already provided in the prompt."
)

def summarize_text(query):
    prompt = f"Summarize the following text in one paragraph:\n{query}"
    return llm(prompt)

summarization_tool = Tool(
    name="Summarizer",
    func=summarize_text,
    description="Use this tool ONLY if the user provides a block of text that needs to be summarized."
)

tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

prompt = ChatPromptTemplate.from_template("""
        You are a helpful agent. You can use the following tools:

        {tool_descriptions}

        Use the following format:

        Question: {input}
        Thought: You should think about what to do
        Action: the name of the tool to use
        Action Input: the input to the tool
        Observation: the result of the action
        ... (repeat Thought/Action/Observation as needed)
        Thought: I now know the final answer
        Final Answer: the answer to the question

        ### Example 1
        Question: What is the latest news about AI?
        Thought: The user is asking for fresh information not already in the prompt. I should use WebSearch.
        Action: <WebSearch>
        Action Input: <"latest news about AI">
        Final Answer: <your answer here>

        ### Example 2
        Question: Summarize this: 'AI is rapidly evolving. Deep learning has made significant progress in the last few years...'
        Thought: The user has provided a block of text. I should summarize it.
        Action: <Summarizer>
        Action Input: <"AI is rapidly evolving. Deep learning has made significant progress...">
        Final Answer: <your answer here>


        Begin!

        {agent_scratchpad}
        """)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools=[search_tool, summarization_tool]

llm_chain = LLMChain(llm=llm, prompt=prompt)

agent = StructuredChatAgent(llm_chain=llm_chain, tools=[search_tool, summarization_tool])

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[search_tool, summarization_tool],
    memory=memory,
    verbose=True
)

if __name__ == "__main__":
    user_input = "Summarize this article: OpenAI released a new model."
    response = agent_executor.invoke({"input": user_input})
    print(response)
