from langchain.llms import LlamaCpp
from langchain.agents import LLMSingleActionAgent, AgentExecutor
from langchain.agents.agent import AgentExecutor
from langchain.agents import Tool
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import os
import re
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

n_threads = 4

llm = LlamaCpp(
    model_path="models/deepseek-llm-7b-chat.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=256,
    n_ctx=2048, # Stops generation here
    use_mlock=True,
    use_mmap=False,
)

from duckduckgo_search import DDGS
from langchain.agents import Tool


def search_ai_news(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
    if not results:
        return "No relevant information found."
    return "\n".join(f"{r['title']} - {r['href']}" for r in results)

# Create the Tool
tools = [
    Tool(
        name="WebSearch",
        func=search_ai_news,
        description="Use this tool to search the web for recent or live information, like news or trends."
    )
]


prompt = PromptTemplate.from_template("""
You are a helpful AI agent. You have access to the following tool:

{tool_names}

Use the following format:

Question: {input}
Thought: you should think about what to do
Action: the name of the tool to use (only one of: {tool_names})
Action Input: the input to the tool
Observation: the result of the action
... (this Thought/Action/Observation can repeat 3 times)
Thought: I now know the final answer
Final Answer: the answer to the question

IMPORTANT:
- You must end with 'Final Answer'.
- Do not keep repeating actions.
- Use the observations to form your final answer.

Begin!

{agent_scratchpad}
""")


class CustomReActParser(AgentOutputParser):
    def parse(self, text: str):
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text,
            )
        match = re.search(r"Action: (.*)\nAction Input: (.*)", text)
        if not match:
            raise ValueError(f"Could not parse output: `{text}`")
        return AgentAction(
            tool=match.group(1).strip(),
            tool_input=match.group(2).strip().strip('"'),
            log=text,
        )


llm_chain = LLMChain(llm=llm, prompt=prompt.partial(tool_names="WebSearch", agent_scratchpad=""))

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=CustomReActParser(),
    stop=["\nObservation:"],
    allowed_tools=["WebSearch"]
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True , max_iterations=3,  # Stops after 3 steps
)

if __name__ == "__main__":
    response = agent_executor.invoke({"input": "What is the latest news about AI?"})
    print(response)