from curses import raw
from html import parser
from urllib import response
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from tools import search_tool, wikipedia_tool, save_tool


load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

LLM_OPENAI = ChatOpenAI(model="gpt-4o")
LLM_ANTHROPIC = ChatAnthropic(model="claude-3-sonnet-20240229")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        """
        You are a research assistant that will help generate a research paper, along with the proper citations.
        Answer the user query and use necessary tools.
        Wrap the output in this format and provide no other text\n{format_instructions}
        """
        ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
tools = [search_tool, wikipedia_tool, save_tool]

AGENT = create_tool_calling_agent(
    llm=LLM_OPENAI,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=AGENT,
    tools=[],
    verbose=True,
)

query = input("What can I help you with today? ")

raw_response = agent_executor.invoke({"query": query, "chat_history": []})


try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print("Structured Response:", structured_response)
except Exception as e:
    print(f"Error parsing response: {e} - Raw Response: {raw_response}")