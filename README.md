# RAG AI Agent Project

This project is an AI-powered research assistant that leverages large language models (LLMs) and external tools to generate research outputs, search the web, query Wikipedia, and save results to a file. It is built using [LangChain](https://python.langchain.com/), integrates with OpenAI and Anthropic LLMs, and uses DuckDuckGo and Wikipedia as information sources.

---

## Project Structure

```
.env
main.py
requirements.txt
tools.py
```

---

## File Overview

### `.env`

Stores your API keys for Anthropic and OpenAI:

```properties
ANTHROPIC_API_KEY = ""  
OPENAI_API_KEY = ""
```

---

### `requirements.txt`

Lists all Python dependencies required for the project:

```txt
langchain
wikipedia
langchain-openai
langchain-community
langchain-anthropic
python-dotenv
pydantic
duckduckgo-search
```

---

### `tools.py`

Defines utility tools for searching the web, querying Wikipedia, and saving research outputs to a file. These tools are integrated into the agent.

```python
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_txt_file(response: str, filename: str = "research_response.txt"):
    """
    Save the response to a text file with a timestamp.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"---Research Output---\nTimestamp: {timestamp}\n\n{response}\n\n"
    
    with open(filename, "a") as file:
        file.write(formatted_text)

    return f"Response saved to {filename}"

search = DuckDuckGoSearchRun()
search_tool = Tool(
    search,
    name="search_tool",
    func=search.run(),
    description="Use this tool to search the web for information. Provide a query string to get relevant results.",
)

wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=1000
    )
)

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt_file,
    description="Saves the response to a text file with a timestamp. Use this tool to save the research output.",
)
```

---

### `main.py`

The main entry point. It loads environment variables, sets up LLMs, defines the output schema, configures the agent with tools, and runs the agent in an interactive loop.

```python
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
```

---

## How It Works

1. **Environment Setup**: Loads API keys from `.env` for OpenAI and Anthropic.
2. **Tool Definition**: `tools.py` defines tools for web search, Wikipedia queries, and saving results.
3. **LLM Setup**: Both OpenAI and Anthropic chat models are initialized.
4. **Prompt & Output Schema**: The agent is prompted to act as a research assistant and must return results in a structured format defined by the `ResearchResponse` schema.
5. **Agent Creation**: The agent is created with the LLM, prompt, and tools.
6. **Execution**: The user enters a research query. The agent uses the tools as needed, generates a structured response, and optionally saves the output to a file.

---

## Usage

1. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

2. **Set your API keys** in `.env`:
    ```
    ANTHROPIC_API_KEY = "your_anthropic_key"
    OPENAI_API_KEY = "your_openai_key"
    ```

3. **Run the assistant**:
    ```sh
    python main.py
    ```

4. **Enter your research question** when prompted.

---

## Key Components

- **LLMs**: Uses OpenAI's GPT-4o and Anthropic's Claude 3 Sonnet.
- **Tools**: 
    - Web search via DuckDuckGo
    - Wikipedia queries
    - Save research output to a timestamped text file
- **Structured Output**: Ensures responses are consistent and easy to parse.

---

## Extending the Project

- Add more tools (e.g., other search engines, PDF summarization).
- Enhance the output schema for more detailed research reports.
- Integrate with a front-end or chat interface.

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/)
- [Anthropic API](https://docs.anthropic.com/claude/docs/overview)

---

**This project provides a foundation for building advanced research assistants using LLMs
