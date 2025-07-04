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

