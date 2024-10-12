# imports for llm with its embadding
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# imports for RAG
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool # to make the (RAG) retriever as tool
# imports for pre defiened tool
from langchain_community.tools.tavily_search import TavilySearchResults 
# imports for predefined prompts
from langchain import hub 
# imports for agent
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor, tool
# imports for mnage the chat history and its chain
from langchain_community.chat_message_histories import ChatMessageHistory  
from langchain_core.runnables.history import RunnableWithMessageHistory
 
# imports for env loading
from dotenv import load_dotenv
import os

load_dotenv()


# load the GOOGLE_API_KEY
google_api_key: str | None  =  os.getenv("GOOGLE_API_KEY")

# Initialize an instance of the ChatGoogleGenerativeAI with specific parameters
llm:ChatGoogleGenerativeAI =  ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Specify the model to use
    temperature=0.2,          
)

# response = llm.invoke("what is machine learning")
# print(response)

# tavily_api_key= os.getenv("Tavily_API_KEY")
tavily_api_key: str | None = os.getenv("Tavily_API_KEY")

# Ensure the Tavily API key is available
if not tavily_api_key:
    raise ValueError("Tavily_API_KEY must be set in the environment variables.")

search_tool:TavilySearchResults  = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    # include_raw_content=True,
    # include_images=True,
    )

# RAG 
loader:WebBaseLoader  = WebBaseLoader("https://www.techloset.com/")
docs:list = loader.load()

documents:list = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)

embadding:GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector:FAISS  = FAISS.from_documents(documents,embedding=embadding)

retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "techloset_search",
    "search for information about techloset. For any question about Techlost Solutions, You must use this tool."
    )

# predefined prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

# craete agent for tools
tools:list = [search_tool, retriever_tool]

# agent = create_tool_calling_agent(llm, tools, prompt)
agent = create_tool_calling_agent(llm, tools, prompt)

# agent = create_tool_calling_agent(model, tools, prompt)

agent_exicuter:AgentExecutor  = AgentExecutor(agent=agent, tools=tools, verbose=True)
# response = agent_exicuter.run("which servises are provided by techlost")
# print(response)


# to mnage yhe chat history
message_history:ChatMessageHistory  =ChatMessageHistory()

agent_with_chat_history:RunnableWithMessageHistory  = RunnableWithMessageHistory(
    agent_exicuter,
    lambda session_id: message_history,
    input_messages_key= "input", 
    history_messages_key= "chat_history",
)


while True:
    agent_with_chat_history.invoke(
        {"input": input("How can I help you today?  ")},
        config={"configurable": {"session_id":"test123"}},
    )

