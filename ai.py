import streamlit as st
from langchain.llms import OpenAI

# used to load text
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import TextLoader

# used to create the retriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# used to create the retrieval tool
from langchain.agents import tool

# used to create the memory
from langchain.memory import ConversationBufferMemory

# used to create the prompt template
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder

# used to create the agent executor
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor

loader = TextLoader('data.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
openai_api_key = st.secrets["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

@tool
def tool(query):
    "gives detail about the shipping company Laparkan"
    docs = retriever.get_relevant_documents(query)
    return docs

tools = [tool]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_message = SystemMessage(
        content=(
            "You are a customer service agent for Laparkan."
            "Your job is to answer questions about the company "
            "and its shipping services. "
            "Greet the customer and ask them how you can help."
            "If they ask a question you don't know the answer to, "
            "say i don't know and move on to the next question."
            "Feel free to use any tools available to look up "
            "relevant information, only if neccessary"
        )
)

prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")]
    )

llm = ChatOpenAI(temperature = 0, openai_api_key=openai_api_key)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)


st.title('Laparkan QA Customer Support bot')

def generate_response(input_text):
  
  st.info(agent_executor({"input": input_text})["output"])


with st.form('my_form'):
  text = st.text_area('Enter text:', 'What is the cost of shipping a barrel to Guyana?')
  submitted = st.form_submit_button('Submit')

  if submitted:
    generate_response(text)