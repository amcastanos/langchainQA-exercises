from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

file = f'C:\\Users\\user\\Documents\\python\\AI\\Langchain\\OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file, encoding='UTF-8')

docs = loader.load()

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(
    docs,
    embeddings
)
retriever = db.as_retriever()

llm_model = "gpt-3.5-turbo-0301"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

qa_stuff = RetrievalQA.from_chain_type()


llm_model = "gpt-3.5-turbo-0301"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

qdocs = "".join([ansDocs[i].page_content for i in range(len(ansDocs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table and summarize each one.")

print(response)
