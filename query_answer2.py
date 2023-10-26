from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from llm_model import get_model

load_dotenv()

file = f'C:\\Users\\user\\Documents\\python\\AI\\Langchain\\OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file, encoding='UTF-8')

docs = loader.load()

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(
    docs,
    embeddings
)

query = "Sugierame una camisa con protección solar por favor"

ansDocs = db.similarity_search(query)
print(len(ansDocs))
print(ansDocs[0])

retriever = db.as_retriever()
llm_model = get_model()
llm = ChatOpenAI(temperature=0.0, model=llm_model)

qdocs = "".join([ansDocs[i].page_content for i in range(len(ansDocs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table and summarize each one.")

print(response)
