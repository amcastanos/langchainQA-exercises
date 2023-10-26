from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from llm_model import get_model

load_dotenv()

file = f'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file, encoding='UTF-8')

docs = loader.load()

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(
    docs,
    embeddings
)
retriever = db.as_retriever()

llm_model = get_model()
llm = ChatOpenAI(temperature=0.0, model=llm_model)

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

query = "Please list all your \
shirts with sun protection in a table and summarize each one."

response = qa_stuff.run(query)
print(response)
