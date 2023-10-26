from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator

load_dotenv()

file = f'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file, encoding='UTF-8')

index_creator = VectorstoreIndexCreator()
index = index_creator.from_loaders([loader])
# vectorstore_cls=DocArrayInMemorySearch,


query = "Listeme todas las camisas con protección solar"
# query = "Please list all your shirts with sun protection \
# in a table and summarize each one."

response = index.query(query)
print(response)
