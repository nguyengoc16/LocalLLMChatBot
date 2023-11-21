from langchain.embeddings import HuggingFaceInstructEmbeddings,HuggingFaceEmbeddings
from chromadb.config import Settings
from langchain.vectorstores import Chroma

class Embedding:

    def __init__(self) -> None:
    
        self.PERSIST_DIRECTORY = "/DB"
        self.CHROMA_SETTINGS = Settings(
                                        chroma_db_impl="duckdb+parquet",
                                        persist_directory=self.PERSIST_DIRECTORY, 
                                        anonymized_telemetry=False,)

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        

    def embed_doc(self, text_documents):
        self.db = Chroma.from_documents(
                                    text_documents,
                                    self.embeddings,
                                    persist_directory=self.PERSIST_DIRECTORY,
                                    # client_settings=CHROMA_SETTINGS,
                                    )
        self.db.persist()
        return self.db

    def retrieve_db(self):
        return self.db

