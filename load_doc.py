import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from document import Document
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

class DocumentLoader:

    def __init__(self) -> None:
        self.DOCUMENT_MAP = {
            ".txt": TextLoader,
            ".md": TextLoader,
            ".py": TextLoader,
            ".pdf": PyPDFLoader,
            ".csv": CSVLoader,
            ".xls": UnstructuredExcelLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
    }


    def load_single_document(self, file_path: str) -> Document:
        # Loads a single document from a file path
        file_extension = os.path.splitext(file_path)[1]
        print("LOADING FILE:" + file_extension)
        loader_class = self.DOCUMENT_MAP.get(file_extension)
        if loader_class:
            loader = loader_class(file_path)
        else:
            raise ValueError("Document type is undefined")
        return loader.load()[0]

    def load_document_batch(self, filepaths):
        # create a thread pool
        with ThreadPoolExecutor(max_workers=len(filepaths)) as exe:
            # load files
            futures = [exe.submit(self.load_single_document, name) for name in filepaths]
            # collect data
            data_list = [future.result() for future in futures]
            # return data and file paths
            return (data_list, filepaths)
        
    def load_documents(self, source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory
        all_files = os.listdir(source_dir)
        paths = []
        for file_path in all_files:
            file_extension = os.path.splitext(file_path)[1]
            source_file_path = os.path.join(source_dir, file_path)
            if file_extension in self.DOCUMENT_MAP.keys():
                paths.append(source_file_path)

        # Have at least one worker and at most INGEST_THREADS workers
        n_workers = min(1, max(len(paths), 1))
        chunksize = round(len(paths) / n_workers)
        docs = []
        with ProcessPoolExecutor(n_workers) as executor:
            futures = []
            # split the load operations into chunks
            for i in range(0, len(paths), chunksize):
                # select a chunk of filenames
                filepaths = paths[i : (i + chunksize)]
                # submit the task
                future = executor.submit(self.load_document_batch, filepaths)
                futures.append(future)
            # process all results
            for future in as_completed(futures):
                # open the file and load the data
                contents, _ = future.result()
                docs.extend(contents)

        return docs
    
    def split_documents(self, documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
        text_docs = [doc for doc in documents]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  
        texts = text_splitter.split_documents(text_docs)
        return texts
