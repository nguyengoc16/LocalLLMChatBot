
from load_doc import DocumentLoader
from embedding import Embedding
from model import Model
from langchain.chains import RetrievalQA
import gradio as gr


def get_ready():
    #load document
    doc_loader = DocumentLoader()
    doc = doc_loader.load_documents(source_dir='PDF')
    text_documents = doc_loader.split_documents(doc)

    #embbeding_DB
    embedding_load = Embedding()
    embedding_load.load_db(text_documents=text_documents)
    retriever = embedding_load.retrieve_db()
    #load model
    model_load = Model()
    llm, prompt, memory = model_load.call_model()

    return llm, prompt, memory, retriever

def load_qa(query):
  llm, prompt, memory, retriever= get_ready()
  qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
        verbose = True)
  res = qa(query)
  return res["result"], res["source_documents"]


def chatbot(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = load_qa(inp)[0]
    history.append((input, output))
    return history, history

if __name__ == "__main__":
    

    block = gr.Blocks()


    with block:
        gr.Markdown("""<h1><center>Hi I'm your friendly neighborhood!!</center></h1>
        """)
        chatbot = gr.Chatbot()
        message = gr.Textbox(placeholder="Enter your question hear")
        state = gr.State()
        submit = gr.Button("SEND")
        submit.click(chatbot, inputs=[message, state], outputs=[chatbot, state])

    block.launch(debug = True)