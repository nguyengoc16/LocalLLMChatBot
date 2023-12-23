import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
class Model:

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.llm = None
        self.template = """[INST] <<SYS>>
                        You are a AI chatbot that use given source to answer question. You must use the information from the given source.\
                        If you cannot find the answer from the source,\
                        you must say that you don't know, don't try to make up an answer.
                        <</SYS>>

                        {context}

                        {history}
                        Question: {question}
                        Helpful Answer:
                        [/INST]"""

    def __load_model(self, name = None):
        if name == None:
            model_name = 'TheBloke/Llama-2-7b-Chat-GPTQ'

        else:
            model_name = name
        print("LOADING TOKENIZER")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("LOADING MODEL")
        self.model = AutoGPTQForCausalLM.from_quantized(model_name,
                                                # revision="gptq-4bit-32g-actorder_True",
                                                # model_basename=model_basename,
                                                use_safetensors=True,
                                                trust_remote_code=False,
                                                # device="cuda:0",
                                                quantize_config=None)
        
    
    def __init_pipeline(self):
        print("INITIALIZING PIPELINE")
        self.pipe = pipeline("text-generation",
                  model=self.model,
                  tokenizer= self.tokenizer,
                  torch_dtype=torch.bfloat16,
                  device_map="auto",
                  max_new_tokens = 1024,
                  do_sample=True,
                  top_k=30,
                  num_return_sequences=1,
                  eos_token_id=self.tokenizer.eos_token_id
                  )
        
    def call_model(self, model_name= None, temperature = 0.5,):
        self.__load_model()
        self.__init_pipeline()
        self.llm = HuggingFacePipeline(pipeline = self.pipe, model_kwargs = {'temperature':temperature})
        prompt = PromptTemplate(input_variables=["history", "context", "question"], template=self.template)
        memory = ConversationBufferWindowMemory(k=3,input_key="question", memory_key="history")


        # qa = RetrievalQA.from_chain_type(
        #                                 llm=self.llm,
        #                                 chain_type="stuff",
        #                                 retriever=db_retrieve,
        #                                 return_source_documents=True,
        #                                 chain_type_kwargs={"prompt": prompt, "memory": memory},
        #                                 verbose = True
        #                             )
        return self.llm, prompt, memory
     

        