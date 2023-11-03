from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model_name = 'TheBloke/Llama-2-7b-Chat-GPTQ'
model = AutoGPTQForCausalLM.from_quantized(model_name,
                                          # revision="gptq-4bit-32g-actorder_True",
                                          # model_basename=model_basename,
                                          use_safetensors=True,
                                          trust_remote_code=False,
                                          device="cuda:0",
                                          quantize_config=None)