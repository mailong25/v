import torch
from transformers import pipeline
import os
from prompt_utils import dialog_prompt

class Llama:
    def __init__(self):
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token="",
        )
        self.pipe = pipe
    
    def predict(self, contexts):
        prompt = dialog_prompt(contexts)
        messages = [{"role": "user", "content": prompt}]
        outputs = self.pipe(
            messages,
            max_new_tokens=256,
        )
        return outputs[0]["generated_text"][-1]['content'].strip('"').strip("'")
    
