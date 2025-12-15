import os
import sys
import time
from os import truncate

import torch

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入目标模型
from cores.minillm import MiniLLMForCasualModel, MiniLLMConfig
from transformers import AutoTokenizer, TextStreamer

class Inference:
    def __init__(self, model_path="../output", device="cuda:0"):
        self.device = device
        self.model_path = model_path

        print(f"Loading model from {self.model_path} ...")
        config = MiniLLMConfig(config_path=f"{model_path}/config.json")
        self.model = MiniLLMForCasualModel(config)

        print(f"Loading model weight from {self.model_path} ...")
        state_dict = torch.load(f"{self.model_path}/pytorch_model.bin", map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Loaded model from {self.model_path} done.")

    def generate(self, prompt: str, max_new_token=200, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.2) -> str:
        start = time.time()
        prompt = self.tokenizer.bos_token + prompt
        inputs = self.tokenizer(prompt, return_tensor="pt", truncation=True, max_length=max_new_token+len(prompt), add_special_tokens=False).to(self.device)

        generated_ids = self.model.generate(inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=max_new_token, do_sample=True, temperature=temperature, repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k, return_dict_in_generate=True, use_cache=True, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        token_ids = generated_ids[0].squeeze().tolist()
        response = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        end = time.time() - start

        # 计算速度
        tokens_generated = generated_ids[0].shape[-1] - inputs["input_ids"].shape[1]
        speed = tokens_generated / end if end > 0 else 0

        return {"text": response, "time":end, "speed": speed}

if __name__ == "__main__":
    inference = Inference(model_path="../output/pretrain")

    prompt = [
        "请告诉我哈利波特的作者是谁?",
        "请给我介绍一本书",
        "请介绍一部好电影",
        "世界最高的山峰是",
        "解释下什么是机器学习",
        "推荐一些中国的美食"
    ]

    for prompt in prompt:
        result = inference.generate(prompt, max_new_token=500)
        print(f"Input: {prompt}")
        print(f"Output: {result['text']}")
        print(f"Speed: {result['speed']}")

