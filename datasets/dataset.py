import ast
from distutils.command.register import register

import torch
from torch.utils.data import Dataset

class PretrainedDataset(Dataset):
    def __init__(self, data_path, tokenizer=None, max_length=8192):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samplers = self.load_samplers()

    def load_samplers(self):
        samplers = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                samplers.append(ast.literal_eval(line))
        return samplers

    def __len__(self):
        return len(self.samplers)

    def __getitem__(self, idx):
        sampler = self.samplers[idx]
        return sampler

    def collate_fn(self, batch):
        input_txts = []
        for txt in batch:
            input_txts.append(txt["text"])

        token_result = self.tokenizer(input_txts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = token_result["input_ids"]
        X = input_ids[:, :-1]
        Y = input_ids[:, 1:]
        mask_loss = (X!=self.tokenizer.pad_token_id)
        return X, Y, mask_loss

class SFTDataset(PretrainedDataset):
    def __init__(self, data_path, tokenizer=None, max_length=8192):
        super().__init__(data_path, tokenizer=tokenizer, max_length=max_length)
        # 生成开始标志
        self.bos_token = self.tokenizer(f"{tokenizer.bos_token}assistant", add_special_tokens=False)["input_ids"]
        self.bos_token_length = len(self.bos_token)
        self.eos_token = self.tokenizer(f"{tokenizer.eos_token}", add_special_tokens=False)["input_ids"][0]

    def _chat_prompt_template(self, conversation):
        if conversation[0].get("role") == 'system' and conversation[0].get("functions"):
            tools = conversation[0].get("functions")
        else:
            tools = None
        return self.tokenizer.apply_chat_template(conversation, tokenize=False, tools=tools)

    def _generate_mask_loss(self, input_ids):
        # 处理loss
        mask_loss = torch.zeros_like(input_ids)
        for i in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                if mask_loss[i][j] == 1:
                    continue
                # find generate text
                start = -1
                if input_ids[i][j:j + self.bos_token_length].tolist() == self.bos_token:
                    start = j + self.bos_token_length
                if start == -1:
                    continue
                mask_loss_str = ""
                for k in range(start, input_ids.shape[1]):
                    mask_loss_str += self.tokenizer.decode(input_ids[i][k].tolist())
                    mask_loss[i][k] = 1
                    if input_ids[i][k] == self.eos_token:
                        break
        return mask_loss

    def __getitem__(self, idx):
        sampler = self.samplers[idx]
        conversation_prompt = self._chat_prompt_template(sampler["conversations"])
        return conversation_prompt

    def collate_fn(self, batch):
        input_ids = self.tokenizer(batch, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt")["input_ids"]
        mask_loss = self._generate_mask_loss(input_ids)
        X = input_ids[:, :-1]
        Y = input_ids[:, 1:]
        mask_loss = mask_loss[:, 1:]
        return X, Y, mask_loss