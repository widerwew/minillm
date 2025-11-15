import ast
import torch
from torch.utils.data import Dataset

class PretrainedDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
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

    @staticmethod
    def collate_fn(batch, tokenizer):
        input_txts = []
        for txt in batch:
            input_txts.append(txt["text"])

        token_result = tokenizer(input_txts, padding=True, truncation=True, return_tensors="pt")
        input_ids = token_result["input_ids"]
        X = input_ids[:, :-1]
        Y = input_ids[:, 1:]
        mask_loss = (X!=tokenizer.eos_token_id)
        return X, Y, mask_loss