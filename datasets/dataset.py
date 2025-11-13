import act
import torch
from torch.utils.data import Dataset

class PretrainedDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.samplers = self.load_samplers()

    def load_data(self):
        samplers = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                samplers.append(act.literal_eval(line))
        return samplers

    def __len__(self):
        return len(self.samplers)

    def __getitem__(self, idx):
        sampler = self.samplers[idx]
        return sampler

    @staticmethod
    def collate_fn(self, batch, tokenizer, max_len=512):
        batch_size = len(batch)
        text_tensors = []
        mask_tensors = []
        prey = torch.ones(batch_size, max_len)
        for index, text in enumerate(batch):
            txt = text["text"]
            txt_tensor, mask_txt = tokenizer.tokenize(txt, max_len=max_len, padding=True, return_tensors="pt")
            text_tensors.append(txt_tensor[:-1])
            mask_tensors.append(mask_txt[1:])
            prey[index] = (text_tensors!= tokenizer.eos_token_id)
            prey[index] = prey[index][1:]

        text_tensors = torch.cat(text_tensors)
        mask_tensors = torch.cat(mask_tensors)
        return text_tensors, prey, mask_tensors


