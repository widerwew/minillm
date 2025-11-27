import os
import sys
sys.path[0] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from datasets.dataset import PretrainedDataset
from cores.minillm import MiniLLMForCasualModel, MiniLLMConfig

import torch
from torch.utils.data import DataLoader

data_path = sys.argv[1]
tokenizer = None
dataset = PretrainedDataset(data_path)

minillm_config = MiniLLMConfig("./configs/minillm_config.json")
minillm_model = MiniLLMForCasualModel(minillm_config)
optimizer = torch.optim.Adam(minillm_model.parameters(), lr=minillm_config.learning_rate)

pretrain_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: dataset.collate_fn(x, tokenizer))
for X, Y, mask_loss in pretrain_dataloader:
    output = minillm_model(X, labels=Y, mask_loss=mask_loss)
    loss = output.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()