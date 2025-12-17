from .dataset import PretrainedDataset, SFTDataset
register_dataset = {
    "PretrainedDataset": PretrainedDataset,
    "SFTDataset": SFTDataset,
}
