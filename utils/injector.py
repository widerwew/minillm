import os
import math
import torch
import logging
import torch.distributed as dist
from datetime import timedelta

def init_trainer_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_TIMEOUT"] = "1800"
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=30))
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def cleanup_dist_model():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return

def init_logger(rank, log_file="train.log"):
    if rank != 0:return None
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - Rank %(rank)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    class RankFilter(logging.Filter):
        def __init__(self, rank):
            self.rank = rank
        def filter(self, record):
            record.rank = self.rank
            return True

    rank_filter = RankFilter(rank)
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(rank_filter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)
    console.addFilter(rank_filter)
    logger.addHandler(console)
    return logger


def get_lr(current_iter, total_iters, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(current_iter / total_iters * math.pi))

