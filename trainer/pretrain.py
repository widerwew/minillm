import os
import sys
import time

sys.path[0] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from datasets import register_dataset
from cores.minillm import MiniLLMForCasualModel, MiniLLMConfig
from utils.injector import init_trainer_mode, init_logger, get_lr, cleanup_dist_model

import torch
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
import torch.distributed as dist


def init_trainer(rank, world_size, mini_config_path, log_file="train.log", mode='pretrain'):
    #初始化日志
    logger = init_logger(rank, log_file)

    #设置设备
    device = torch.device(f"cuda:{rank}")

    #初始化配置文件
    mini_config = MiniLLMConfig(mini_config_path)

    #初始化模型
    mini_model = MiniLLMForCasualModel(mini_config)
    if mode != 'pretrain':
        assert mini_config.model_path is not None, "You must specify a pretrained model when you not in pretrain mode."
        print(f"Loading model weight from {mini_config.model_path} ...")
        state_dict = torch.load(f"{mini_config.model_path}/pytorch_model.bin", map_location=device, weights_only=True)
        mini_model.model.load_state_dict(state_dict)
    mini_model = mini_model.to(device)

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mini_config.tokenizer_path)

    # 初始化训练数据
    dataset = register_dataset[mini_config.dataset](mini_config.data_path, tokenizer=tokenizer, max_length=mini_config.max_length)

    # dist模式适配
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=mini_config.seed)
        shuffle = False
        mini_model = DDP(mini_model, device_ids=[rank], output_device=rank,  find_unused_parameters=False, broadcast_buffers=False, static_graph=False, gradient_as_bucket_view=True)
    else:
        sampler = None
        shuffle = True

    # 初始化dataloader
    data_loader = DataLoader(dataset, batch_size=mini_config.batch_size, num_workers=world_size, shuffle=shuffle, sampler=sampler, drop_last=True, pin_memory=True, prefetch_factor=2, collate_fn=dataset.collate_fn)

    # 初始化 optimizer
    optimizer = torch.optim.AdamW(mini_model.parameters(), lr=mini_config.learning_rate)
    scaler = GradScaler()
    return logger, mini_config, mini_model, data_loader, tokenizer, optimizer, scaler, device

def trainer(logger, mini_config, mini_model, data_loader, tokenizer, optimizer, scaler, device, rank, world_size):
    epochs = mini_config.num_epochs
    batch_size = mini_config.batch_size
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    # 打印设备模型配置信息
    if rank == 0:
        total_parameter = sum(p.numel() for p in mini_model.parameters() if p.requires_grad)
        logger.info(f"Training {mini_config.model_name} ....")
        logger.info(f"GPU名称: {torch.cuda.get_device_name(rank)}")
        logger.info(f"GPU内存：{torch.cuda.get_device_properties(rank).total_memory / 1e9:.2f}")
        logger.info(f"GPU卡数：{world_size}")
        logger.info(f"RANK ID: {rank}")
        logger.info(f"可训练参数：{total_parameter / 1e6:.2f} 百万")
        logger.info(f"训练 Epochs：{epochs}")
        logger.info(f"训练 Batch_size: {batch_size}")

    try:
        mini_model.train()
        step_per_epoch = len(data_loader)
        total_steps = step_per_epoch * epochs
        if rank == 0:
            start_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()
            for step, (X, Y, mask_loss) in enumerate(data_loader):
                try:
                    X = X.to(device)
                    Y = Y.to(device)
                    mask_loss = mask_loss.to(device)
                    current_step = epoch * step_per_epoch + step
                    lr = get_lr(current_step, total_steps, lr=mini_config.learning_rate)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    optimizer.zero_grad()
                    with autocast(device_type="cuda"):
                        output = mini_model(X)
                        loss = loss_fn(output.logits.view(-1, output.logits.size(-1)), Y.view(-1)).view(Y.size())
                        loss = (loss * mask_loss).sum() / mask_loss.sum()
                        loss += output.aux_loss
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(mini_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    # 打印训练日志
                    if (rank == 0 and step!=0) and (step % mini_config.log_step_interval == 0 or step == total_steps - 1):
                        spent_time = time.time() - start_time
                        average_time_step = spent_time / current_step
                        remaining_time_step = (total_steps - current_step) * average_time_step / 60

                        gem_memory = torch.cuda.memory_allocated() / 1e9
                        gpu_total_memory = torch.cuda.get_device_properties(rank).total_memory / 1e9
                        logger.info(f"Epoch: [{epoch + 1}/{epochs}] [{current_step}/{total_steps}] Loss: {loss.item():.4f} Lr: {lr:.7f} Time/Step: {average_time_step} Est Time: {remaining_time_step:.2f}min GPU: {gem_memory:.2f} / {gpu_total_memory:.2f}")
                except RuntimeError as e:
                    logger.error(f"Step: {step} RuntimeError: {str(e)}")
                    if "out of memory" in str(e).lower():
                        logger.error("❎Out of Memory! You can try reduce your batch size!")
                    raise
            # 进程同步
            if world_size > 1:
                dist.barrier()

            if rank == 0:
                epoch_end_time = (time.time() - epoch_start_time) / 60
                logger.info(f"Epoch Time: {epoch_end_time:.2f} minutes.")

                # 保存模型
                try:
                    mini_model.eval()
                    if not os.path.exists(mini_config.save_path):
                        os.makedirs(mini_config.save_path)
                    model_to_save = mini_model.module if isinstance(mini_model, DDP) else mini_model
                    model_to_save.save_pretrained(mini_config.save_path, safe_serialization=False)
                    tokenizer.save_pretrained(mini_config.save_path)
                    mini_model.train()
                    logger.info(f"Saved model to path: {mini_config.save_path}.")
                except Exception as e:
                    logger.error(f"Failed to save model to path: {mini_config.save_path}. Error: {str(e)}")
                    raise
    except KeyboardInterrupt:
        if rank == 0:
            logger.info("🀄Training interrupted by user.")
    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        raise
    finally:
        cleanup_dist_model()


if __name__ == "__main__":
    mini_config_path = "../configs/minillm_config.json"
    rank, world_size = init_trainer_mode()
    logger, mini_config, mini_model, data_loader, tokenizer, optimizer, scaler, device=init_trainer(rank, world_size, mini_config_path=mini_config_path)
    trainer(logger, mini_config, mini_model, data_loader, tokenizer, optimizer, scaler, device, rank, world_size)

















for X, Y, mask_loss in pretrain_dataloader:
    output = minillm_model(X, labels=Y, mask_loss=mask_loss)
    loss = output.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()