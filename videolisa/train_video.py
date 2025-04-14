import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
from transformers import AutoProcessor, AutoTokenizer
from transformers import (
    TrainingArguments,
)
from model.VideoLISA import VideoLISA, LISATrainer
from dataset.video_reson_dataset import VideoDataset
from dataset.dataset import DataCollatorForLISA
from trl import TrlParser
from peft import LoraConfig, TaskType, get_peft_model
from utils import ModelArguments, ScriptArguments, find_linear_layers, predict

global rank

# TODO: Make this iterable datasets compatible with multi batch size
def main(training_args, model_args, script_args):
    # 确保分布式环境已初始化
    global rank
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        enable_parallel = True
    else:
        rank = 0
        enable_parallel = False

    # 配置Backbone
    model_path = model_args.model_name_or_path
    if enable_parallel:
        model = VideoLISA.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    else:
        model = VideoLISA.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path)
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    # 配置LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_args.target_modules,
        inference_mode=False,  # 训练模式
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
    )
    peft_model = get_peft_model(model, config)
    if rank == 0:
        peft_model.print_trainable_parameters()

    train_dataset = VideoDataset(base_data_dir=script_args.data_root,
                                 processor=processor, tokenizer=tokenizer)

    # 配置Trainer
    trainer = LISATrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLISA(tokenizer=tokenizer, padding=True),
    )
    # 开启模型训练
    trainer.train()

if __name__ == "__main__":
    parser = TrlParser((TrainingArguments, ModelArguments, ScriptArguments))
    training_args, model_args, script_args = parser.parse_args_and_config()
    main(training_args, model_args, script_args)
