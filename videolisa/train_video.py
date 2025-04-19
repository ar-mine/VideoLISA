import torch
import torch.distributed as dist
import os
import wandb
from transformers import AutoProcessor, AutoTokenizer, TrainingArguments
from model.VideoLISA import VideoLISA, LISATrainer
from dataset.video_reson_dataset import VideoDataset
from dataset.dataset import DataCollatorForLISA
from trl import TrlParser
from peft import LoraConfig, TaskType, get_peft_model
from utils import ModelArguments, ScriptArguments

global rank

# TODO: Make this iterable datasets compatible with multi batch size
def main(training_args, model_args, script_args):
    ## Step 1: Initialization
    # Initialize Distributed Training
    global rank
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        enable_parallel = True
    else:
        rank = 0
        enable_parallel = False
    # Initialize wandb
    if rank == 0:
        wandb.init(mode="offline", project="videolisa")
    # Initialize Backbone
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
    if script_args.resume:
        model_params = torch.load(os.path.join(training_args.output_dir, "video.pt"))
        model.load_state_dict(model_params)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)\
    # TODO: Enable fast processor
    processor = AutoProcessor.from_pretrained(model_path)
    model.enable_input_require_grads()  # executed when the gradient checkpoint is turned on
    # Initialize Lora
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

    ## Step 2: Load dataset
    train_dataset = VideoDataset(base_data_dir=script_args.data_root,
                                 max_frames=script_args.max_frames,
                                 processor=processor, tokenizer=tokenizer)

    ## Step 3: Config Trainer and Execute training
    trainer = LISATrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLISA(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    ## Step 4: Postprocess the Training
    if enable_parallel:
        torch.distributed.barrier()
    # Save the merged model
    if rank == 0:
        model = peft_model.merge_and_unload()
        if script_args.resume:
            torch.save(model.state_dict(),
                       os.path.join(training_args.output_dir, "video-r.pt"))
        else:
            torch.save(model.state_dict(),
                       os.path.join(training_args.output_dir, "video.pt"))

if __name__ == "__main__":
    parser = TrlParser((TrainingArguments, ModelArguments, ScriptArguments))
    training_args, model_args, script_args = parser.parse_args_and_config()
    main(training_args, model_args, script_args)
