import torch
import torch.distributed as dist
import os
from transformers import AutoProcessor, AutoTokenizer, AutoConfig, TrainingArguments
from model.VideoLISA import VideoLISA, LISATrainer
from dataset.refer_seg_dataset import ReferSegDataset
from dataset.dataset import DataCollatorForQwen
from trl import TrlParser
from peft import LoraConfig, TaskType, get_peft_model
from utils import ModelArguments, ScriptArguments, find_linear_layers
import wandb

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
        wandb.init(mode="offline", project="pretrain-refseg")
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
    torch.set_default_dtype(torch.bfloat16)
    model.enable_segmentation = True
    model.init_sam_module(model_path=script_args.sam_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    tokenizer.add_tokens("<seg>", special_tokens=False)
    config = AutoConfig.from_pretrained(model_path)
    model.seg_token_idx = tokenizer.convert_tokens_to_ids("<seg>")
    new_tokenizer_vocab_size = len(tokenizer)
    if config.vocab_size < new_tokenizer_vocab_size:
        print("Resizing model embeddings to match new tokenizer vocab size...")
        model.resize_token_embeddings(new_tokenizer_vocab_size)
        # 更新 config 中的 vocab_size
        config.vocab_size = new_tokenizer_vocab_size
    else:
        print("Model vocab size is already sufficient, no resizing needed.")
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer = tokenizer
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    # 获取模型的输入嵌入层
    embedding_layer = model.get_input_embeddings()
    # 选择参考token：使用start_header_id token
    reference_token_id = tokenizer.convert_tokens_to_ids("<|file_sep|>")
    # 将参考token的嵌入权重复制到新token
    for token in ["<seg>"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        embedding_layer.weight.data[token_id] = embedding_layer.weight.data[reference_token_id].clone()

    # 配置LoRA
    lora_layers = find_linear_layers(model, model_args.target_modules, ["sam", "text_hidden_fcs"])
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_layers,
        inference_mode=False,  # 训练模式
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
    )
    peft_model = get_peft_model(model, config)
    peft_model.base_model.lm_head.weight.requires_grad = True
    peft_model.base_model.model.model.language_model.embed_tokens.weight.requires_grad = True
    for param in peft_model.base_model.model.sam.sam_prompt_encoder.project_text.parameters():
        param.requires_grad = True
    for param in peft_model.base_model.model.sam.sam_mask_decoder.parameters():
        param.requires_grad = True
    if rank == 0:
        peft_model.print_trainable_parameters()

    train_dataset = ReferSegDataset(base_image_dir=script_args.data_root,
                                    refer_seg_data=script_args.datasets,
                                    image_size=1024)
    # 配置Trainer
    trainer = LISATrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForQwen(tokenizer=tokenizer, padding=True, processor=processor),
    )
    # 开启模型训练
    trainer.train()

    # 训练结束
    if enable_parallel:
        torch.distributed.barrier()
    if rank == 0:
        model = peft_model.merge_and_unload()
        torch.save(model.state_dict(),
                   os.path.join(training_args.output_dir, "video-lisa.pt"))
        # Eval
        # with torch.no_grad():
        #     origin_image_path = os.path.join(script_args.data_root["ade20k"], "images/training/ADE_train_00000001.jpg")
        #     messages = [{
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image",
        #                 "image": origin_image_path
        #             },
        #             {
        #                 "type": "text",
        #                 "text": "Can you segment the floor in this image?"
        #             }
        #         ]}]
        #
        #     response, image = predict(messages, model, processor)
        #     if image is not None:
        #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #         cv2.imwrite("output/image.png", image.astype(np.uint8))
        #     messages.append({"role": "assistant", "content": f"{response}"})
        #     print(messages[-1])


if __name__ == "__main__":
    parser = TrlParser((TrainingArguments, ModelArguments, ScriptArguments))
    training_args, model_args, script_args = parser.parse_args_and_config()
    main(training_args, model_args, script_args)
