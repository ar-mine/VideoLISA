import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from transformers import (
    TrainingArguments,
    TrainerCallback,
    Trainer,
)
from model.VideoLISA import VideoLISA
from dataset.sem_seg_dataset import SemSegDataset
from dataset.dataset import DataCollatorForLISA
from trl import TrlParser
from peft import LoraConfig, TaskType, get_peft_model
from utils import ModelArguments, ScriptArguments


# On server dataset is organized without folders
LOCAL = True


class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        在每次日志记录时调用
        """
        if logs is not None:
            # 默认日志输出
            print(f"Step: {state.global_step}, Logs: {logs}")
            # 添加自定义参数（假设 ce_loss 已包含在 logs 中）
            if "ce_loss" in logs:
                print(f"Custom - ce_loss: {logs['ce_loss']}")


# TODO: Make this iterable datasets compatible with multi batch size
def main(training_args, model_args, script_args):
    # 配置Backbone
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

    model = VideoLISA.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.init_sam_module(model_path="/media/automan/ExSpace/Projects/VideoLISA/checkpoints/sam_vit_h_4b8939.pth")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.add_tokens("<seg>", special_tokens=False)
    model.seg_token_idx = tokenizer.convert_tokens_to_ids("<seg>")
    model.resize_token_embeddings(len(tokenizer))
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
    peft_model.base_model.lm_head.weight.requires_grad = True
    peft_model.base_model.model.model.embed_tokens.weight.requires_grad = True
    text_hidden_fcs_params = {}
    for name, param in peft_model.base_model.text_hidden_fcs.named_parameters():
        param.requires_grad = True
        text_hidden_fcs_params[name] = param

    train_dataset = SemSegDataset(base_image_dir=script_args.data_root,
                                  processor=processor, tokenizer=tokenizer)
    # 配置Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLISA(tokenizer=tokenizer, padding=True),
        callbacks=[CustomLoggingCallback()],
    )
    # 开启模型训练
    trainer.train()
    torch.save(text_hidden_fcs_params, script_args.save_path)


if __name__ == "__main__":
    parser = TrlParser((TrainingArguments, ModelArguments, ScriptArguments))
    training_args, model_args, script_args = parser.parse_args_and_config()
    main(training_args, model_args, script_args)
