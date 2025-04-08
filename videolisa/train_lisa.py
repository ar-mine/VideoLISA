import torch
import os
import cv2
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from transformers import (
    TrainingArguments,
    TrainerCallback,
)
from model.VideoLISA import VideoLISA, LISATrainer
from dataset.sem_seg_dataset import SemSegDataset
from dataset.dataset import DataCollatorForLISA
from trl import TrlParser
from peft import LoraConfig, TaskType, get_peft_model
from utils import ModelArguments, ScriptArguments, find_linear_layers
from qwen_vl_utils import process_vision_info



# On server dataset is organized without folders
LOCAL = True



def predict(messages, model, processor):
    # 准备推理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids, pred_masks = model.generate(original_images=image_inputs, **inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    # )
    image_np = np.array(image_inputs[0])
    pred_mask = pred_masks[0][0].to(bool).cpu().numpy()
    highlight = np.zeros_like(image_np, dtype=np.uint8)
    highlight[pred_mask] = (255, 0, 0)
    # 将高亮遮罩与原图叠加
    highlighted_image = cv2.addWeighted(image_np, 0.5, highlight, 0.5, 0)
    return output_text[0], highlighted_image


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
    peft_model.base_model.model.model.embed_tokens.weight.requires_grad = True
    text_hidden_fcs_params = {}
    for name, param in peft_model.base_model.text_hidden_fcs.named_parameters():
        param.requires_grad = True
        text_hidden_fcs_params[name] = param
    peft_model.print_trainable_parameters()

    train_dataset = SemSegDataset(base_image_dir=script_args.data_root,
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
    torch.save(text_hidden_fcs_params,
               os.path.join(training_args.output_dir, "text_hidden_fcs_params.pt"))

    # Eval
    origin_image_path = "/media/automan/6E94666294662CB1/A_Content/Datasets/ADEChallengeData2016/images/train/ADE_train_00000001.jpg"
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": origin_image_path
            },
            {
                "type": "text",
                "text": "Can you segment the floor in this image?"
            }
        ]}]

    response, image = predict(messages, peft_model, processor)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("output/image.png", image.astype(np.uint8))
    messages.append({"role": "assistant", "content": f"{response}"})
    print(messages[-1])


if __name__ == "__main__":
    parser = TrlParser((TrainingArguments, ModelArguments, ScriptArguments))
    training_args, model_args, script_args = parser.parse_args_and_config()
    main(training_args, model_args, script_args)
