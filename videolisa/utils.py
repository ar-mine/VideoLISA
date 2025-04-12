from dataclasses import dataclass, field
import torch
from qwen_vl_utils import process_vision_info
import numpy as np
import cv2

SHORT_QUESTION_LIST = [
    "Can you segment the object in this image?",
    "Please segment the object in this image.",
    "What is object in this image? Please respond with segmentation mask."
]

ANSWER_LIST = [
    "It is <seg>.",
    "Sure, <seg>.",
    "Sure, it is <seg>.",
    "Sure, the segmentation result is <seg>.",
    "<seg>.",
]

@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    attn_implementation: str = field()

    target_modules: list[str] = field()
    lora_r: int = field()
    lora_alpha: int = field()
    lora_dropout: float = field()

@dataclass
class ScriptArguments:
    max_frames: int = field(
        metadata={'help': 'Max number of frames to process'},
    )
    data_root: str = field()
    sam_model_path: str = field()
    train_dataset_path: str = field()
    val_dataset_type: str = field()
    val_dataset_path: str = field()

def find_linear_layers(model, lora_target_modules, excluded_prefix):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
                isinstance(module, cls)
                and all([x not in name for x in excluded_prefix]
        )
                and any([x in name for x in lora_target_modules])
        ):
            lora_module_names.add(name)
    return sorted(list(lora_module_names))

def predict_seg(messages, model, processor):
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
    if len(pred_masks) > 0:
        image_np = np.array(image_inputs[0])
        # TODO: Change bool method
        pred_mask = pred_masks[0][0] > 0
        pred_mask = pred_mask.cpu().numpy()
        highlight = np.zeros_like(image_np, dtype=np.uint8)
        highlight[pred_mask] = (255, 0, 0)
        # 将高亮遮罩与原图叠加
        highlighted_image = cv2.addWeighted(image_np, 0.5, highlight, 0.5, 0)
    else:
        highlighted_image = None
    return output_text[0], highlighted_image