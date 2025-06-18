from dataclasses import dataclass, field
import torch
from qwen_vl_utils import process_vision_info
import numpy as np
import cv2
from typing import Optional

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

SSV2_QUESTION_LIST = [
    "Output the action shown in the video with its interacting objects in JSON format and it should contains 'action' and 'objects' as keys.",
    "Generate the action depicted in the video along with its interacting objects in JSON format, including 'action' and 'objects' as keys.",
    "Produce the action shown in the video and its related objects in JSON format, with 'action' and 'objects' as keys."
    "Output the action from the video and the objects involved in JSON format, containing 'action' and 'objects' as keys."
    "Create a JSON representation of the action in the video and its interacting objects, using 'action' and 'objects' as keys."
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
    data_root: str = field()
    datasets: str = field()
    sam_model_path: Optional[str] = field(default=None)
    train_dataset_path: str = field(default="")
    val_dataset_type: str = field(default="")
    val_dataset_path: str = field(default="")
    resume: bool = field(default=False)
    max_frames: int = field(
        metadata={'help': 'Max number of frames to process'},
        default=0
    )

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

def predict(messages, model, processor):
    # 准备推理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, fps = process_vision_info(messages, return_video_kwargs=True)
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
    return output_text[0], highlighted_image, fps

def calculate_iou(start_A, end_A, start_B, end_B):
    """
    Calculate the Intersection over Union (IoU) for two time intervals.

    Parameters:
    start_A (float): Start time of the first interval.
    end_A (float): End time of the first interval.
    start_B (float): Start time of the second interval.
    end_B (float): End time of the second interval.

    Returns:
    float: The IoU value for the two intervals.
    """
    # Ensure that start < end for each interval
    if start_A >= end_A or start_B >= end_B:
        raise ValueError("Invalid intervals: start time must be less than end time.")

    # Calculate intersection
    intersection_start = max(start_A, start_B)
    intersection_end = min(end_A, end_B)
    intersection = max(0, intersection_end - intersection_start)

    # Calculate union
    union = (end_A - start_A) + (end_B - start_B) - intersection

    # Calculate IoU
    if union == 0:
        return 0.0  # If both intervals have zero length, IoU is 0
    else:
        return intersection / union

def init_segmentation(model, tokenizer, processor, config):
    pass