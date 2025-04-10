from dataclasses import dataclass, field
import torch


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

