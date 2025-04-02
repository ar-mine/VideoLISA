from dataclasses import dataclass, field


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
    train_dataset_path: str = field()
    val_dataset_type: str = field()
    val_dataset_path: str = field()

