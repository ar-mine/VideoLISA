import numpy as np
import torch
import itertools
from typing import Union, Optional, Any
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq
from transformers.utils import PaddingStrategy
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.processing_utils import ProcessorMixin
from qwen_vl_utils import process_vision_info


@dataclass
class DataCollatorForLISA(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # reconvert list[None] to None if necessary
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None

        # Videos Concat
        videos = list(
            itertools.chain(
                *(
                    feature["pixel_values_videos"]
                    for feature in features
                    if "pixel_values_videos" in feature
                )
            )
        )
        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        feature["video_grid_thw"]
                        for feature in features
                        if "video_grid_thw" in feature
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        special_keys = ["original_images", "gt_masks", "video_grid_thw", "pixel_values_videos"]
        special_features = {}
        for s_key in special_keys:
            special_features[s_key] = [feature[s_key] for feature in features]
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name and k not in special_keys} for feature in features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        batch.data.update(special_features)

        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                            (max_label_length + self.pad_to_multiple_of - 1)
                            // self.pad_to_multiple_of
                            * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                # batch["labels"] = torch.cat(batch["labels"]).to(dtype=torch.int64)
                batch["labels"] = torch.tensor(np.array(batch["labels"]), dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        # Prepare batch
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch

@dataclass
class DataCollatorForQwen(DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    processor: ProcessorMixin = None
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if self.processor is None:
            raise RuntimeError("processor must be initialized before calling this method")
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Pre-process for Qwen input
        pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], []
        images, gt_masks, new_features = [], [], []
        for (image, question, answer, masks) in features:
            text = self.processor.apply_chat_template(
                question, tokenize=False, add_generation_prompt=False
            )
            # return_video_kwargs=True will return 3 values
            image_input, video_input = process_vision_info(question)
            inputs = self.processor(
                text=[text],
                images=image_input,
                videos=video_input,
                padding=True,
                return_tensors="pt",
            )
            response = self.tokenizer(f"{answer}", add_special_tokens=False)

            inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,为了方便拼接
            instruction = inputs

            input_ids = (
                    instruction["input_ids"][0] + response["input_ids"] + [self.tokenizer.pad_token_id]
            )
            attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
            label = (
                    [-100] * len(instruction["input_ids"][0])
                    + response["input_ids"]
                    + [self.tokenizer.pad_token_id]
            )
            # if len(input_ids) > MAX_LENGTH:  # 做一个截断
            #     input_ids = input_ids[:MAX_LENGTH]
            #     attention_mask = attention_mask[:MAX_LENGTH]
            #     labels = labels[:MAX_LENGTH]

            images.append(image)
            gt_masks.append(masks)
            if "pixel_values" in inputs.keys():
                pixel_values.append(torch.tensor(inputs["pixel_values"]))
                image_grid_thw.append(torch.tensor(inputs["image_grid_thw"]).squeeze(0))
            if "pixel_values_video" in inputs.keys():
                pixel_values_videos.append(torch.tensor(inputs["pixel_values_video"]))
                video_grid_thw.append(torch.tensor(inputs["video_grid_thw"]).squeeze(0))
            new_features.append({"input_ids": input_ids,
                                 "attention_mask": attention_mask,
                                 "label": label,
                                 })

        # Images Concat
        # TODO: check whether to use cat or stack
        if len(pixel_values) != 0:
            concat_images = torch.cat([image_pixel for image_pixel in pixel_values], dim=0)
            image_grid_thw = torch.stack(image_grid_thw, dim=0)
        else:
            concat_images = None
            image_grid_thw = None

        # Videos Concat
        if len(pixel_values_videos) != 0:
            concat_videos = torch.cat([video for video in pixel_values_videos], dim=0)
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        # concat_masks = torch.cat([mask for mask in masks], dim=0)

        visual_features = {
            "pixel_values": concat_images,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": concat_videos,
            "video_grid_thw": video_grid_thw,
        }
        # non_labels_features = [{
        #     "input_ids": input_id,
        #     "attention_masks": attention_mask,
        #     "labels": label,
        # } for input_id, attention_mask, label in zip(input_ids, attention_masks, labels)]

        label_name = "label" if "label" in new_features[0].keys() else "labels"
        labels = [feature[label_name] for feature in new_features] if label_name in new_features[0].keys() else None
        # reconvert list[None] to None if necessary
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None


        special_keys = ["images", "gt_masks"]
        special_features = {
            "images": images,
            "gt_masks": gt_masks,
        }
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name and k not in special_keys} for feature in new_features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        batch.data.update(special_features)
        batch.data.update(visual_features)

        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(new_features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                            (max_label_length + self.pad_to_multiple_of - 1)
                            // self.pad_to_multiple_of
                            * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                if isinstance(new_features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                # batch["labels"] = torch.cat(batch["labels"]).to(dtype=torch.int64)
                batch["labels"] = torch.tensor(np.array(batch["labels"]), dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        return batch
