from typing import Optional, List, Union, Tuple
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from transformers.modeling_outputs import ModelOutput
from .segment_anything import build_sam_vit_h
from .segment_anything.utils.transforms import ResizeLongestSide


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        scale: float=1000,
        eps: float=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks:
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


@dataclass
class VideoLISACausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    mask_bce_loss: Optional[torch.FloatTensor] = None
    mask_dice_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class VideoLISA(Qwen2_5_VLForConditionalGeneration):
    # TODO: Update meta info

    def __init__(self, config):
        # TODO: Load SAM model
        self.seg_token_idx = -1
        super().__init__(config)

        self.ce_loss_weight = 1.0
        self.dice_loss_weight = 0.5
        self.bce_loss_weight = 2.0

        self.sam = None
        self.text_hidden_fcs = None

    def init_sam_module(self, model_path):
        # TODO: Apply lora only on LLM
        # SAM Initialization
        self.sam = build_sam_vit_h(model_path)
        self.sam.to(self.device).to(self.dtype)
        # TODO: Load from checkpoint
        # Projection layer (from <seg> hidden states to sam)
        in_dim = 2048
        out_dim = 256
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.to(self.device).to(self.dtype)

    def forward(
            self,
            original_images: Optional[torch.Tensor] = None,
            gt_masks: Optional[torch.Tensor] = None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  position_ids=position_ids,
                                  past_key_values=past_key_values,
                                  inputs_embeds=inputs_embeds,
                                  labels=labels,
                                  use_cache=use_cache,
                                  output_attentions=output_attentions,
                                  return_dict=return_dict,
                                  pixel_values=pixel_values,
                                  pixel_values_videos=pixel_values_videos,
                                  image_grid_thw=image_grid_thw,
                                  video_grid_thw=video_grid_thw,
                                  rope_deltas=rope_deltas,
                                  cache_position=cache_position,
                                  second_per_grid_ts=second_per_grid_ts,
                                  output_hidden_states=True,
                                  )

        if labels is None:
            return outputs

        output_hidden_states = outputs.hidden_states

        seg_token_ids = torch.where(input_ids[:, :] == self.seg_token_idx)

        seg_token_hidden_states = []
        for b, i in zip(*seg_token_ids):
            offset = input_ids.shape[1]-i
            seg_token_hidden_states.append(output_hidden_states[-offset][-1][b].unsqueeze(0))

        seg_token_hidden_states = torch.cat(seg_token_hidden_states, dim=0)
        pred_embeddings = self.text_hidden_fcs[0](seg_token_hidden_states)
        assert pred_embeddings is not None, "Cannot find <seg> token."

        # TODO: image input
        transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        assert pred_embeddings.shape[0] == len(original_images), "Prediction size mismatch image number"
        features, input_sizes, original_sizes = [], [], []
        for idx, image in enumerate(original_images):
            image_np = image
            input_image = transform.apply_image(image_np)
            input_image_torch = torch.as_tensor(input_image, device=self.device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

            transformed_image = input_image_torch
            original_image_size = image_np.shape[:2]
            assert (
                    len(transformed_image.shape) == 4
                    and transformed_image.shape[1] == 3
                    and max(*transformed_image.shape[2:]) == self.sam.image_encoder.img_size
            ), f"set_torch_image input must be BCHW with long side {self.sam.image_encoder.img_size}."

            original_size = original_image_size
            input_size = tuple(transformed_image.shape[-2:])
            input_image = self.sam.preprocess(transformed_image)
            feature = self.sam.image_encoder(input_image)

            features.append(feature)
            input_sizes.append(input_size)
            original_sizes.append(original_size)

        # TODO: sam process, parallel
        pred_masks = []
        for i in range(pred_embeddings.shape[0]):
            (sparse_embeddings, dense_embeddings
             ) = self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(0).unsqueeze(1),
            )

            sparse_embeddings = sparse_embeddings.to(self.dtype)
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=features[i],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.sam.postprocess_masks(
                low_res_masks,
                input_size=input_sizes[i],
                original_size=original_sizes[i],
            )
            pred_masks.append(pred_mask[:, 0])

        # loss, ce_loss, mask_bce_loss, mask_dice_loss = None, None, None, None
        ce_loss = outputs.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx][np.newaxis, ...]
            pred_mask = pred_masks[batch_idx]

            assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
            )
            mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        return VideoLISACausalLMOutputWithPast(
            loss=loss,
            ce_loss=ce_loss,
            mask_bce_loss=mask_bce_loss,
            mask_dice_loss=mask_dice_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            # pred_masks=pred_masks
        )

    @torch.no_grad()
    def generate(self, original_images, *args, **kwargs):
        outputs = super().generate(output_hidden_states=True,
                                   return_dict_in_generate=True,
                                   *args, **kwargs)
        # Tuple[
        # Tuple[
        # Torch, (b, 1 or num(input_ids), hidden_states)
        # ], Len=Num of attention layers
        # ], Len=New generated tokens
        output_hidden_states = outputs.hidden_states
        output_ids = outputs.sequences

        seg_token_ids = torch.where(output_ids[:, :] == self.seg_token_idx)

        seg_token_hidden_states = []
        for b, i in zip(*seg_token_ids):
            offset = output_ids.shape[1]-i
            seg_token_hidden_states.append(output_hidden_states[-offset][-1][b].unsqueeze(0))

        seg_token_hidden_states = torch.cat(seg_token_hidden_states, dim=0)
        pred_embeddings = self.text_hidden_fcs[0](seg_token_hidden_states)

        # TODO: image input
        transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        assert pred_embeddings.shape[0] == len(original_images), "Prediction size mismatch image number"
        pred_masks = []
        for idx, image in enumerate(original_images):
            image_np = np.array(image)
            input_image = transform.apply_image(image_np)
            input_image_torch = torch.as_tensor(input_image, device=self.device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

            transformed_image = input_image_torch
            original_image_size = image_np.shape[:2]
            assert (
                    len(transformed_image.shape) == 4
                    and transformed_image.shape[1] == 3
                    and max(*transformed_image.shape[2:]) == self.sam.image_encoder.img_size
            ), f"set_torch_image input must be BCHW with long side {self.sam.image_encoder.img_size}."

            original_size = original_image_size
            input_size = tuple(transformed_image.shape[-2:])
            input_image = self.sam.preprocess(transformed_image)
            features = self.sam.image_encoder(input_image)

            # TODO: sam process
            (sparse_embeddings, dense_embeddings
            ) = self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings,
            )

            sparse_embeddings = sparse_embeddings.to(self.dtype)
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=features,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.sam.postprocess_masks(
                low_res_masks,
                input_size=input_size,
                original_size=original_size,
            )
            pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks