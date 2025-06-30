from typing import Optional, List, Union, Tuple, Dict, Any
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from transformers.modeling_outputs import ModelOutput
from videolisa.model.sam2.build_sam import build_sam2_video_predictor
from transformers import Trainer
from transformers.utils import (is_torch_mlu_available,
                                is_torch_mps_available,
                                is_torch_musa_available,
                                is_torch_npu_available,
                                is_torch_xpu_available,
                                is_apex_available)
from transformers.training_args import OptimizerNames
from accelerate.utils import DistributedType
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from collections import OrderedDict
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os


THRESHOLD = 1.5

@torch.no_grad()
def debug_mask(prediction_mask, groundtruth_mask, original_img):
    original_img = original_img / 255.0
    # Convert masks to numpy and squeeze to (H, W)
    pred_mask_np = torch.sigmoid(prediction_mask).squeeze().numpy()  # Apply sigmoid to logits, shape: (H, W)
    gt_mask_np = groundtruth_mask.squeeze().numpy()   # Shape: (H, W)

    # Create RGB versions of masks for overlay
    pred_mask_rgb = np.stack([pred_mask_np, pred_mask_np, pred_mask_np], axis=-1)  # Shape: (H, W, 3)
    gt_mask_rgb = np.stack([gt_mask_np, gt_mask_np, gt_mask_np], axis=-1)         # Shape: (H, W, 3)

    # Overlay masks on original image (e.g., red color for pred, blue for gt)
    alpha = 0.5  # Transparency factor
    masked_pred = original_img * (1 - alpha * pred_mask_rgb) + np.array([1, 0, 0]) * alpha * pred_mask_rgb
    masked_gt = original_img * (1 - alpha * gt_mask_rgb) + np.array([0, 0, 1]) * alpha * gt_mask_rgb

    # Clip values to [0, 1]
    masked_pred = np.clip(masked_pred, 0, 1)
    masked_gt = np.clip(masked_gt, 0, 1)

    output_dir = "output"
    # Save original image
    plt.imsave(os.path.join(output_dir, "original_image.png"), original_img)

    # Save masked images
    plt.imsave(os.path.join(output_dir, "prediction_mask.png"), masked_pred)
    plt.imsave(os.path.join(output_dir, "groundtruth_mask.png"), masked_gt)

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        scale: float=1000,
        eps: float=1e-6,
):
    """
    B C W H
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(2, 3)
    targets = targets.flatten(2, 3)
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
    loss = loss.flatten(2, 3).mean(2).sum() / (num_masks + 1e-8)
    return loss


def load_images(
        images,
        image_size,
        offload_video_to_cpu=False,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        compute_device=torch.device("cuda"),
):
    img_mean = torch.tensor(img_mean)[:, None, None]
    img_std = torch.tensor(img_std)[:, None, None]
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 调整到指定尺寸
        transforms.ToTensor(),
    ])
    video_height, video_width = images[0].shape[:2]
    images = [preprocess(Image.fromarray(image, mode='RGB')) for image in images]

    images = torch.stack(images, dim=0)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


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
        super().__init__(config)

        self.ce_loss_weight = 1.0
        self.dice_loss_weight = 0.5
        self.bce_loss_weight = 2.0

        self.enable_segmentation = False

        self.seg_token_idx = -1
        self.sam = None

        self.training = False

    def init_sam_module(self, model_path):
        # SAM2 Initialization
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="sam2"):
            self.sam = build_sam2_video_predictor(model_cfg, model_path, text_embed_dim=2048, device=self.device)

    def init_sam_state(self, images):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        inference_state = {}
        images, video_height, video_width = load_images(images, image_size=1024)
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["text_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self.sam._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def forward(
            self,
            images: Optional[torch.Tensor] = None,
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
    ) -> tuple | Qwen2_5_VLCausalLMOutputWithPast | VideoLISACausalLMOutputWithPast:

        #
        if not self.enable_segmentation:
            return super().forward(input_ids=input_ids,
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
                                 output_hidden_states=False,
                                 )
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
        last_hidden_state = output_hidden_states[-1]
        seg_token_mask = input_ids == self.seg_token_idx
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )
        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        assert pred_embeddings is not None, "Cannot find <seg> token."

        # TODO: image input
        assert len(pred_embeddings) == len(images), "Prediction size mismatch image number"
        pred_masks = []
        for idx, image in enumerate(images):
            inference_state = self.init_sam_state(images=[image])
            self.sam.reset_state(inference_state)
            _, out_obj_ids, out_mask_logits = self.sam.add_text_embeddings(
                inference_state=inference_state,
                texts=pred_embeddings[idx],
                frame_idx=0,
                obj_id=0,
            )
            pred_masks.append(out_mask_logits)

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
            if mask_dice_loss < 0.3:
                print("mask dice loss: {}".format(mask_dice_loss))
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

    @torch.inference_mode()
    def generate(self, images, *args, **kwargs):
        outputs = super().generate(output_hidden_states=True,
                                   return_dict_in_generate=True,
                                   num_beams=1,
                                   *args, **kwargs)
        # Tuple[
        # Tuple[
        # Torch, (b, 1 or num(input_ids), hidden_states)
        # ], Len=Num of attention layers
        # ], Len=New generated tokens
        output_hidden_states = outputs.hidden_states
        output_ids = outputs.sequences

        seg_token_ids = torch.where(output_ids == self.seg_token_idx)
        seg_token_hidden_states = []
        for b, i in zip(*seg_token_ids):
            offset = output_ids.shape[1]-i
            seg_token_hidden_states.append(output_hidden_states[-offset][-1])

        if len(seg_token_hidden_states) > 0:
            seg_token_hidden_states = torch.cat(seg_token_hidden_states, dim=0)
            pred_embeddings = seg_token_hidden_states

            assert pred_embeddings.shape[0] == len(images), "Prediction size mismatch image number"
            pred_masks = []
            for idx, image in enumerate(images):
                inference_state = self.init_sam_state(images=[image])
                self.sam.reset_state(inference_state)
                _, out_obj_ids, out_mask_logits = self.sam.add_text_embeddings(
                    inference_state=inference_state,
                    texts=pred_embeddings[idx],
                    frame_idx=0,
                    obj_id=0,
                )
                pred_masks.append(out_mask_logits)
        else:
            pred_masks = []
        return output_ids, pred_masks


class LISATrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.sam.fill_hole_area = 0
        self.model.sam.multimask_output = False
        self.model.sam.multimask_output_for_tracking = False
        self.model.sam.multimask_output_in_sam = False

        self.ce_loss = 0
        self.mask_bce_loss = 0
        self.mask_dice_loss = 0

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Config model to be training mode
        self.model.train()
        # self.model.base_model.sam.eval()
        model.base_model.sam.sam_prompt_encoder.project_text.train()
        model.base_model.sam.sam_mask_decoder.train()
        self.model.sam.pred_obj_scores = False


        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if "ce_loss" in outputs.keys():
            self.ce_loss += outputs["ce_loss"].item()
            self.mask_bce_loss += outputs["mask_bce_loss"].item()
            self.mask_dice_loss += outputs["mask_dice_loss"].item()
            if self.state.global_step % self.args.logging_steps == 0:
                extra_metrics = {"ce_loss": self.ce_loss/self.args.logging_steps,
                                 "mask_bce_loss": self.mask_bce_loss/self.args.logging_steps,
                                 "mask_dice_loss": self.mask_dice_loss/self.args.logging_steps}
                self.ce_loss, self.mask_bce_loss, self.mask_dice_loss = 0, 0, 0
                self.log(extra_metrics)
            # with torch.no_grad():
            # #     for name, param in self.model.base_model.sam.sam_prompt_encoder.project_text.named_parameters():
            # #         print(f"{name}: {param.grad}")
            #     print(self.model.base_model.sam.sam_prompt_encoder.project_text.layers[0].weight.detach().sum())
            #     print(self.model.base_model.model.lm_head.weight.detach().sum())

        del inputs
        if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            self.accelerator.backward(loss, **kwargs)
            # self.accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # from torchviz import make_dot
            # self.accelerator.backward(loss, retain_graph=True, **kwargs)
            #
            # dot = make_dot(loss, params=dict(self.model.base_model.sam.sam_prompt_encoder.project_text.named_parameters()))
            # dot.render("computation_graph-large", format="png")  # 保存为 PNG 文件
            return loss.detach()
