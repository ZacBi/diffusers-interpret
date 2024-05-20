from typing import List, Optional, Tuple, Union

import accelerate
import torch
from detectron2.data.detection_utils import convert_PIL_to_numpy
from diffusers import DiffusionPipeline
from PIL.Image import Image

from diffusers_interpret.data import (
    AttributionAlgorithm,
    AttributionMethods,
    PipelineExplainerForBoundingBoxOutput,
    PipelineExplainerOutput,
    PipelineImg2ImgExplainerForBoundingBoxOutputOutput,
    PipelineImg2ImgExplainerOutput,
)
from diffusers_interpret.explainers import StableDiffusionPipelineExplainer
from diffusers_interpret.generated_images import GeneratedImages


class StableDiffusionPipelineDetExplainer(StableDiffusionPipelineExplainer):

    def __init__(
        self,
        pipe: DiffusionPipeline,
        verbose: bool = True,
        gradient_checkpointing: bool = False,
        det_model=None,
    ) -> None:
        super().__init__(pipe, verbose, gradient_checkpointing)
        self.det_model = det_model

    def _get_attributions(
        self,
        output: Union[PipelineExplainerOutput, PipelineExplainerForBoundingBoxOutput],
        attribution_method: AttributionMethods,
        tokens: List[List[str]],
        text_embeddings: torch.Tensor,
        explanation_2d_bounding_box: Optional[
            Tuple[Tuple[int, int], Tuple[int, int]]
        ] = None,
        consider_special_tokens: bool = False,
        clean_token_prefixes_and_suffixes: bool = True,
        n_last_diffusion_steps_to_consider_for_attributions: Optional[int] = None,
        **kwargs,
    ) -> Union[
        PipelineExplainerOutput,
        PipelineExplainerForBoundingBoxOutput,
        PipelineImg2ImgExplainerOutput,
        PipelineImg2ImgExplainerForBoundingBoxOutputOutput,
    ]:
        if self.verbose:
            print("Calculating token attributions... ", end="")

        target_cls_id = kwargs["target_cls_id"]
        raw_embeds = kwargs['raw_embeds']

        input_embeds = (text_embeddings,)
        if raw_embeds is not None:
            input_embeds = (raw_embeds,)

        token_attributions = (
            self.gradients_attribution(
                pred_logits=output.image,
                input_embeds=input_embeds,
                attribution_algorithms=[attribution_method.tokens_attribution_method],
                explanation_2d_bounding_box=explanation_2d_bounding_box,
                target_cls_id=target_cls_id
            )[0]
            .detach()
            .cpu()
            .numpy()
        )

        # 直接传入embeddings时, 不存在tokens, 需要mock
        if tokens is None:
            tokens = [[str(i) for i in range(raw_embeds.shape[1])]]

        output = self._post_process_token_attributions(
            output=output,
            tokens=tokens,
            token_attributions=token_attributions,
            consider_special_tokens=consider_special_tokens,
            clean_token_prefixes_and_suffixes=clean_token_prefixes_and_suffixes,
        )

        if self.verbose:
            print("Done!")

        return output

    def _mask_target_cls(self, image: torch.Tensor, target_cls_id: int) -> torch.Tensor:
        """
        use detect model to mask the target cls
        """
        if target_cls_id == -1 or self.det_model is None:
            # 返回一个identity矩阵
            return torch.ones_like(image, dtype=torch.bool)

        # clone and detach
        image = image.clone().detach()
        height, width, channel = image.shape
        # transform image to PIL type to adapt the input of detect model
        all_images = GeneratedImages(
            all_generated_images=[image],
            pipe=self.pipe,
            remove_batch_dimension=True,
            prepare_image_slider=False,
        )

        image = torch.as_tensor(
            convert_PIL_to_numpy(all_images[-1], format="BGR")
            .astype("float32")
            .transpose(2, 0, 1)
        )

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.det_model([inputs])[0]
        # offload det_model to cpu for saving gpu memory
        accelerate.cpu_offload(self.det_model)

        instances = predictions["instances"]
        pred_masks = instances.pred_masks
        pred_classes = instances.pred_classes
        # 预测的class tensor可能为[cat, chair, cat], 需要merge boolean matrix
        mask = torch.any(pred_masks[pred_classes == target_cls_id], dim=0)
        # 扩张到和image同样的维度
        return mask.unsqueeze(0).repeat(channel, 1, 1).permute(1, 2, 0)

    def gradients_attribution(
        self,
        pred_logits: torch.Tensor,
        input_embeds: Tuple[torch.Tensor],
        attribution_algorithms: List[AttributionAlgorithm],
        explanation_2d_bounding_box: Optional[
            Tuple[Tuple[int, int], Tuple[int, int]]
        ] = None,
        retain_graph: bool = False,
        target_cls_id: int = -1,
    ) -> List[torch.Tensor]:
        # TODO: add description

        assert len(pred_logits.shape) == 3
        if explanation_2d_bounding_box:
            upper_left, bottom_right = explanation_2d_bounding_box
            pred_logits = pred_logits[
                upper_left[0] : bottom_right[0], upper_left[1] : bottom_right[1], :
            ]

        assert len(input_embeds) == len(attribution_algorithms)

        # get mask matrix for target class
        traget_mask = self._mask_target_cls(pred_logits, target_cls_id)

        # Construct tuple of scalar tensors with all `pred_logits`
        # The code below is equivalent to `tuple_of_pred_logits = tuple(torch.flatten(pred_logits))`,
        #  but for some reason the gradient calculation is way faster if the tensor is flattened like this
        tuple_of_pred_logits = []
        for px, mx in zip(pred_logits, traget_mask):
            for py, my in zip(px, mx):
                for pz, mz in zip(py, my):
                    if mz:
                        tuple_of_pred_logits.append(pz)
        tuple_of_pred_logits = tuple(tuple_of_pred_logits)

        # get the sum of back-prop gradients for all predictions with respect to the inputs
        if torch.is_autocast_enabled():
            # FP16 may cause NaN gradients https://github.com/pytorch/pytorch/issues/40497
            # TODO: this is still an issue, the code below does not solve it
            with torch.autocast(input_embeds[0].device.type, enabled=False):
                grads = torch.autograd.grad(
                    tuple_of_pred_logits, input_embeds, retain_graph=retain_graph
                )
        else:
            grads = torch.autograd.grad(
                tuple_of_pred_logits, input_embeds, retain_graph=retain_graph
            )

        if torch.isnan(grads[-1]).any():
            raise RuntimeError(
                "Found NaNs while calculating gradients. "
                "This is a known issue of FP16 (https://github.com/pytorch/pytorch/issues/40497).\n"
                "Try to rerun the code or deactivate FP16 to not face this issue again."
            )

        # Aggregate
        aggregated_grads = []
        for grad, inp, attr_alg in zip(grads, input_embeds, attribution_algorithms):
            if attr_alg == AttributionAlgorithm.GRAD_X_INPUT:
                aggregated_grads.append(torch.norm(grad * inp, dim=-1))
            elif attr_alg == AttributionAlgorithm.MAX_GRAD:
                aggregated_grads.append(grad.abs().max(-1).values)
            elif attr_alg == AttributionAlgorithm.MEAN_GRAD:
                aggregated_grads.append(grad.abs().mean(-1).values)
            elif attr_alg == AttributionAlgorithm.MIN_GRAD:
                aggregated_grads.append(grad.abs().min(-1).values)
            elif attr_alg == AttributionAlgorithm.GRAD_X_INPUT_NO_NORM:
                aggregated_grads.append(grad * inp)
            else:
                raise NotImplementedError(
                    f"aggregation type `{attr_alg}` not implemented"
                )

        return aggregated_grads
