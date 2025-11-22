import inspect
import os
from contextlib import nullcontext

import numpy as np
import torch
from unsloth.models.rl_replacements import (
    RL_FUNCTIONS,
    RL_REPLACEMENTS,
    grpo_trainer_compute_loss,
)
from unsloth_zoo.rl_replacements import (
    UnslothEfficientGRPO,
    calculate_pad_tokens_in_prompt,
    create_completion_attention_mask,
    left_pack_padding,
)


def grpo_accumulated_loss(
    trainer,
    input_ids,
    attention_mask,
    logits_to_keep,
    completion_mask,
    advantages,
    old_hidden_states,
    ref_hidden_states,
    thinking_embeds=None,
    thinking_mask=None,
    n_chunks=-1,
    **kwargs,
):
    # All Unsloth Zoo code licensed under LGPLv3
    bsz, qlen = input_ids.shape

    pixel_values = kwargs.get("pixel_values", None)
    image_grid_thw = kwargs.get("image_grid_thw", None)
    pixel_attention_mask = kwargs.get("pixel_attention_mask", None)
    image_sizes = kwargs.get("image_sizes", None)
    # Find closest multiple
    factors = [i for i in range(1, bsz + 1) if bsz % i == 0]
    if n_chunks == -1:
        n_chunks = bsz
    n_chunks = factors[min(np.searchsorted(factors, n_chunks), len(factors) - 1)]

    if not hasattr(trainer, "_autocast_dtype"):
        trainer._autocast_dtype = (
            torch.float16
            if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
            else torch.bfloat16
        )
        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            trainer._autocast_dtype = None
    pass
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    lm_head = trainer.model.get_output_embeddings().weight

    if pixel_values is None:
        left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(
            input_ids, logits_to_keep, trainer.processing_class.pad_token_id
        )

        max_left_pad = max(left_pad_tokens_per_prompt).item()

        input_ids = left_pack_padding(input_ids, trainer.processing_class.pad_token_id)

        completion_input_ids = input_ids[:, -(logits_to_keep + max_left_pad) :]

        completion_mask = create_completion_attention_mask(
            completion_input_ids,
            left_pad_tokens_per_prompt,
            max_left_pad,
            trainer.processing_class.pad_token_id,
        ).to(attention_mask.dtype)
        attention_mask = input_ids != trainer.processing_class.pad_token_id
        attention_mask = attention_mask.to(attention_mask.dtype)
    else:
        completion_input_ids = input_ids[:, -logits_to_keep:]

    unwrapped_model = trainer.accelerator.unwrap_model(
        trainer.model, keep_fp32_wrapper=False
    )

    # Do not move hidden_states from device 1 to device 0:
    for module in unwrapped_model.modules():
        if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "io_same_decice"):
            module._hf_hook.io_same_decice = False
    pass

    # Get autocaster
    if trainer._autocast_dtype is None:
        autocaster = nullcontext()
    else:
        autocaster = torch.amp.autocast(
            device_type=trainer.model.device.type, dtype=trainer._autocast_dtype
        )
    with autocaster:
        if pixel_values is None:
            if thinking_embeds is not None:
                thinking_embeds = thinking_embeds.clone()
            if thinking_mask is not None:
                thinking_mask = thinking_mask.clone()

            new_hidden_states = unwrapped_model(
                input_ids=input_ids,
                inputs_embeds=thinking_embeds,
                thinking_mask=thinking_mask,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_attention_mask=pixel_attention_mask,
                image_sizes=image_sizes,
                # logits_to_keep = logits_to_keep + 1,
            ).logits

            # keep extra logit as we generated a new token
            new_hidden_states = new_hidden_states[
                :, -(logits_to_keep + max_left_pad + 1) :, :
            ]
            if ref_hidden_states is not None:
                ref_hidden_states = ref_hidden_states[
                    :, -(logits_to_keep + max_left_pad + 1) :, :
                ]
            if old_hidden_states is not None:
                old_hidden_states = old_hidden_states[
                    :, -(logits_to_keep + max_left_pad + 1) :, :
                ]
        else:
            raise NotImplementedError(
                "ERROR: HRPO not implemented for pixel_values is not None"
            )

    loss, completion_length, mean_kl = UnslothEfficientGRPO.apply(
        new_hidden_states,
        old_hidden_states,
        ref_hidden_states,
        lm_head,
        completion_input_ids,
        completion_mask,
        advantages,
        trainer.beta,
        trainer.accelerator.scaler,
        n_chunks,
        kwargs,  # pass kwargs as a dict
    )

    # Must force not returning hidden states but logits otherwise gibberish
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

    return loss, completion_length, mean_kl
    # Old non efficient code path
    new_logits = torch.matmul(new_hidden_states, lm_head.t())
    new_logits = new_logits[
        :, :-1, :
    ]  # exclude the last logit: it corresponds to the next token pred
    old_logits = torch.matmul(old_hidden_states, lm_head.t())
    old_logits = old_logits[
        :, :-1, :
    ]  # exclude the last logit: it corresponds to the next token pred
    loss, completion_length, mean_kl = grpo_compute_loss(
        old_logits,
        new_logits,
        completion_input_ids,
        completion_mask,
        trainer.beta,
        advantages,
    )
    return loss, completion_length, mean_kl
    pass


pass
RL_REPLACEMENTS["grpo_accumulated_loss"] = grpo_accumulated_loss


# Edit _get_per_token_logps to handle mixed precision
def _grpo_trainer_compute_loss(function_name, function):
    if function_name != "compute_loss":
        return function

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        thinking_embeds, thinking_mask = (
            inputs["thinking_embeds"],
            inputs["thinking_mask"],
        )
        pixel_values, image_grid_thw = inputs.get("pixel_values", None), inputs.get(
            "image_grid_thw", None
        )
        pixel_attention_mask, image_sizes = inputs.get(
            "pixel_attention_mask", None
        ), inputs.get("image_sizes", None)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        bsz, qlen = input_ids.shape
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        # attention_mask = None
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        _input_ids = input_ids
        _thinking_embeds = thinking_embeds
        _logits_to_keep = logits_to_keep

        get_logps_func = lambda model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False, compute_efficient=False: (
            self._get_per_token_logps(
                model, input_ids, attention_mask, logits_to_keep, compute_efficient
            )
            if hasattr(self, "_get_per_token_logps")
            else self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size,
                compute_entropy,
                # compute_efficient,
                thinking_embeds=thinking_embeds,
                thinking_mask=thinking_mask,
            )[0]
        )  # logps
        # breakpoint()
        per_token_logps = get_logps_func(
            model, input_ids, attention_mask, logits_to_keep, compute_efficient=True
        )
        # Compute the KL divergence between the model and the reference model
        # _prepare_inputs doesn't return reference log probs anymore. We need to calculate it ourselves.
        # https://github.com/huggingface/trl/blob/05bc43e960396581e458195b8388efe6b82cae1f/trl/trainer/grpo_trainer.py#L1328
        # if self.beta != 0.0:
        #     with torch.inference_mode(), model.disable_adapter():
        #         ref_per_token_logps = per_token_logps = get_logps_func(model, input_ids, attention_mask, logits_to_keep)
        # else:
        #     ref_per_token_logps = None
        ref_hidden_states = inputs.get("ref_per_token_logps", None)
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        old_hidden_states = inputs.get("old_per_token_logps", None)

        input_ids = input_ids[:, -logits_to_keep:]

        # Get logit softcapping and logit scale
        logit_softcapping = getattr(model.config, "final_logit_softcapping", 0)  # Gemma
        if logit_softcapping is None:
            logit_softcapping = 0
        logit_scale_multiply = getattr(model.config, "logit_scale", 0)  # Cohere
        if logit_scale_multiply is None:
            logit_scale_multiply = 0
        logit_scale_divide = getattr(model.config, "logits_scaling", 0)  # Granite
        if logit_scale_divide is None:
            logit_scale_divide = 0

        if per_token_logps is not None:
            raise NotImplementedError(
                "ERROR: HRPO not implemented for per_token_logps is not None"
            )

            if ref_hidden_states is not None:
                ref_hidden_states = ref_hidden_states[
                    :, :-1, :
                ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            if old_hidden_states is not None:
                old_hidden_states = old_hidden_states[
                    :, :-1, :
                ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            per_token_logps = per_token_logps[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            loss, completion_length, mean_kl = grpo_compute_loss_slow(
                ref_hidden_states,
                per_token_logps,
                old_hidden_states,
                input_ids,
                completion_mask,
                self.beta,
                advantages,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                loss_type=self.args.loss_type,
                importance_sampling_level=self.importance_sampling_level,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                max_completion_length=self.args.max_completion_length,
                delta=self.args.delta,
                temperature=self.args.temperature,
                logit_softcapping=logit_softcapping,
                logit_scale_multiply=logit_scale_multiply,
                logit_scale_divide=logit_scale_divide,
            )
        else:
            if hasattr(self.args, "loss_type"):
                loss, completion_length, mean_kl = grpo_accumulated_loss(
                    trainer=self,
                    input_ids=_input_ids,
                    thinking_embeds=_thinking_embeds,
                    thinking_mask=thinking_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    logits_to_keep=logits_to_keep,
                    completion_mask=completion_mask,
                    advantages=advantages,
                    old_hidden_states=old_hidden_states,
                    ref_hidden_states=ref_hidden_states,
                    n_chunks=self.args.unsloth_num_chunks,
                    loss_type=self.args.loss_type,
                    importance_sampling_level=self.importance_sampling_level,
                    epsilon_low=self.epsilon_low,
                    epsilon_high=self.epsilon_high,
                    max_completion_length=self.args.max_completion_length,
                    delta=self.args.delta,
                    temperature=self.args.temperature,
                    logit_softcapping=logit_softcapping,
                    logit_scale_multiply=logit_scale_multiply,
                    logit_scale_divide=logit_scale_divide,
                    attention_mask=attention_mask,
                )
            else:
                # to ensure backwards compatibility with trl 0.15.2 and maybe even 0.17
                loss, completion_length, mean_kl = grpo_accumulated_loss(
                    trainer=self,
                    input_ids=_input_ids,
                    thinking_embeds=_thinking_embeds,
                    thinking_mask=thinking_mask,
                    logits_to_keep=logits_to_keep,
                    completion_mask=completion_mask,
                    advantages=advantages,
                    old_hidden_states=old_hidden_states,
                    ref_hidden_states=ref_hidden_states,
                    n_chunks=self.args.unsloth_num_chunks,
                    temperature=self.args.temperature,
                    logit_softcapping=logit_softcapping,
                    logit_scale_multiply=logit_scale_multiply,
                    logit_scale_divide=logit_scale_divide,
                    attention_mask=attention_mask,
                )
            pass
        pass
        # Log the metrics
        # completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        # mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        if "train" in self._metrics:
            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics[mode]["completion_length"].append(completion_length.item())
            self._metrics[mode]["kl"].append(mean_kl.item())
        else:
            self._metrics["completion_length"].append(completion_length.item())
            self._metrics["kl"].append(mean_kl.item())
        return loss

    pass

    function = inspect.getsource(compute_loss)
    return function


pass
_idx = RL_FUNCTIONS["grpo_trainer"].index(grpo_trainer_compute_loss)
RL_FUNCTIONS["grpo_trainer"][_idx] = _grpo_trainer_compute_loss

grpo_trainer_compute_loss = _grpo_trainer_compute_loss
