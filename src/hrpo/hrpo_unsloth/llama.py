from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from unsloth.models.llama import *
from unsloth.models._utils import __version__, _prepare_model_for_qat
from unsloth_zoo.utils import _get_dtype
from peft import get_peft_model as _get_peft_model
from src.hrpo.hrpo_transformers.modeling_llama import HRPOLLamaModel


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
def LlamaModel_fast_forward(
    self,
    input_ids:            torch.LongTensor,
    causal_mask:          Optional[BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_values:      Optional[List[torch.FloatTensor]] = None,
    inputs_embeds:        Optional[torch.FloatTensor] = None,
    use_cache:            Optional[bool] = None,
    output_attentions:    Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict:          Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    assert(output_attentions is False)
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        thinking_embeds = inputs_embeds
        batch_size, seq_length, _ = inputs_embeds.shape
        # raise ValueError("Unsloth: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("Unsloth: You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length

    # Fix out of bounds tokenization
    if hasattr(self, "max_seq_length"):
        if seq_length > self.max_seq_length:
            shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
            logger.warning_once(
                f"Unsloth: Input IDs of shape {shape} with length {seq_length} > the model's max sequence length of {self.max_seq_length}.\n"\
                "We shall truncate it ourselves. It's imperative if you correct this issue first."
            )
        if input_ids is not None:
            input_ids = input_ids[:,:self.max_seq_length]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:,:self.max_seq_length,:]
        pass
    pass

    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    pass

    # We already handle KV cache position_ids ourselves.
    if False:#(past_key_values_length != 0):
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length,
            dtype  = torch.int32,
            device = f"{DEVICE_TYPE}:0",
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    elif position_ids is not None:
        position_ids = position_ids.view(-1, seq_length).to(torch.int32)#.long()
    else:
        position_ids = None
    pass

    if position_ids is not None:
        if position_ids.shape[0] != batch_size:
            position_ids = position_ids.repeat((batch_size, 1))
    pass

    # Embed positions
    if input_ids is not None:
        inputs_embeds = self.embed_tokens(input_ids)

    thinking_mask = kwargs.get('thinking_mask')
    if thinking_mask is not None:
        new_inputs_embeds = inputs_embeds.clone()
        new_inputs_embeds[thinking_mask] = self.thinking_residual(
            inputs_embeds[thinking_mask], thinking_embeds[thinking_mask],
        )[0].to(inputs_embeds.dtype)
        inputs_embeds = new_inputs_embeds

    inputs_embeds = inputs_embeds.to(_get_dtype(dtype_from_config(self.config)))

    # Normalized from Gemma
    IS_GEMMA   = self.config.model_type.startswith("gemma")
    IS_GEMMA2  = self.config.model_type.startswith("gemma2")
    IS_COHERE  = self.config.model_type.startswith("cohere")
    IS_GRANITE = self.config.model_type.startswith("granite")
    IS_FALCON_H1 = self.config.model_type.startswith("falcon_h1")

    train_embed_tokens = self.embed_tokens.weight.requires_grad

    if IS_GEMMA:
        # Match Gemma exactly by casting to bfloat16 / float16
        # inputs_embeds *= math_sqrt(self.config.hidden_size)
        # Ie 3072**0.5 = 55.5000 in bfloat16, whilst 55.4256 in float32
        # &  2048**0.5 = 45.2500 in bfloat16, whilst 45.2548 in float32
        normalizer = torch.tensor(math_sqrt(self.config.hidden_size), dtype = inputs_embeds.dtype)

        if train_embed_tokens:
            # Careful we must not do an inplace op!
            inputs_embeds = inputs_embeds * normalizer
        else:
            inputs_requires_grad = inputs_embeds.requires_grad
            if not inputs_embeds.is_leaf:
                inputs_embeds = inputs_embeds.detach()
                inputs_requires_grad = True
            elif inputs_requires_grad:
                inputs_embeds.requires_grad_(False)
            pass
            inputs_embeds *= normalizer
            # inputs_embeds *= math_sqrt(self.config.hidden_size)
            if inputs_requires_grad: inputs_embeds.requires_grad_(True)
        pass
    pass

    # Fix up attention mask by setting elements to 0
    # Specifically for DPO
    if getattr(self, "_has_no_labels", False) is True and (attention_mask is not None) and (past_key_values is None) and \
        (not train_embed_tokens) and self.training:
        # Careful for inference the attention_mask is size (1, kv_seq_len)
        # Whilst the input_embeds is size (1, 1, 4096)
        inputs_requires_grad = inputs_embeds.requires_grad
        if not inputs_embeds.is_leaf:
            inputs_embeds = inputs_embeds.detach()
            inputs_requires_grad = True
        elif inputs_requires_grad:
            inputs_embeds.requires_grad_(False)
        pass
        attention_mask = attention_mask[:,:self.max_seq_length] # Must resize!
        inputs_embeds *= attention_mask.unsqueeze(0).transpose(0, 1).transpose(1, 2)
        if inputs_requires_grad: inputs_embeds.requires_grad_(True)
    pass

    # Ignore attention_mask
    if attention_mask is None:
        padding_mask = None
    elif self.training:
        attention_mask = None
        padding_mask = None
    else:
        # if 0 in attention_mask:
        #     padding_mask = attention_mask
        # else:
        padding_mask = None

        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window = getattr(self.config, "sliding_window", None),
        )
        # Must NOT convert to bool - weirdly this causes stuff to error out!
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(torch.bool)
    pass

    hidden_states = inputs_embeds
    if IS_GRANITE or IS_FALCON_H1: #granite has embedding multiplier
        hidden_states = self.config.embedding_multiplier * hidden_states

    if past_key_values is None and self.training:
        use_cache = False
        # if use_cache:
        #     logger.warning_once(
        #         "Unsloth: `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`"
        #     )
        #     use_cache = False
    pass

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # Gradient checkpointing methods (ie sqrt)
    if hasattr(self, "_gradient_checkpointing_boundaries"):
        boundaries = self._gradient_checkpointing_boundaries
    else:
        boundaries = None
    pass

    # Check checkpointing method
    gradient_checkpointing = False

    if (self.gradient_checkpointing and self.training and not use_cache):
        gradient_checkpointing = True
    pass

    # Gemma2 has alternating SWA and global attn
    use_static_mask  = True
    dynamic_SWA_mask = None
    dynamic_GA_mask  = None
    if IS_GEMMA2:
        if HAS_FLASH_ATTENTION_SOFTCAPPING and attention_mask is None:
            self.SWA_mask = True
            self.GA_mask  = False
        elif attention_mask is not None:
            # Fixes https://github.com/unslothai/unsloth/issues/853
            # Unsloth needs a 2D mask, not a [2, 1, n, n] mask!

            # https://github.com/pytorch/pytorch/issues/103749
            # Need to convert to float and not using bool
            # attention_mask = (1.0 - attention_mask.float()) * torch.finfo(inputs_embeds.dtype).min
            dynamic_SWA_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window = self.config.sliding_window,
            )
            dynamic_GA_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window = None,
            )
            use_static_mask = False

        elif not hasattr(self, "SWA_mask"):
            if HAS_FLEX_ATTENTION:
                # Use Flex Attention instead!
                self.SWA_mask = create_flex_attention_sliding_window_mask(self.max_seq_length, self.config.sliding_window)
                self.GA_mask  = create_flex_attention_causal_mask(self.max_seq_length)
            else:
                n = self.max_seq_length # self.config.max_position_embeddings
                # masked_fill is making stuff slower!
                # self. GA_mask = create_boolean_mask(n = n, sliding_window = 0)
                # self.SWA_mask = create_boolean_mask(n = n, sliding_window = self.config.sliding_window)
                from transformers.modeling_attn_mask_utils import AttentionMaskConverter
                self.SWA_mask = AttentionMaskConverter(
                    is_causal = True,
                    sliding_window = self.config.sliding_window,
                )\
                    .to_causal_4d(1, n, n, dtype = inputs_embeds.dtype, device = DEVICE_TYPE,)\
                    .squeeze(0).squeeze(0)

                self.GA_mask = AttentionMaskConverter(
                    is_causal = True,
                )\
                    .to_causal_4d(1, n, n, dtype = inputs_embeds.dtype, device = DEVICE_TYPE,)\
                    .squeeze(0).squeeze(0)
            pass
        pass
    pass

    if (IS_ATTENTION_REFACTOR and (hasattr(self, "rotary_emb") or not hasattr(self.layers[0].self_attn, "rotary_emb"))) or IS_GRANITE:
        # Transformers main has made it mandatory to pass position_embeddings
        # https://github.com/huggingface/transformers/pull/34858
        # Also, transformers 4.45.0 supports granite but with the attention refactor (it always had the refactor)
        # unsloth's check for granite too has "version >= 4.45.0 (rightly so)".
        # so let granite always use the attention refactor implementation.
        position_embeddings = self.rotary_emb.get_cached(self.config.max_position_embeddings, hidden_states.device.index)
    else:
        position_embeddings = None

    # Go through every layer!
    for idx, decoder_layer in enumerate(self.layers):

        if output_hidden_states: all_hidden_states += (hidden_states,)
        past_key_value = past_key_values[idx] if past_key_values is not None else None

        mask = causal_mask
        if IS_GEMMA2:
            if (idx % 2 == 0):
                mask = self.SWA_mask if use_static_mask else dynamic_SWA_mask
            else:
                mask = self. GA_mask if use_static_mask else dynamic_GA_mask
        pass

        if gradient_checkpointing and not isinstance(decoder_layer, GradientCheckpointingLayer):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions, padding_mask = padding_mask, position_embeddings = position_embeddings)
                return custom_forward
            pass
            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                mask,
                attention_mask,
                position_ids,
                use_reentrant = True,
                preserve_rng_state = False,
            )
            hidden_states = layer_outputs[0]

        else:
            layer_outputs = decoder_layer(
                hidden_states,
                causal_mask         = mask,
                attention_mask      = attention_mask,
                position_ids        = position_ids,
                past_key_value      = past_key_value,
                output_attentions   = output_attentions,
                use_cache           = use_cache,
                padding_mask        = padding_mask,
                position_embeddings = position_embeddings,
            )
            hidden_states = layer_outputs[0]
        pass

        if use_cache: next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        if output_attentions: all_self_attns += (layer_outputs[1],)
    pass

    # Final layernorm
    if use_cache:
        if IS_FALCON_H1:
            hidden_states = fast_rms_layernorm_inference(self.final_layernorm, hidden_states)
        else:
            hidden_states = \
                (fast_rms_layernorm_inference_gemma if IS_GEMMA else fast_rms_layernorm_inference)\
                (self.norm, hidden_states)
    elif IS_COHERE:
        hidden_states = self.norm(hidden_states)
    elif IS_FALCON_H1:
        hidden_states = fast_rms_layernorm(self.final_layernorm, hidden_states, gemma = IS_GEMMA)
    else:
        hidden_states = fast_rms_layernorm(self.norm, hidden_states, gemma = IS_GEMMA)
    pass

    if output_hidden_states: all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
pass

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
def _LlamaModel_fast_forward_inference(attention_fast_forward_inference=LlamaAttention_fast_forward_inference, mlp_fast_forward_inference=fast_swiglu_inference):
    # This makes the attention and MLP customisable.
    # Now for models like qwen3 or cohere which use custom attention operations, we can use this function
    def LlamaModel_fast_forward_inference_custom(
        self,
        input_ids,
        past_key_values,
        position_ids,
        attention_mask = None,
        *args, **kwargs,
    ):
        input_ids = input_ids[:,:self.max_seq_length]
        bsz, q_len = input_ids.shape
        hd = self.config.hidden_size
        mlp_size = self.config.intermediate_size

        X = self.model.embed_tokens(input_ids)
        X = X.to(_get_dtype(dtype_from_config(self.config)))
        is_thinking = kwargs.get('is_thinking')
        last_thinking_states = kwargs.get('last_thinking_states')
        if is_thinking is not None and last_thinking_states is not None:
            thinking_embeds = last_thinking_states
            X_hat, a_t = self.model.thinking_residual(
                X, last_thinking_states.unsqueeze(1),
            )
            embeds_ratio = a_t.mean(-1).squeeze()
            embeds_ratio[~torch.tensor(is_thinking)] = 1.
            X[is_thinking] = X_hat[is_thinking].to(X.dtype)
            
        bsz, q_len, hd = X.shape
        assert(q_len == 1)
        # Get saved buffers to reduce memory movement
        residual = torch.empty((bsz, q_len, hd), dtype = torch.float32, device = f"{DEVICE_TYPE}:0")
        _XX = torch.empty((2, bsz, q_len, hd), dtype = torch.float32, device = f"{DEVICE_TYPE}:0")
        XX, XX2 = _XX[0], _XX[1]
        variance = torch.empty((bsz, q_len, 1), dtype = torch.float32, device = f"{DEVICE_TYPE}:0")
        temp_mlp = torch.empty((2, bsz, 1, mlp_size), dtype = X.dtype, device = f"{DEVICE_TYPE}:0")
        temp_gates, temp_ups = tuple(temp_mlp[0].to(torch.device(x)) for x in range(DEVICE_COUNT)), tuple(temp_mlp[1].to(torch.device(x)) for x in range(DEVICE_COUNT))

        seq_len = past_key_values[0][0].shape[-2]
        if bsz != 1:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (bsz, q_len),
                X,
                seq_len,
                sliding_window = getattr(self.config, "sliding_window", None),
            )
        else:
            attention_mask = None
        pass

        next_decoder_cache = []

        for idx, decoder_layer in enumerate(self.model.layers):
            device_index = getattr(decoder_layer, "_per_layer_device_index", 0)
            X, residual, position_ids = move_to_device(
                device_index, X, residual, position_ids
            )
            residual.copy_(X) # residual = X
            X = fast_rms_layernorm_inference(
                decoder_layer.input_layernorm,
                X,
                XX = XX,
                XX2 = XX2,
                variance = variance,
            )
            X, present_key_value = attention_fast_forward_inference(
                decoder_layer.self_attn,
                hidden_states = X,
                past_key_value = past_key_values[idx],
                position_ids = position_ids,
                attention_mask = attention_mask,
                do_prefill = not hasattr(decoder_layer.self_attn, "paged_attention"),
            )
            X += residual

            residual.copy_(X) # residual = X
            X = fast_rms_layernorm_inference(
                decoder_layer.post_attention_layernorm,
                X,
                XX = XX,
                XX2 = XX2,
                variance = variance,
            )
            X = mlp_fast_forward_inference(
                decoder_layer.mlp,
                X,
                temp_gate = temp_gates[device_index],
                temp_up = temp_ups[device_index],
            )
            X += residual

            next_decoder_cache.append(present_key_value)
        pass
        X = fast_rms_layernorm_inference(
            self.model.norm,
            X,
            XX = XX,
            XX2 = XX2,
            variance = variance,
        )

        return BaseModelOutputWithPast(
            last_hidden_state = X,
            past_key_values = next_decoder_cache,
            hidden_states = []  if is_thinking is None else [thinking_embeds, is_thinking, embeds_ratio],
            attentions = [],
        )
    pass
    return LlamaModel_fast_forward_inference_custom

# For ensuring backwards compatibility, we create LlamaModel_fast_forward_inference that is consumed by other models
LlamaModel_fast_forward_inference = _LlamaModel_fast_forward_inference()

class HRPOFastLlamaModel(FastLlamaModel):
    @staticmethod
    def pre_patch():
        init_name, function = patch_llama_rope_scaling(
            model_name           = "llama",
            rope_module          = LlamaRotaryEmbedding,
            scaled_rope_module   = LlamaLinearScalingRotaryEmbedding,
            extended_rope_module = LlamaExtendedRotaryEmbedding,
            attention_module     = LlamaAttention,
            longrope_module      = LongRopeRotaryEmbedding,
        )
        if init_name is not None:
            exec(function, globals())
            LlamaAttention.__init__  = eval(init_name)
        pass
        LlamaAttention      .forward = LlamaAttention_fast_forward
        LlamaSdpaAttention  .forward = LlamaAttention_fast_forward
        LlamaFlashAttention2.forward = LlamaAttention_fast_forward
        LlamaDecoderLayer   .forward = LlamaDecoderLayer_fast_forward
        HRPOLLamaModel      .forward = LlamaModel_fast_forward
        LlamaForCausalLM    .forward = CausalLM_fast_forward(LlamaModel_fast_forward_inference)
        PeftModelForCausalLM.forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(LlamaForCausalLM)

        # Solves https://github.com/unslothai/unsloth/issues/168
        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inference can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.llama.modeling_llama
        transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = LlamaRotaryEmbedding
        transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding = LlamaLinearScalingRotaryEmbedding
        return
    pass

    @staticmethod
    def get_peft_model(
        model,
        r                   = 16,
        target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
        lora_alpha          = 16,
        lora_dropout        = 0.0,
        bias                = "none",
        layers_to_transform = None,
        layers_pattern      = None,
        use_gradient_checkpointing = "unsloth",
        random_state        = 3407,
        max_seq_length      = 2048, # not used anymore
        use_rslora          = False,
        modules_to_save     = None,
        init_lora_weights   = True,
        loftq_config        = {},
        temporary_location  = "_unsloth_temporary_saved_buffers",
        qat_scheme          = None,
        **kwargs,
    ):
        if os.environ.get("UNSLOTH_USE_NEW_MODEL", "0") == "1":
            # Check for other PEFT args in kwargs
            for (peft_arg, flag) in (
                ("finetune_vision_layers", False),
                ("finetune_language_layers", True),
                ("finetune_attention_modules", True),
                ("finetune_mlp_modules", True),
            ):
                if peft_arg not in kwargs: kwargs[peft_arg] = flag
            return FastBaseModel.get_peft_model(
                model                      = model,
                r                          = r,
                target_modules             = target_modules,
                lora_alpha                 = lora_alpha,
                lora_dropout               = lora_dropout,
                bias                       = bias,
                layers_to_transform        = layers_to_transform,
                layers_pattern             = layers_pattern,
                use_gradient_checkpointing = use_gradient_checkpointing,
                random_state               = random_state,
                max_seq_length             = max_seq_length,
                use_rslora                 = use_rslora,
                modules_to_save            = modules_to_save,
                init_lora_weights          = init_lora_weights,
                loftq_config               = loftq_config,
                temporary_location         = temporary_location,
                **kwargs,
            )
        pass
        if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
            print("Unsloth: Full finetuning is enabled, so .get_peft_model has no effect")
            return model
        pass
        transformers_set_seed(random_state)

        if use_gradient_checkpointing == "unsloth":
            patch_unsloth_smart_gradient_checkpointing(dtype = model.get_input_embeddings().weight.dtype)

        if type(r) is not int:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be an integer.")
        if r <= 0:
            raise TypeError(f"Unsloth: Rank of {str(r)} must be larger than 0.")

        if isinstance(model, PeftModelForCausalLM) or isinstance(model, PeftModelForSequenceClassification):
            # Check if exactly the same and then pass through!
            assert(hasattr(model, "peft_config"))

            peft_config = model.peft_config["default"].to_dict()
            check_parameters = [
                "r", "lora_alpha", "lora_dropout",
                "bias", "layers_to_transform", "layers_pattern",
                "use_rslora", "init_lora_weights",
            ]
            check_all = True
            for param in check_parameters:
                check_all = check_all and (peft_config[param] == eval(param))
            pass

            # Check save_modules
            old_target_modules = list(peft_config["target_modules"])
            modules_to_save = peft_config["modules_to_save"]
            if modules_to_save is None: modules_to_save = {}
            modules_to_save = list(modules_to_save)
            old_target_modules += modules_to_save

            # Combine all
            new_target_modules = list(target_modules) + \
                list(modules_to_save if modules_to_save is not None else [])

            # Now check!
            new_target_modules = set(new_target_modules)
            check_all = check_all and (
                len(set(old_target_modules) ^ new_target_modules) == 0
            )

            check_all = check_all and (
                (loftq_config == {} or loftq_config is None) and \
                (peft_config["loftq_config"] == {} or peft_config["loftq_config"] is None)
            )

            if check_all:
                # Simply pass through!
                logger.warning(
                    "Unsloth: Already have LoRA adapters! We shall skip this step."
                )

                # Offload!
                # [TODO] First offload lm_head and embed_tokens to CPU (should be disk!!)
                if "embed_tokens" in new_target_modules:
                    print("Unsloth: Training embed_tokens in mixed precision to save VRAM")

                    new_dtype = model.get_input_embeddings().modules_to_save.default.weight.dtype
                    if new_dtype == torch.float16:
                        # See https://github.com/unslothai/unsloth/pull/1200
                        # Tesla T4 must use float32 and not float16
                        new_dtype = torch.float32
                    pass

                    model.get_input_embeddings().modules_to_save.default\
                        .to(device = DEVICE_TYPE, dtype = new_dtype, non_blocking = True)
                    model.get_input_embeddings().modules_to_save.default.requires_grad_(True)

                    # [TODO] Move old embed_tokens to CPU - should be disk!
                    model.get_input_embeddings().original_module\
                        .to(device = "cpu", non_blocking = True)
                    model.get_input_embeddings().original_module.requires_grad_(False)
                pass

                if "lm_head" in new_target_modules:
                    print("Unsloth: Training lm_head in mixed precision to save VRAM")

                    new_dtype = model.get_output_embeddings().modules_to_save.default.weight.dtype
                    if new_dtype == torch.float16:
                        # See https://github.com/unslothai/unsloth/pull/1200
                        # Tesla T4 must use float32 and not float16
                        new_dtype = torch.float32
                    pass

                    model.get_output_embeddings().modules_to_save.default\
                        .to(device = DEVICE_TYPE, dtype = new_dtype, non_blocking = True)
                    model.get_output_embeddings().modules_to_save.default.requires_grad_(True)

                    # [TODO] Move old lm_head to CPU - should be disk!
                    model.get_output_embeddings().original_module\
                        .to(device = "cpu", non_blocking = True)
                    model.get_output_embeddings().original_module.requires_grad_(False)
                pass

                return model
            else:
                raise TypeError(
                    "Unsloth: Your model already has LoRA adapters. Your new parameters are different."
                )
            pass
        pass

        if loftq_config is None: loftq_config = {}

        signature = str(inspect.signature(LoraConfig))
        SUPPORTS_LOFTQ  = "loftq_config" in signature
        SUPPORTS_RSLORA = "use_rslora"   in signature

        if lora_dropout != 0:
            logger.warning_once(
                f"Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = {lora_dropout}.\n"\
                f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
            )
        pass

        if bias != "none":
            logger.warning_once(
                f"Unsloth: bias = `none` is supported for fast patching. You are using bias = {bias}.\n"\
                f"Unsloth will patch all other layers, except LoRA matrices, causing a performance hit."
            )
        pass

        if not (type(init_lora_weights) is bool or \
            init_lora_weights == "gaussian" or init_lora_weights == "loftq"):
            raise ValueError(
                'Unsloth: `init_lora_weights` must be either [True, False, "gaussian", "loftq"].'
            )
        pass

        if init_lora_weights == "loftq":

            if not SUPPORTS_LOFTQ:
                import peft
                raise RuntimeError(
                    f"Unsloth: Your PEFT version of {peft.__version__} does not support LoftQ init.\n"\
                    "Please install PEFT 0.7.2 or higher.\n"\
                    "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
                )
            pass

            if loftq_config == {}:
                from peft import LoftQConfig
                logger.warning_once(
                    "Unsloth: init_lora_weights = `loftq` is set, but `loftq_config` is None.\n"\
                    "We shall use `loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)`."
                )
                loftq_config = LoftQConfig(loftq_bits = 4, loftq_iter = 1)
            pass

            if hasattr(model.config, "quantization_config"):
                raise ValueError(
                    "Unsloth: You are using `loftq` init, yet `load_in_4bit = True` was set.\n"\
                    "Reload your model without any quantization by setting `load_in_4bit = False`."
                )
            pass
        pass

        assert(type(use_rslora) is bool)
        if use_rslora:
            if not SUPPORTS_RSLORA:
                # We manually check for PEFT
                import peft
                raise RuntimeError(
                    f"Unsloth: Your PEFT version of {peft.__version__} does not support `use_rslora`.\n"\
                    "Please install PEFT 0.7.2 or higher.\n"\
                    "You can also install from source: `pip install git+https://github.com/huggingface/peft.git"
                )
            pass
        pass

        accepted_modules = frozenset(("q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj",),)
        model.config.update({"unsloth_version" : __version__})

        if type(modules_to_save) is tuple:
            modules_to_save = list(modules_to_save)
        pass

        train_lm_head = False
        train_embed_tokens = False
        train_thinking_residual = False
        final_modules = []
        for module in target_modules:
            if module == "lm_head":
                # logger.warning_once(
                #     "Unsloth: `lm_head` should be placed in `modules_to_save` and not `target_modules`. "\
                #     "Luckily, we shall do it for you!"
                # )
                train_lm_head = True
                if modules_to_save is None: modules_to_save = ["lm_head"]
                else: modules_to_save.append("lm_head")

            elif module == "embed_tokens":
                # logger.warning_once(
                #     "Unsloth: `embed_tokens` should be placed in `modules_to_save` and not `target_modules`. "\
                #     "Luckily, we shall do it for you!"
                # )
                train_embed_tokens = True
                if modules_to_save is None: modules_to_save = ["embed_tokens"]
                else: modules_to_save.append("embed_tokens")

            elif "thinking_residual" in module:
                train_thinking_residual = True
                if modules_to_save is None: modules_to_save = [module]
                else: modules_to_save = list(set(modules_to_save).add(module))
            else:
                try:
                    assert(module in accepted_modules)
                    final_modules.append(module)
                except AssertionError as e:
                    final_modules.append(module)
                    print(
                        "Unsloth: You added custom modules, but Unsloth hasn't optimized for this.\n"\
                        "Beware - your finetuning might be noticeably slower!"
                    )
                pass
            pass
        pass

        # Check if we added new tokens!
        if hasattr(model, "_need_to_train_embeddings"):
            if not train_lm_head or not train_embed_tokens:
                print(
                    "Unsloth: You added new tokens but did not specify if you wanted to "\
                    "train the lm_head and embed_tokens.\nWe must turn it on for you."
                )
                train_lm_head = True
                train_embed_tokens = True

                if modules_to_save is None: modules_to_save = ["embed_tokens"]
                else: modules_to_save.append("embed_tokens")

                if modules_to_save is None: modules_to_save = ["lm_head"]
                else: modules_to_save.append("lm_head")
            pass
        pass

        # Check for Llama-3
        # if hasattr(model._saved_temp_tokenizer, "_using_llama3_template"):
        #     if not train_embed_tokens and not train_lm_head:
        #         raise RuntimeError("")

        # First fix untrained tokens
        # Wrong - can cause reserved tokens to pop out!!
        # if train_embed_tokens or train_lm_head:
        #     fix_untrained_tokens(model, eps = 1e-16)
        # pass

        # Check modules_to_save
        if modules_to_save is not None:
            for module in modules_to_save:
                if module == "lm_head":
                    train_lm_head = True
                elif module == "embed_tokens":
                    train_embed_tokens = True
                elif "thinking_residual" in module:
                    train_thinking_residual = True
                else:
                    raise TypeError(
                        f"Unsloth: Module = {module} is not allowed. Only 'lm_head', 'embed_tokens' "
                        "and 'thinking_residual' components are allowed."
                    )
            pass
        pass
        if isinstance(modules_to_save, (tuple, list)):
            modules_to_save = list(set(modules_to_save))
        pass

        vllm_engine = None
        if hasattr(model, "vllm_engine"):
            # Fast inference!
            vllm_engine = model.vllm_engine
            vllm_fast_generate = model.fast_generate
            vllm_fast_generate_batches = model.fast_generate_batches

            if modules_to_save is not None:
                raise NotImplementedError("Unsloth: Currently fast inference does not work with training embeddings or lm_head.")

            if bias != "none":
                raise NotImplementedError("Unsloth: Currently fast inference does not work with using biases for LoRA.")
        pass

        # Does not get lora yet, so get name from model, not base model
        is_classification = "Classification" in str(type(model))

        arguments = dict(
            r                   = r,
            lora_alpha          = lora_alpha,
            target_modules      = final_modules,
            lora_dropout        = lora_dropout,
            bias                = bias,
            task_type           = TaskType.CAUSAL_LM if not is_classification else TaskType.SEQ_CLS,
            layers_to_transform = layers_to_transform,
            init_lora_weights   = init_lora_weights,
            loftq_config        = loftq_config,
            use_rslora          = use_rslora,
            modules_to_save     = modules_to_save,
            **kwargs,
        )
        if not SUPPORTS_LOFTQ:  del arguments["loftq_config"]
        if not SUPPORTS_RSLORA: del arguments["use_rslora"]

        _saved_temp_tokenizer = model._saved_temp_tokenizer

        lora_config = LoraConfig(**arguments)
        # First offload lm_head and embed_tokens to disk
        input_embeddings_device  = model.get_input_embeddings().weight.device
        if is_classification:
             output_embeddings_device = model.score.weight.device
        else:
            output_embeddings_device = model.get_output_embeddings().weight.device

        if use_gradient_checkpointing == "unsloth":
            if train_embed_tokens:
                print("Unsloth: Offloading input_embeddings to disk to save VRAM")
                offload_input_embeddings(model, temporary_location)
            pass

            # Remove old items to save VRAM
            for _ in range(3):
                gc.collect()
                clean_gpu_cache()
            pass

            if train_lm_head:
                print("Unsloth: Offloading output_embeddings to disk to save VRAM")
                offload_output_embeddings(model, temporary_location)
            pass

            # Remove old items to save VRAM
            for _ in range(3):
                gc.collect()
                clean_gpu_cache()
            pass
        pass

        model = _get_peft_model(model, lora_config)
        # Fix LoraConfig.auto_mapping is None
        fix_lora_auto_mapping(model)

        # Apply QAT + LoRA if specified
        if qat_scheme is not None:
            print("Unsloth: Applying QAT to mitigate quantization degradation")
            model = _prepare_model_for_qat(model, qat_scheme)
        pass

        model._saved_temp_tokenizer = _saved_temp_tokenizer

        model = HRPOFastLlamaModel.patch_peft_model(model, use_gradient_checkpointing)

        if train_embed_tokens:
            print("Unsloth: Training embed_tokens in mixed precision to save VRAM")
            assert(hasattr(model.get_input_embeddings(), "modules_to_save"))

            new_dtype = model.get_input_embeddings().modules_to_save.default.weight.dtype
            if new_dtype == torch.float16:
                # See https://github.com/unslothai/unsloth/pull/1200
                # Tesla T4 must use float32 and not float16
                new_dtype = torch.float32
            pass

            model.get_input_embeddings().modules_to_save.default\
                .to(device = DEVICE_TYPE, dtype = new_dtype, non_blocking = True)
            model.get_input_embeddings().modules_to_save.default.requires_grad_(True)
        pass

        if train_lm_head:
            print("Unsloth: Training lm_head in mixed precision to save VRAM")
            assert(hasattr(model.get_output_embeddings(), "modules_to_save"))

            new_dtype = model.get_output_embeddings().modules_to_save.default.weight.dtype
            if new_dtype == torch.float16:
                # See https://github.com/unslothai/unsloth/pull/1200
                # Tesla T4 must use float32 and not float16
                new_dtype = torch.float32
            pass

            model.get_output_embeddings().modules_to_save.default\
                .to(device = DEVICE_TYPE, dtype = new_dtype, non_blocking = True)
            model.get_output_embeddings().modules_to_save.default.requires_grad_(True)
        pass

        if train_thinking_residual:
            print("Unsloth: Training thinking_residual in mixed precision to save VRAM")
            try:
                new_dtype = model.get_input_embeddings().weight.dtype
            except:
                new_dtype = model.get_input_embeddings().modules_to_save.default.weight.dtype

            for module in modules_to_save:
                if "thinking_residual_gate_r" in module:
                    assert(hasattr(model.model.model.thinking_residual_gate_r, "modules_to_save"))
                    model.model.model.thinking_residual_gate_r.modules_to_save.default\
                        .to(device = "cuda", dtype = new_dtype, non_blocking = True)
                    model.model.model.thinking_residual_gate_r.modules_to_save.default.requires_grad_(True)
                if "thinking_residual_gate_i" in module:
                    assert(hasattr(model.model.model.thinking_residual_gate_i, "modules_to_save"))
                    model.model.model.thinking_residual_gate_i.modules_to_save.default\
                        .to(device = "cuda", dtype = new_dtype, non_blocking = True)
                    model.model.model.thinking_residual_gate_i.modules_to_save.default.requires_grad_(True)
                if "thinking_residual_Lambda" in module:
                    model.model.model.thinking_residual_Lambda.modules_to_save.default\
                        .to(device = "cuda", dtype = torch.float32, non_blocking = True)
                    model.model.model.thinking_residual_Lambda.modules_to_save.default.requires_grad_(True)

        # Patch tokenizer to pad to the right
        internal_model = model
        while hasattr(internal_model, "model"):
            if hasattr(internal_model, "_saved_temp_tokenizer"):
                internal_model._saved_temp_tokenizer.padding_side = "right"
            pass
            # Also set is_loaded_in_8bit to disable incorrect DDP
            internal_model.is_loaded_in_8bit = True
            internal_model = internal_model.model
        pass
        if hasattr(internal_model, "_saved_temp_tokenizer"):
            internal_model._saved_temp_tokenizer.padding_side = "right"
        pass
        # Also set is_loaded_in_8bit to disable incorrect DDP
        internal_model.is_loaded_in_8bit = True

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            clean_gpu_cache()
        pass

        patch_peft_fast_inference(model)

        # Add for_inference and for_training
        model.for_training  = functools.partial(HRPOFastLlamaModel.for_training,  model)
        model.for_inference = functools.partial(HRPOFastLlamaModel.for_inference, model)
        m = model
        while hasattr(m, "model"):
            m.for_training  = functools.partial(FastBaseModel.for_training,  m)
            m.for_inference = functools.partial(FastBaseModel.for_inference, m)
            m = m.model
        return model
    pass