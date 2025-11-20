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
        raise ValueError("Unsloth: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
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
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

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