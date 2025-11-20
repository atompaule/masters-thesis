from .llama import HRPOFastLlamaModel
from .qwen2 import HRPOFastQwen2Model
from unsloth.models.loader import *
from unsloth_zoo.utils import _get_dtype

class HRPOFastLanguageModel(HRPOFastLlamaModel):
    @staticmethod
    def from_pretrained(
        model_name                 = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length             = 2048,
        dtype                      = None,
        load_in_4bit               = True,  # 4bit QLoRA
        load_in_8bit               = False, # 8bit  LoRA
        load_in_16bit              = False, # 16bit LoRA
        full_finetuning            = False,
        token                      = None,
        device_map                 = "sequential",
        rope_scaling               = None,
        fix_tokenizer              = True,
        trust_remote_code          = False,
        use_gradient_checkpointing = "unsloth",
        resize_model_vocab         = None,
        revision                   = None,
        use_exact_model_name       = False,
        offload_embedding          = False,

        fast_inference             = False, # uses vLLM
        gpu_memory_utilization     = 0.5,
        float8_kv_cache            = False,
        random_state               = 3407,
        max_lora_rank              = 64,
        disable_log_stats          = True,
        qat_scheme                 = None,
        *args, **kwargs,
    ):
        # Login to allow private models
        if token is None: token = get_token()
        if token is not None:
            try:
                from huggingface_hub import login
                login(token = token)
            except:
                pass
        if load_in_8bit or full_finetuning or qat_scheme is not None:
            return FastModel.from_pretrained(
                model_name                 = model_name,
                max_seq_length             = max_seq_length,
                dtype                      = dtype,
                load_in_4bit               = load_in_4bit,
                load_in_8bit               = load_in_8bit,
                load_in_16bit              = load_in_16bit,
                full_finetuning            = full_finetuning,
                token                      = token,
                device_map                 = device_map,
                rope_scaling               = rope_scaling, # [TODO] No effect
                fix_tokenizer              = fix_tokenizer, # [TODO] No effect
                trust_remote_code          = trust_remote_code,
                use_gradient_checkpointing = use_gradient_checkpointing,
                resize_model_vocab         = resize_model_vocab, # [TODO] No effect
                revision                   = revision,
                return_logits              = False, # Return logits
                fullgraph                  = True, # No graph breaks
                use_exact_model_name       = use_exact_model_name,
                offload_embedding          = offload_embedding,

                # Pass vLLM/inference parameters
                fast_inference             = fast_inference,
                gpu_memory_utilization     = gpu_memory_utilization,
                float8_kv_cache            = float8_kv_cache,
                random_state               = random_state,
                max_lora_rank              = max_lora_rank,
                disable_log_stats          = disable_log_stats,

                qat_scheme                 = qat_scheme,
                *args, **kwargs,
            )
        pass

        if token is None: token = get_token()
        if isinstance(dtype, str) and dtype in ["float16", "bfloat16"]:
            dtype = getattr(torch, dtype)
        assert (dtype is None or dtype == torch.float16 or dtype == torch.bfloat16
                or dtype == torch.float32)

        if fast_inference:
            if importlib.util.find_spec("vllm") is None:
                raise ImportError(
                    "Unsloth: Please install vLLM before enabling `fast_inference`!\n"\
                    "You can do this in a terminal via `pip install vllm`"
                )
            pass
        pass

        old_model_name = model_name
        if not use_exact_model_name:
            model_name = get_model_name(model_name, load_in_4bit)

        if USE_MODELSCOPE and not os.path.exists(model_name):
            from modelscope import snapshot_download
            model_name = snapshot_download(model_name)
        pass

        # First check if it's a normal model via AutoConfig
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars, are_progress_bars_disabled
        was_disabled = are_progress_bars_disabled()
        disable_progress_bars()

        autoconfig_error = None
        peft_error = None
        model_config = None
        peft_config = None
        try:
            model_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                revision = revision,
                trust_remote_code = trust_remote_code,
            )
            is_model = True
        except Exception as error:
            autoconfig_error = str(error)
            is_model = False
        try:
            peft_config = PeftConfig.from_pretrained(
                model_name,
                token = token,
                revision = revision,
                trust_remote_code = trust_remote_code,
            )
            is_peft = True
        except Exception as error:
            peft_error = str(error)
            is_peft = False
        pass

        # Old transformers versions check
        both_exist = (is_model and is_peft) and not SUPPORTS_LLAMA32

        # Error out if both LoRA and normal model config exists.
        if both_exist:
            raise RuntimeError(
                "Unsloth: Your repo has a LoRA adapter and a base model.\n"\
                "You have 2 files `config.json` and `adapter_config.json`.\n"\
                "We must only allow one config file.\n"\
                "Please separate the LoRA and base models to 2 repos."
            )
        model_types = get_transformers_model_type(
            peft_config if peft_config is not None else model_config
        )
        if len(model_types) == 1:
            model_type = model_types[0]
        else:
            # Leave as tuple if more than one arch
            model_type = model_types

        # New transformers need to check manually.
        if SUPPORTS_LLAMA32:
            # Check if folder exists locally
            if os.path.isdir(model_name):
                exist_adapter_config = os.path.exists(os.path.join(model_name, "adapter_config.json"))
                exist_config         = os.path.exists(os.path.join(model_name, "config.json"))
                both_exist = exist_adapter_config and exist_config
            else:
                # Because HfFileSystem assumes linux paths, we need to set the path with forward slashes, even on Windows.
                files = HfFileSystem(token = token).glob(f"{model_name}/*.json")
                files = list(os.path.split(x)[-1] for x in files)
                if sum(x == "adapter_config.json" or x == "config.json" for x in files) >= 2:
                    both_exist = True
                pass
            pass
        pass

        if not is_model and not is_peft:
            error = autoconfig_error if autoconfig_error is not None else peft_error
            # Old transformers version
            if "rope_scaling" in error.lower() and not SUPPORTS_LLAMA31:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support new RoPE scaling methods.\n"\
                    f"This includes Llama 3.1. The minimum required version is 4.43.2\n"\
                    f'Try `pip install --upgrade "transformers>=4.43.2"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                )
            # Create a combined error message showing both failures
            combined_error = (
                "Unsloth: Failed to load model. Both AutoConfig and PeftConfig loading failed.\n\n"
                f"AutoConfig error: {autoconfig_error}\n\n"
                f"PeftConfig error: {peft_error}\n\n"
            )
            raise RuntimeError(combined_error)
        pass

        # Get base model for PEFT:
        if is_peft:
            # Check base model again for PEFT
            model_name = peft_config.base_model_name_or_path
            if not use_exact_model_name:
                model_name = get_model_name(model_name, load_in_4bit)
            model_config = AutoConfig.from_pretrained(
                model_name,
                token = token,
                trust_remote_code = trust_remote_code,
            )
        pass

        if not was_disabled: enable_progress_bars()

        if model_type == "llama":
            scaling_type = None
            if getattr(model_config, "rope_scaling", None) is not None:
                scaling_type1 = model_config.rope_scaling.get("type", None)
                scaling_type2 = model_config.rope_scaling.get("rope_type", None)
                scaling_type = scaling_type1 if scaling_type1 is not None else scaling_type2
            pass

            if scaling_type == "llama3" and not SUPPORTS_LLAMA31:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Llama 3.1.\n"\
                    f"The minimum required version is 4.43.2\n"\
                    f'Try `pip install --upgrade "transformers>=4.43.2"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                )

            dispatch_model = HRPOFastLlamaModel

        elif model_type == "mistral": dispatch_model = FastMistralModel
        elif model_type == "gemma":
            if not SUPPORTS_GEMMA:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Gemma.\n"\
                    f"The minimum required version is 4.38.\n"\
                    f'Try `pip install --upgrade "transformers>=4.38"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                )
            dispatch_model = FastGemmaModel
        elif model_type == "gemma2":
            if not SUPPORTS_GEMMA2:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Gemma2.\n"\
                    f"The minimum required version is 4.42.3.\n"\
                    f'Try `pip install --upgrade "transformers>=4.42.3"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                )
            # Also check for softcapping support in flash-attn which is faster!
            if is_bfloat16_supported() and not HAS_FLASH_ATTENTION:
                print(
                    "Unsloth: If you want to finetune Gemma 2, install flash-attn to make it faster!\n"\
                    "To install flash-attn, do the below:\n"\
                    '\npip install --no-deps --upgrade "flash-attn>=2.6.3"'
                )
            elif HAS_FLASH_ATTENTION and not HAS_FLASH_ATTENTION_SOFTCAPPING:
                print(
                    "Unsloth: If you want to finetune Gemma 2, upgrade flash-attn to version 2.6.3 or higher!\n"\
                    "Newer versions support faster and less memory usage kernels for Gemma 2's attention softcapping!\n"\
                    "To update flash-attn, do the below:\n"\
                    '\npip install --no-deps --upgrade "flash-attn>=2.6.3"'
                )

            dispatch_model = FastGemma2Model
        elif model_type == "qwen2":
            dispatch_model = HRPOFastQwen2Model
        elif model_type == "qwen3":# or model_type == "qwen3_moe":
            if not SUPPORTS_QWEN3 or not SUPPORTS_QWEN3_MOE:
                raise ImportError(
                    f"Unsloth: Your transformers version of {transformers_version} does not support Qwen3.\n"\
                    f"The minimum required version is 4.50.3.\n"\
                    f'Try `pip install --upgrade "transformers>=4.50.3"`\n'\
                    f"to obtain the latest transformers build, then restart this session."\
                )
            dispatch_model = FastQwen3Model if model_type == "qwen3" else FastQwen3MoeModel
        # elif model_type == "falcon_h1":
        #     dispatch_model = FastFalconH1Model
        #     if not SUPPORTS_FALCON_H1:
        #         raise ImportError(
        #             f"Unsloth: Your transformers version of {transformers_version} does not support FalconH1.\n"\
        #             f"The minimum required version is 4.50.3.\n"\
        #             f'Try `pip install --upgrade "transformers>=4.50.3"`\n'\
        #             f"to obtain the latest transformers build, then restart this session."\
        #         )
        # Temporary disable optimized Cohere until errors match
        # elif model_type == "cohere":
        #     dispatch_model = FastCohereModel
        # Temporary disable optimized Granite until errors match
        # elif model_type == "granite":
        #     dispatch_model = FastGraniteModel
        else:
            return FastModel.from_pretrained(
                model_name                 = old_model_name,
                max_seq_length             = max_seq_length,
                dtype                      = dtype,
                load_in_4bit               = load_in_4bit,
                load_in_8bit               = load_in_8bit,
                load_in_16bit              = load_in_16bit,
                full_finetuning            = full_finetuning,
                token                      = token,
                device_map                 = device_map,
                rope_scaling               = rope_scaling, # [TODO] No effect
                fix_tokenizer              = fix_tokenizer, # [TODO] No effect
                trust_remote_code          = trust_remote_code,
                use_gradient_checkpointing = use_gradient_checkpointing,
                resize_model_vocab         = resize_model_vocab, # [TODO] No effect
                revision                   = revision,
                return_logits              = False, # Return logits
                fullgraph                  = True, # No graph breaks
                use_exact_model_name       = use_exact_model_name,
                offload_embedding          = offload_embedding,

                # Pass vLLM/inference parameters
                fast_inference             = fast_inference,
                gpu_memory_utilization     = gpu_memory_utilization,
                float8_kv_cache            = float8_kv_cache,
                random_state               = random_state,
                max_lora_rank              = max_lora_rank,
                disable_log_stats          = disable_log_stats,

                *args, **kwargs,
            )
        pass

        if use_gradient_checkpointing == "unsloth":
            patch_unsloth_smart_gradient_checkpointing(dtype = dtype)

        # Check if this is local model since the tokenizer gets overwritten
        if  os.path.exists(os.path.join(old_model_name, "tokenizer_config.json")) and \
            os.path.exists(os.path.join(old_model_name, "tokenizer.json")) and \
            os.path.exists(os.path.join(old_model_name, "special_tokens_map.json")):

            tokenizer_name = old_model_name
        else:
            tokenizer_name = kwargs.pop("tokenizer_name", None)
        pass

        if fast_inference:
            fast_inference, model_name = fast_inference_setup(model_name, model_config)

        model, tokenizer = dispatch_model.from_pretrained(
            model_name        = model_name,
            max_seq_length    = max_seq_length,
            dtype             = _get_dtype(dtype),
            load_in_4bit      = load_in_4bit,
            token             = token,
            device_map        = device_map,
            rope_scaling      = rope_scaling,
            fix_tokenizer     = fix_tokenizer,
            model_patcher     = dispatch_model,
            tokenizer_name    = tokenizer_name,
            trust_remote_code = trust_remote_code,
            revision          = revision if not is_peft else None,

            fast_inference    = fast_inference,
            gpu_memory_utilization = gpu_memory_utilization,
            float8_kv_cache   = float8_kv_cache,
            random_state      = random_state,
            max_lora_rank     = max_lora_rank,
            disable_log_stats = disable_log_stats,
            *args, **kwargs,
        )

        if resize_model_vocab is not None:
            model.resize_token_embeddings(resize_model_vocab)
        pass

        # In case the model supports tagging, add the unsloth tag.
        if hasattr(model, "add_model_tags"):
            model.add_model_tags(["unsloth",])
        pass
        if hasattr(tokenizer, "add_model_tags"):
            tokenizer.add_model_tags(["unsloth",])
        pass

        if load_in_4bit:
            # Fix up bitsandbytes config
            compute_dtype = dtype_from_config(model.config)
            quantization_config = \
            {
                # Sometimes compute_dtype is not a string!!
                "bnb_4bit_compute_dtype"           : compute_dtype,
                "bnb_4bit_quant_type"              : "nf4",
                "bnb_4bit_use_double_quant"        : True,
                "llm_int8_enable_fp32_cpu_offload" : False,
                "llm_int8_has_fp16_weight"         : False,
                "llm_int8_skip_modules"            : None,
                "llm_int8_threshold"               : 6.0,
                "load_in_4bit"                     : True,
                "load_in_8bit"                     : False,
                "quant_method"                     : "bitsandbytes",
            }
            model.config.update({"quantization_config" : quantization_config})
        pass

        if is_peft:
            # From https://github.com/huggingface/peft/issues/184
            # Now add PEFT adapters
            model = PeftModel.from_pretrained(
                model,
                old_model_name,
                token = token,
                revision = revision,
                is_trainable = True,
                trust_remote_code = trust_remote_code,
            )
            # Patch it as well!
            model = dispatch_model.patch_peft_model(model, use_gradient_checkpointing)
        pass
        return model, tokenizer
    pass
pass