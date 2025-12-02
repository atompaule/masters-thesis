import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath("src/external/transformers/src"))

from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model

from src.external.transformers.src.transformers.models.llama.modeling_llama import (
    HRPOLlamaForCausalLM,
)
from src.external.transformers.src.transformers.models.qwen2.modeling_qwen2 import (
    HRPOQwen2ForCausalLM,
)
from src.external.transformers.src.transformers.models.qwen2.tokenization_qwen2 import (
    Qwen2Tokenizer,
)
from src.external.transformers.src.transformers.models.auto import AutoTokenizer
from src.external.trl.trl.trainer.hrpo_trainer import GRPOConfig, HRPOTrainer
from src.hrpo.patch import patch_trainer_optimizer
from src.hrpo.utils import (
    ANSWER_START,
    get_reward_func,
    process_gsm8k,
    process_gsm8k_answer,
)

os.environ["WANDB_PROJECT"] = "masters-thesis"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def preprocess_gsm8k(split="train", chunk_size=1000) -> Dataset:
    dataset = load_dataset("openai/gsm8k", "main")[split]
    return dataset.map(
        process_gsm8k, batched=True, batch_size=chunk_size, load_from_cache_file=False
    )


def main(args):
    logger.info(
        f"Starting experiment {args.model_name} with group size {args.group_size}"
    )
    exp_name = (
        f"./experiments/{args.model_name.split('/')[-1]}-gsm8k-group{args.group_size}"
        f"-lora{args.lora_rank}-rmin{args.residual_r_min}-temp{args.temperature}"
    )
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Exiting...")
        exit()

    # tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen-tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_prompt_length + args.max_completion_length

    model = HRPOQwen2ForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.answer_start = ANSWER_START

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            modules_to_save=[
                "latent_gate_r",
                "latent_gate_i",
                "latent_gate_a",
            ],
        ),
    )

    model.model.model.latent_gate_a.reset_lambda_parameters(
        r_min=args.residual_r_min,
        r_max=args.residual_r_max,
    )

    training_args = GRPOConfig(
        use_vllm=False,
        learning_rate=args.lr,
        beta=args.beta,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optimizer,
        max_grad_norm=args.max_grad_norm,
        logging_steps=1,
        bf16=True,
        temperature=args.temperature,
        num_generations=args.group_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=1,
        save_steps=250,
        save_total_limit=3,
        report_to="wandb",
        output_dir=exp_name,
    )

    dataset = preprocess_gsm8k("train", chunk_size=500)
    trainer = HRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            get_reward_func(process_gsm8k_answer),
        ],
        args=training_args,
        train_dataset=dataset,
    )
    patch_trainer_optimizer(
        trainer,
        args.lr_residual_gate,
        args.lr_residual_Lambda,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank", type=int, default=32)

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--residual_r_min", type=float, default=0.99)
    parser.add_argument("--residual_r_max", type=float, default=0.999)
    parser.add_argument("--lr_residual_gate", type=float, default=1e-4)
    parser.add_argument("--lr_residual_Lambda", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"

    main(args)
