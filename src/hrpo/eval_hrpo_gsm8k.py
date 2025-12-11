import json
import os
import sys
from datetime import datetime

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src", "external", "transformers", "src"))

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm

from src.external.transformers.src.transformers import GenerationConfig
from src.external.transformers.src.transformers.models.auto import AutoTokenizer
from src.external.transformers.src.transformers.models.qwen2.modeling_qwen2 import (
    HRPOQwen2ForCausalLM,
)
from src.hrpo.sanity_guards import check_adapter_sanity
from src.hrpo.utils import (
    ANSWER_START,
    SYSTEM_PROMPT,
    extract_from_response,
    extract_hash_answer,
    process_gsm8k_answer,
)


def evaluate_model(
    model_path: str,
    adapter_path: str,
    temperature: float,
    is_inference: bool,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = HRPOQwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.answer_start = ANSWER_START

    model = PeftModel.from_pretrained(model, adapter_path)
    print(f"Loaded adapter from {adapter_path}")
    check_adapter_sanity(adapter_path)
    
    if hasattr(model.base_model.model.model, "latent_gate_a"):
        print("Latent gate parameters present in model.")
        
    model.eval()

    dataset = load_dataset("openai/gsm8k", "main")["test"]
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    results = []
    correct = 0
    total = 0

    progress_bar = tqdm(
        total=total_samples,
        desc="Processing samples",
        unit="examples",
        dynamic_ncols=True,
    )
    progress_bar.set_postfix({"acc": "0.00%", "correct": "0"})

    # Process samples in batches
    for i in range(0, total_samples, batch_size):
        batch_data = dataset[i : i + batch_size]
        current_batch_size = len(batch_data["question"])

        # Prepare prompts using the same format as training
        prompts = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q.strip()},
            ]
            for q in batch_data["question"]
        ]

        # Convert chat prompts to the required format
        formatted_prompts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]

        prompt_inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )
        prompt_ids = prompt_ids.to(model.device)
        prompt_mask = prompt_mask.to(model.device)
        prompt_length = prompt_ids.size(1)

        # Generate responses
        outputs, _, _ = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            generation_config=GenerationConfig(
                do_sample=not is_inference,  # greedy decoding if is_inference=True, sampling otherwise
                temperature=temperature,
                max_new_tokens=512,
            ),
            processing_class=tokenizer,
        )

        # Process each generated response
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[prompt_length:])
            response = response.split(tokenizer.special_tokens_map["eos_token"])[0]

            # Extract the generated answer using XML tags
            extracted = extract_from_response(response)
            generated_answer = process_gsm8k_answer(extracted)
            true_answer = extract_hash_answer(batch_data["answer"][j])
            true_answer = process_gsm8k_answer(true_answer)
            print(generated_answer, true_answer, generated_answer == true_answer)

            # Store the result
            result = {
                "question": batch_data["question"][j],
                "true_answer": true_answer,
                "generated_answer": generated_answer,
                "full_response": response,
                "correct": generated_answer == true_answer,
            }
            results.append(result)

            if generated_answer == true_answer:
                correct += 1
            total += 1

        progress_bar.update(current_batch_size)
        progress_bar.set_postfix(
            {
                "acc": f"{(correct/total)*100:.2f}%",
                "correct": f"{correct}/{total}",
            }
        )

    progress_bar.close()
    accuracy = correct / total if total > 0 else 0
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "model_path": adapter_path,
        "timestamp": datetime.now().isoformat(),
    }

    if save_results:
        save_path = adapter_path + "/eval_results.json"
        with open(save_path, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    base_model = None
    checkpoint_path = args.checkpoint_path
    base_models = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
    for model in base_models:
        if model.split("/")[-1] in checkpoint_path:
            base_model = model
    temperature = args.temperature
    print(checkpoint_path, base_model, temperature)

    if "eval_results.json" not in os.listdir(checkpoint_path):
        print(f"Starting GSM8k evaluation on {checkpoint_path}")
        metrics = evaluate_model(
            model_path=base_model,
            adapter_path=checkpoint_path,
            temperature=temperature,
            is_inference=args.greedy,
            batch_size=args.batch_size,
            num_samples=None,
            save_results=True,
        )
