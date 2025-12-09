import logging

import torch
from transformers import TrainerCallback

import wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class LambdaMonitoringCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        _model = model.base_model.model.base_model

        # latent_gate_a is a ModulesToSaveWrapper
        # Lambda is a Parameter
        lambda_param = _model.latent_gate_a.Lambda
        stats = {
            "lambda/mean": lambda_param.mean().item(),
            "lambda/std": lambda_param.std().item(),
            "lambda/min": lambda_param.min().item(),
            "lambda/max": lambda_param.max().item(),
        }

        if lambda_param.grad is not None:
            stats.update(
                {
                    "lambda/grad_mean": lambda_param.grad.mean().item(),
                    "lambda/grad_std": lambda_param.grad.std().item(),
                    "lambda/grad_min": lambda_param.grad.min().item(),
                    "lambda/grad_max": lambda_param.grad.max().item(),
                }
            )
        else:
            stats["lambda/grad_mean"] = 0.0

        wandb.log(stats)
