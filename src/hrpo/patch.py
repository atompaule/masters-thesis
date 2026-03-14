import types

from transformers.trainer import is_sagemaker_mp_enabled, logger, nn

try:
    if is_sagemaker_mp_enabled():
        import smdistributed.modelparallel.torch as smp  # type: ignore
    else:
        smp = None
except ImportError:
    smp = None


def patch_trainer_optimizer(trainer, lr_latent_gates=1e-4, lr_latent_gate_Lambda=1e-3):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        print("DEBUG: PATCHED create_optimizer CALLED")
        from src.external.transformers.src.transformers.models.qwen2.modeling_qwen2 import (
            LatentGateA,
        )

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            "latent_gate" not in n
                            and n in decay_parameters
                            and p.requires_grad
                        )
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            "latent_gate" not in n
                            and n not in decay_parameters
                            and p.requires_grad
                        )
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            ("latent_gate_r" in n or "latent_gate_i" in n)
                            and p.requires_grad
                        )
                    ],
                    "lr": lr_latent_gates,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if ("latent_gate_a" in n and p.requires_grad)
                    ],
                    "lr": lr_latent_gate_Lambda,
                    "weight_decay": self.args.weight_decay,
                },
            ]

            print("DEBUG: param groups summary")
            for i, g in enumerate(optimizer_grouped_parameters):
                param_ids = {id(p) for p in g["params"]}
                names = [
                    n for n, p in opt_model.named_parameters() if id(p) in param_ids
                ]
                print(
                    f"group {i}: lr={g['lr']}, wd={g.get('weight_decay')}, count={len(g['params'])}"
                )
                print("  sample:", names[:5])

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args, opt_model
                )

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        if is_sagemaker_mp_enabled() and smp is not None:
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    trainer._old_create_optimizer = trainer.create_optimizer
    trainer.create_optimizer = types.MethodType(create_optimizer, trainer)
