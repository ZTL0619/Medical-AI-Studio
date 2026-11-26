"""Build PEFT/QLoRA configuration objects lazily."""


def build_lora_config(config):
    from peft import LoraConfig

    return LoraConfig(
        r=int(config.lora_rank),
        lora_alpha=float(config.lora_alpha),
        lora_dropout=float(config.lora_dropout),
        bias="none",
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
    )
