"""Multimodal LoRA/QLoRA supervised fine-tuning.

Key properties of this training path:
- the image/question/system prompt is context only;
- only assistant-answer tokens contribute to the language-model loss;
- a validation set is used when available;
- the best checkpoint is selected by validation loss and exported as an adapter.
"""
import inspect
import json
import os
from pathlib import Path

from foundation_model.data.jsonl_dataset import load_jsonl_samples
from foundation_model.distributed.deepspeed import write_zero2_config
from foundation_model.peft.config import build_lora_config
from foundation_model.runtime.dependencies import require
from foundation_model.training.callbacks import FoundationStatusCallback
from foundation_model.training.status import write_status


def _get_model_class(transformers):
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError("Transformers does not provide a multimodal auto-model class in this version.")


def _dataset_from_path(path):
    require(["datasets"])
    from datasets import Dataset

    samples = load_jsonl_samples(path)
    rows = [
        {
            "image": str(s.image),
            "question": s.question,
            "answer": s.answer,
            "system": s.system,
            "sample_id": s.sample_id or str(i),
        }
        for i, s in enumerate(samples)
    ]
    return Dataset.from_list(rows)


def _build_train_eval_datasets(config):
    train_dataset = _dataset_from_path(config.dataset_path)

    if config.validation_dataset_path:
        eval_dataset = _dataset_from_path(config.validation_dataset_path)
        return train_dataset, eval_dataset, "explicit"

    # The bundled one-row smoke dataset must still be runnable. Real training
    # data with >=2 rows is split deterministically for validation.
    if len(train_dataset) < 2:
        return train_dataset, None, "disabled_too_small"

    split = train_dataset.train_test_split(
        test_size=float(config.validation_split),
        seed=int(config.seed),
        shuffle=True,
    )
    return split["train"], split["test"], "auto_split"


def _mask_assistant_only_labels(labels, prompt_lengths, pad_token_id=None, image_token_ids=()):
    """Mask everything before the assistant answer with -100.

    ``prompt_lengths`` are the token lengths of system + image + user + the
    assistant-generation prefix. Therefore tokens after this boundary are the
    answer (plus its normal end-of-turn/eos token), which remain supervised.
    """
    masked = labels.clone()
    seq_len = int(masked.shape[1])
    for row, length in enumerate(prompt_lengths):
        boundary = min(seq_len, max(0, int(length)))
        masked[row, :boundary] = -100
    if pad_token_id is not None:
        masked[masked == int(pad_token_id)] = -100
    for token_id in image_token_ids:
        masked[masked == int(token_id)] = -100
    return masked


def _build_collator(processor):
    from PIL import Image

    tokenizer = processor.tokenizer

    def _special_image_ids():
        candidates = {
            "<image_soft_token>", "<start_of_image>", "<end_of_image>", "<image>",
        }
        mapping = getattr(tokenizer, "special_tokens_map", {}) or {}
        for key, value in mapping.items():
            if "image" in key or key == "boi_token":
                if isinstance(value, str):
                    candidates.add(value)
                elif isinstance(value, (list, tuple)):
                    candidates.update(str(v) for v in value)
        ids = set()
        unk = getattr(tokenizer, "unk_token_id", None)
        for token in candidates:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                continue
            if isinstance(token_id, int) and token_id >= 0 and token_id != unk:
                ids.add(token_id)
        return ids

    image_token_ids = _special_image_ids()

    def collate_fn(examples):
        full_messages = []
        prompt_messages = []
        images = []

        for example in examples:
            image = Image.open(example["image"]).convert("RGB")
            images.append([image])
            system = example.get("system") or "You are a medical visual assistant."
            context = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["question"]},
                    ],
                },
            ]
            prompt_messages.append(context)
            full_messages.append(
                context
                + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": example["answer"]}],
                    }
                ]
            )

        full_texts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False).strip()
            for msgs in full_messages
        ]
        prompt_texts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True).strip()
            for msgs in prompt_messages
        ]

        batch = processor(images=images, text=full_texts, return_tensors="pt", padding=True)
        prompt_batch = processor(images=images, text=prompt_texts, return_tensors="pt", padding=True)

        attention_mask = prompt_batch.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("Processor did not return attention_mask; cannot build assistant-only labels safely.")
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        labels = _mask_assistant_only_labels(
            batch["input_ids"],
            prompt_lengths,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            image_token_ids=image_token_ids,
        )
        if not bool((labels != -100).any()):
            raise RuntimeError("All SFT labels were masked. Check the model chat template / processor.")
        batch["labels"] = labels
        return batch

    return collate_fn


def _set_supported(kwargs, cls, name, value):
    """Set an argument only when the installed TRL/Transformers version accepts it."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        params = {}
    if name in params:
        kwargs[name] = value
        return True
    return False


def train_foundation_model(config, status_path):
    required = ["transformers", "accelerate", "peft", "datasets", "trl"]
    if config.peft_strategy == "qlora":
        required.append("bitsandbytes")
    if config.distributed_strategy == "deepspeed_zero2":
        required.append("deepspeed")
    require(required)

    import torch
    import transformers
    from peft import prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
    processor = transformers.AutoProcessor.from_pretrained(
        config.base_model,
        trust_remote_code=config.trust_remote_code,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.peft_strategy == "qlora":
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = {"": local_rank}

    model_cls = _get_model_class(transformers)
    model = model_cls.from_pretrained(config.base_model, **model_kwargs)
    if config.peft_strategy == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )

    train_dataset, eval_dataset, validation_mode = _build_train_eval_datasets(config)
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    collator = _build_collator(processor)
    peft_config = build_lora_config(config)

    ds_config_path = None
    if config.distributed_strategy == "deepspeed_zero2":
        ds_config_path = write_zero2_config(config, Path(config.output_dir) / "deepspeed_zero2.json")

    kwargs = dict(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=int(config.per_device_train_batch_size),
        gradient_accumulation_steps=int(config.gradient_accumulation_steps),
        num_train_epochs=float(config.num_train_epochs),
        learning_rate=float(config.learning_rate),
        # warmup_ratio=float(config.warmup_ratio),
        weight_decay=float(config.weight_decay),
        logging_steps=int(config.logging_steps),
        bf16=config.precision == "bf16",
        fp16=config.precision == "fp16",
        gradient_checkpointing=bool(config.gradient_checkpointing),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=1024,
        seed=int(config.seed),
        data_seed=int(config.seed),
        ddp_find_unused_parameters=False,
    )

    if has_eval:
        # Evaluate and save at the same cadence so load_best_model_at_end is valid.
        kwargs["save_strategy"] = "epoch"
        _set_supported(kwargs, SFTConfig, "eval_strategy", "epoch") or _set_supported(
            kwargs, SFTConfig, "evaluation_strategy", "epoch"
        )
        _set_supported(kwargs, SFTConfig, "load_best_model_at_end", True)
        _set_supported(kwargs, SFTConfig, "metric_for_best_model", "eval_loss")
        _set_supported(kwargs, SFTConfig, "greater_is_better", False)
        _set_supported(kwargs, SFTConfig, "save_total_limit", 2)
    else:
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = int(config.save_steps)

    if ds_config_path is not None:
        kwargs["deepspeed"] = str(ds_config_path)

    training_args = SFTConfig(**kwargs)
    callback = FoundationStatusCallback.build(status_path, config)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval else None,
        data_collator=collator,
        processing_class=processor,
        peft_config=peft_config,
        callbacks=[callback],
    )
    result = trainer.train(
        resume_from_checkpoint=str(config.resume_from_checkpoint) if config.resume_from_checkpoint else None
    )

    eval_metrics = trainer.evaluate() if has_eval else {}

    if trainer.is_world_process_zero():
        # When validation is enabled, Trainer reloads the checkpoint with the
        # lowest eval_loss before this save. Export that adapter explicitly.
        adapter_name = "adapter_best" if has_eval else "adapter"
        adapter_dir = Path(config.output_dir) / adapter_name
        trainer.save_model(str(adapter_dir))
        processor.save_pretrained(str(adapter_dir / "processor"))

        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
        best_metric = getattr(trainer.state, "best_metric", None)
        manifest = {
            "schema_version": 2,
            "base_model": config.base_model,
            "adapter_path": str(adapter_dir),
            "best_checkpoint": best_checkpoint,
            "best_eval_loss": float(best_metric) if best_metric is not None else eval_metrics.get("eval_loss"),
            "validation_mode": validation_mode,
            "validation_samples": int(len(eval_dataset)) if has_eval else 0,
            "train_samples": int(len(train_dataset)),
            "peft_strategy": config.peft_strategy,
            "distributed_strategy": config.distributed_strategy,
            "world_size": config.world_size,
            "global_batch_size": config.global_batch_size,
            "trainable_parameters": int(trainable),
            "total_parameters": int(total),
            "trainable_ratio": float(trainable / total) if total else 0.0,
            "train_metrics": result.metrics,
            "eval_metrics": eval_metrics,
            "config": config.to_json_dict(),
        }
        (Path(config.output_dir) / "training_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_status(
            status_path,
            state="finalizing",
            progress=1.0,
            global_step=int(trainer.state.global_step),
            max_steps=int(trainer.state.max_steps or 0),
            world_size=config.world_size,
            global_batch_size=config.global_batch_size,
            base_model=config.base_model,
            peft_strategy=config.peft_strategy,
            distributed_strategy=config.distributed_strategy,
            eval_loss=eval_metrics.get("eval_loss"),
            best_eval_loss=manifest["best_eval_loss"],
            best_checkpoint=best_checkpoint,
            adapter_path=str(adapter_dir),
        )
    return result
