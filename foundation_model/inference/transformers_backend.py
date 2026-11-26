"""Transformers image+text -> text backend.

The ``LoadedTransformersVLM`` class keeps model weights resident on one GPU so
interactive chat does not reload the model for every question. ``generate_transformers``
is retained as a one-shot compatibility wrapper.
"""
import gc
import time
from pathlib import Path

from foundation_model.profiling.latency import InferenceProfile


def _is_local_path(path_or_id):
    try:
        return Path(str(path_or_id)).exists()
    except Exception:
        return False


def _is_lingshu(base_model):
    return "lingshu" in str(base_model).lower()


def _generic_model_class(transformers):
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError("Installed Transformers does not expose a multimodal auto-model class.")


def _load_model_and_processor(base_model, dtype, gpu_id, trust_remote_code, quantization="none"):
    import transformers

    local_only = _is_local_path(base_model)
    processor = transformers.AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        local_files_only=local_only,
    )

    if _is_lingshu(base_model) and hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
        model_cls = transformers.Qwen2_5_VLForConditionalGeneration
    else:
        model_cls = _generic_model_class(transformers)

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": int(gpu_id)},
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_only,
    }
    quantization = str(quantization or "none").lower()
    if quantization == "4bit_nf4":
        load_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization not in {"none", "bf16", "fp16"}:
        raise ValueError(f"Unsupported inference quantization: {quantization}")

    model = model_cls.from_pretrained(base_model, **load_kwargs)
    return model, processor


def _build_messages(prompt, history=None):
    messages = []
    for item in history or []:
        role = str(item.get("role", "")).strip()
        text = str(item.get("content", "")).strip()
        if role in {"user", "assistant", "system"} and text:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": str(prompt)},
            ],
        }
    )
    return messages


class LoadedTransformersVLM:
    """A single loaded multimodal model reused across many inference requests."""

    def __init__(
        self,
        base_model,
        adapter_path=None,
        gpu_id=0,
        precision="bf16",
        trust_remote_code=False,
        quantization="none",
    ):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this foundation-model inference path.")

        self.torch = torch
        self.gpu_id = int(gpu_id)
        self.base_model = str(base_model)
        self.quantization = str(quantization or "none")
        torch.cuda.set_device(self.gpu_id)
        dtype = torch.bfloat16 if str(precision).lower() == "bf16" else torch.float16

        self.model, self.processor = _load_model_and_processor(
            base_model=self.base_model,
            dtype=dtype,
            gpu_id=self.gpu_id,
            trust_remote_code=trust_remote_code,
            quantization=quantization,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path),
                is_trainable=False,
            )
        self.model.eval()

    def generate(
        self,
        image_path,
        prompt,
        max_new_tokens=256,
        temperature=0.0,
        history=None,
    ):
        from PIL import Image

        torch = self.torch
        image = Image.open(Path(image_path)).convert("RGB")
        messages = _build_messages(prompt=prompt, history=history)
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[rendered],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_tokens = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0

        torch.cuda.reset_peak_memory_stats(self.gpu_id)
        start = time.perf_counter()
        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": float(temperature) > 0,
        }
        if float(temperature) > 0:
            generation_kwargs["temperature"] = float(temperature)

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)

        torch.cuda.synchronize(self.gpu_id)
        elapsed = time.perf_counter() - start

        completion_ids = generated[:, input_tokens:] if input_tokens else generated
        texts = self.processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = texts[0] if texts else ""
        completion_tokens = int(completion_ids[0].numel()) if len(completion_ids) else 0
        peak_mb = torch.cuda.max_memory_allocated(self.gpu_id) / (1024 ** 2)
        profile = InferenceProfile(
            total_latency_s=elapsed,
            prompt_tokens=input_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=(completion_tokens / elapsed) if elapsed > 0 else 0.0,
            peak_vram_mb=peak_mb,
        )
        return text, profile.to_dict()

    def close(self):
        model = getattr(self, "model", None)
        processor = getattr(self, "processor", None)
        self.model = None
        self.processor = None
        del model, processor
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()
        except Exception:
            pass


def generate_transformers(
    base_model,
    image_path,
    prompt,
    adapter_path=None,
    gpu_id=0,
    precision="bf16",
    max_new_tokens=256,
    temperature=0.0,
    trust_remote_code=False,
    history=None,
    quantization="none",
):
    """Compatibility one-shot API used by the legacy worker."""
    runtime = LoadedTransformersVLM(
        base_model=base_model,
        adapter_path=adapter_path,
        gpu_id=gpu_id,
        precision=precision,
        trust_remote_code=trust_remote_code,
        quantization=quantization,
    )
    try:
        return runtime.generate(
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            history=history,
        )
    finally:
        runtime.close()
