"""MedGemma request adapter.

The adapter owns prompt/message semantics while Transformers/vLLM own execution.
"""
from foundation_model.models.base import MultimodalRequest


class MedGemmaAdapter:
    key = "medgemma4b"

    def __init__(self, model_id="google/medgemma-4b-it"):
        self.model_id = str(model_id)

    def build_messages(self, request: MultimodalRequest):
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": request.system_prompt}]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": request.image_path},
                {"type": "text", "text": request.prompt},
            ],
        })
        return messages
