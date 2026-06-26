"""
Predict object phrases from an ego-view video using Qwen3-VL-4B-Instruct.

Produces a list of phrases compatible with the EgoSim State pipeline config.
This implementation intentionally mirrors the old quicktest flow, including
the original prompt, in-hand expansion, and hardcoded human phrases.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from pathlib import Path

import torch


warnings.filterwarnings("ignore")

logger = logging.getLogger("predict_phrases")

DEFAULT_MODEL_PATH = ""

HARDCODED_PHRASES = ["person", "hand", "arm"]

SYSTEM_PROMPT = (
    "You are inspecting an ego-view video of human hands interacting with objects. "
    "Identify all distinct objects that are visible and interacted with by the person's hands. "
    "For each object, output a short noun phrase describing it (e.g. 'brown glasses', 'beige case', 'red Lego block'). "
    "Use adjectives for color or appearance followed by the object noun. "
    "Exclude static background, atmosphere, table surfaces, and body parts. "
    "Return ONLY a JSON list of strings, for example: "
    '["brown glasses", "beige case"]'
)


def load_qwen_model(
    model_path: str = DEFAULT_MODEL_PATH,
    device_map: str = "auto",
):
    """Load Qwen3-VL-4B model and processor. Returns (model, processor)."""
    if not model_path:
        raise ValueError("Qwen model path is required. Set models.qwen_vl_root or pass --model_path.")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    logger.info("Loading Qwen3-VL-4B from %s", model_path)
    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    logger.info("Qwen3-VL-4B loaded")
    return model, processor


def _parse_phrases_from_text(raw_text: str) -> list[str]:
    """Extract a list of phrase strings from the model's raw output."""
    text = raw_text.strip()

    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        try:
            items = json.loads(json_match.group())
            if isinstance(items, list):
                return [str(s).strip() for s in items if str(s).strip()]
        except json.JSONDecodeError:
            pass

    lines = re.split(r"[\n,;]", text)
    phrases = []
    for line in lines:
        cleaned = re.sub(r"^[\s\-\d.*)+]+", "", line).strip().strip('"').strip("'")
        if cleaned and len(cleaned) < 80:
            phrases.append(cleaned)
    return phrases


def _build_in_hand_variants(phrases: list[str]) -> list[str]:
    """For each object phrase, add an 'X in hand' variant."""
    result = []
    for phrase in phrases:
        result.append(phrase)
        lower = phrase.lower()
        if "in hand" not in lower and lower not in ("person", "hand"):
            result.append(f"{phrase} in hand")
    return result


@torch.no_grad()
def predict_phrases(
    video_path: str,
    model=None,
    processor=None,
    max_new_tokens: int = 256,
) -> list[str]:
    """Predict object phrases from an ego-view video.

    Returns a deduplicated list with 'in hand' variants and hardcoded
    person/hand/arm appended, matching the legacy quicktest behavior.
    """
    if model is None or processor is None:
        model, processor = load_qwen_model()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    logger.info("QwenVL raw output: %s", raw_text)

    object_phrases = _parse_phrases_from_text(raw_text)
    if not object_phrases:
        logger.warning("No phrases parsed from QwenVL output, using fallback")
        object_phrases = ["object"]

    full_phrases = _build_in_hand_variants(object_phrases)

    seen = set()
    deduped = []
    for phrase in full_phrases:
        key = phrase.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(phrase)

    for phrase in HARDCODED_PHRASES:
        key = phrase.lower()
        if key not in seen:
            deduped.append(phrase)
            seen.add(key)

    return deduped


def format_phrases_yaml(phrases: list[str]) -> str:
    """Format phrases as YAML list string for Hydra override."""
    items = ", ".join(f'"{phrase}"' for phrase in phrases)
    return f"[{items}]"


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Predict EgoSim State phrases from ego-view video")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_json", type=str, default=None, help="Optional: save phrases to JSON file")
    args = parser.parse_args()

    model, processor = load_qwen_model(args.model_path)
    phrases = predict_phrases(args.video, model, processor, args.max_new_tokens)

    print("Predicted phrases:")
    for phrase in phrases:
        print(f"  - {phrase}")
    print(f"\nHydra override: pipeline.init.instance.phrases={format_phrases_yaml(phrases)}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(phrases, handle, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
