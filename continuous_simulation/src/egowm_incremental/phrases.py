from __future__ import annotations

DEFAULT_SCENE_PHRASES = ("arm", "person", "hand")
FALLBACK_QWEN_PHRASES = ("object", "object in hand")


def normalize_scene_phrases(
    phrases: list[str] | tuple[str, ...] | None,
) -> list[str]:
    normalized_phrases: list[str] = []
    seen: set[str] = set()

    for phrase in list(phrases or []):
        if phrase is None:
            continue
        normalized = str(phrase).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized_phrases.append(normalized)
    return normalized_phrases


def has_custom_scene_phrases(
    phrases: list[str] | tuple[str, ...] | None,
) -> bool:
    normalized = normalize_scene_phrases(phrases)
    return bool(normalized) and normalized != normalize_scene_phrases(DEFAULT_SCENE_PHRASES)


def merge_scene_phrases(
    predicted_phrases: list[str] | tuple[str, ...] | None,
    *,
    base_phrases: list[str] | tuple[str, ...] = DEFAULT_SCENE_PHRASES,
) -> list[str]:
    return normalize_scene_phrases(list(base_phrases) + list(predicted_phrases or []))
