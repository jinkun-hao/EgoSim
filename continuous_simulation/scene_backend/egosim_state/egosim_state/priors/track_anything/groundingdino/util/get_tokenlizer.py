import os

from transformers import AutoTokenizer, BertModel, RobertaModel

from egosim_state.utils.model_paths import (
    get_bert_base_uncased_path,
    get_grounding_dino_tokenizer_path,
    get_roberta_base_path,
)


def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )

    tokenizer = AutoTokenizer.from_pretrained(str(get_grounding_dino_tokenizer_path()))
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    if text_encoder_type == "bert-base-uncased" or (
        os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)
    ):
        return BertModel.from_pretrained(str(get_bert_base_uncased_path()))
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(str(get_roberta_base_path()))

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
