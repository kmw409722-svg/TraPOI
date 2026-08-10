import argparse
import json
import math
import os
import random
import re
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import transformers
from peft import PeftModel
from tqdm import tqdm
from transformers import BitsAndBytesConfig


DEFAULT_EOS_TOKEN = "</s>"
TRAFFIC_INDEX_PATTERN = re.compile(
    r"<\s*TRAFFIC\s+(\d+)\s*>",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate next-POI prediction with POI, traffic, and GPS features."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base language model.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to the test QA text file.",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        default="projector",
        choices=["base", "projector", "llm"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="LoRA adapter directory for test_type=llm.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--flash_attn",
        type=lambda value: str(value).lower() == "true",
        default=True,
    )

    parser.add_argument(
        "--poi_embedding_path",
        type=str,
        required=True,
        help="Path to the POI embedding file.",
    )
    parser.add_argument(
        "--poi_projector_path",
        type=str,
        default=None,
        help="Path to the merged POI projector.",
    )
    parser.add_argument(
        "--prepacked_pt",
        type=str,
        default=None,
        help="Path to prepacked_train.pt.",
    )
    parser.add_argument(
        "--traffic_compact_pt",
        type=str,
        default=None,
        help="Path to traffic_compact.pt.",
    )
    parser.add_argument(
        "--traffic_projector_path",
        type=str,
        default=None,
        help="Path to the merged traffic projector.",
    )
    parser.add_argument(
        "--traffic_id_map_json",
        type=str,
        default=None,
        help="Path to the traffic ID map.",
    )
    parser.add_argument(
        "--gps_projector_path",
        type=str,
        default=None,
        help="Path to the merged GPS projector.",
    )
    parser.add_argument(
        "--geohash_vocab_path",
        type=str,
        default=None,
        help="Path to the geohash vocabulary.",
    )
    parser.add_argument(
        "--geohash_embedding_ckpt",
        type=str,
        default=None,
        help="Path to the trained geohash embedding.",
    )
    return parser.parse_args()


def set_seed(seed: int = 2) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_geohash_embedding(
    geohash_vocab_path: str,
    device: torch.device,
) -> tuple[nn.Embedding, dict]:
    if geohash_vocab_path is None:
        raise ValueError("--geohash_vocab_path is required for GPS features.")

    with open(geohash_vocab_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    geohash2id = vocab["geohash2id"]
    max_id = max(int(value) for value in geohash2id.values())
    embedding = nn.Embedding(max_id + 1, 128).to(device)
    return embedding, geohash2id


def load_geohash_embedding(
    geohash_vocab_path: str,
    device: torch.device,
    checkpoint_path: Optional[str],
) -> tuple[nn.Embedding, dict]:
    embedding, geohash2id = build_geohash_embedding(
        geohash_vocab_path,
        device,
    )

    if checkpoint_path is None:
        raise ValueError("A geohash embedding checkpoint is required.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Geohash embedding not found: {checkpoint_path}"
        )

    try:
        if checkpoint_path.endswith(".npy"):
            weight = torch.tensor(
                np.load(checkpoint_path),
                dtype=embedding.weight.dtype,
                device=device,
            )
            if weight.shape != embedding.weight.shape:
                raise ValueError(
                    f"Geohash shape mismatch: {tuple(weight.shape)} != "
                    f"{tuple(embedding.weight.shape)}"
                )
            with torch.no_grad():
                embedding.weight.copy_(weight)
        else:
            state = torch.load(checkpoint_path, map_location=device)
            if isinstance(state, dict):
                embedding.load_state_dict(state)
            elif isinstance(state, torch.Tensor):
                if state.shape != embedding.weight.shape:
                    raise ValueError(
                        f"Geohash shape mismatch: {tuple(state.shape)} != "
                        f"{tuple(embedding.weight.shape)}"
                    )
                with torch.no_grad():
                    embedding.weight.copy_(state.to(device))
            else:
                raise TypeError("Unsupported geohash checkpoint format.")
    except Exception as error:
        raise RuntimeError(
            f"Could not load geohash embedding: {error}"
        ) from error

    print(f"[INFO] Loaded geohash embedding: {checkpoint_path}")
    return embedding, geohash2id


def resize_model_embeddings(tokenizer, model) -> None:
    old_size = model.get_input_embeddings().weight.size(0)
    new_size = len(tokenizer)
    if new_size <= old_size:
        return

    model.resize_token_embeddings(new_size)
    with torch.no_grad():
        input_weight = model.get_input_embeddings().weight.data
        input_average = input_weight[:old_size].mean(dim=0, keepdim=True)
        input_weight[old_size:new_size] = input_average

        output_layer = model.get_output_embeddings()
        if output_layer is not None:
            output_weight = output_layer.weight.data
            output_average = output_weight[:old_size].mean(
                dim=0,
                keepdim=True,
            )
            output_weight[old_size:new_size] = output_average


def build_projector(
    weight_0: torch.Tensor,
    bias_0: Optional[torch.Tensor],
    weight_2: torch.Tensor,
    bias_2: Optional[torch.Tensor],
) -> nn.Sequential:
    projector = nn.Sequential(
        nn.Linear(
            weight_0.shape[1],
            weight_0.shape[0],
            bias=bias_0 is not None,
        ),
        nn.GELU(),
        nn.Linear(
            weight_2.shape[1],
            weight_2.shape[0],
            bias=bias_2 is not None,
        ),
    )

    with torch.no_grad():
        projector[0].weight.copy_(weight_0)
        if bias_0 is not None:
            projector[0].bias.copy_(bias_0)
        projector[2].weight.copy_(weight_2)
        if bias_2 is not None:
            projector[2].bias.copy_(bias_2)

    return projector


def load_projector(
    projector_path: str,
    device: torch.device,
) -> nn.Sequential:
    if not projector_path:
        raise ValueError("A projector path is required.")
    if not os.path.isfile(projector_path):
        raise FileNotFoundError(f"Projector not found: {projector_path}")

    state = torch.load(projector_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("The projector checkpoint must be a dictionary.")

    if "layer_0_weight" in state and "layer_2_weight" in state:
        weight_0 = state["layer_0_weight"]
        bias_0 = state.get("layer_0_bias")
        weight_2 = state["layer_2_weight"]
        bias_2 = state.get("layer_2_bias")
    elif "0.weight" in state and "2.weight" in state:
        weight_0 = state["0.weight"]
        bias_0 = state.get("0.bias")
        weight_2 = state["2.weight"]
        bias_2 = state.get("2.bias")
    else:
        keys = list(state.keys())
        weight_0_key = next(
            (key for key in keys if key.endswith(".0.weight")),
            None,
        )
        weight_2_key = next(
            (key for key in keys if key.endswith(".2.weight")),
            None,
        )
        if weight_0_key is None or weight_2_key is None:
            raise ValueError(
                f"Unsupported projector format: {projector_path}"
            )

        prefix_0 = weight_0_key[: -len("0.weight")]
        prefix_2 = weight_2_key[: -len("2.weight")]
        weight_0 = state[weight_0_key]
        bias_0 = state.get(prefix_0 + "0.bias")
        weight_2 = state[weight_2_key]
        bias_2 = state.get(prefix_2 + "2.bias")

    projector = build_projector(
        weight_0,
        bias_0,
        weight_2,
        bias_2,
    ).to(device)
    projector.eval()

    if not all(
        torch.isfinite(parameter).all()
        for parameter in projector.parameters()
    ):
        raise ValueError(f"Projector contains NaN or Inf: {projector_path}")

    print(f"[INFO] Loaded projector: {projector_path}")
    return projector


def normalize_prompt_tokens(text: str) -> str:
    text = text.replace("`", "")
    text = re.sub(r"<\s*POI\s+(\d+)\s*>", r"<POI \1>", text)
    text = re.sub(r"<\s*TRAFFIC\s+(\d+)\s*>", r"<TRAFFIC \1>", text)
    text = re.sub(r"(?<!\s)(<POI\s+\d+>)", r" \1", text)
    text = re.sub(r"(?<!\s)(<TRAFFIC\s+\d+>)", r" \1", text)
    text = re.sub(r"(<POI\s+\d+>)(?!\s)", r"\1 ", text)
    text = re.sub(r"(<TRAFFIC\s+\d+>)(?!\s)", r"\1 ", text)
    return re.sub(r"\s+", " ", text)


def load_poi_embeddings(path: str) -> torch.Tensor:
    return torch.tensor(
        np.load(path, allow_pickle=False),
        dtype=torch.float32,
    )


def load_traffic_compact(pt_path: str):
    data = torch.load(pt_path, map_location="cpu")
    return (
        data["traffic_compact"],
        int(data["Dg"]),
        list(data["station_ids"]),
    )


def infer_consecutive_start_id(tokenizer, tokens: List[str]) -> int:
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if not token_ids:
        raise ValueError("No special tokens were created.")

    start_id = min(token_ids)
    expected = list(range(start_id, start_id + len(token_ids)))
    if token_ids != expected:
        raise ValueError("Special token IDs are not consecutive.")
    return start_id


def load_reverse_map(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    reverse = data.get("reverse")
    if not isinstance(reverse, list) or not reverse:
        raise ValueError("The traffic ID map has no valid 'reverse' list.")
    return [str(value) for value in reverse]


def rewrite_traffic_ids(
    text: str,
    index_to_id: List[str],
) -> str:
    def replace(match):
        index = int(match.group(1))
        if 0 <= index < len(index_to_id):
            return f"<TRAFFIC {index_to_id[index]}>"
        return match.group(0)

    return TRAFFIC_INDEX_PATTERN.sub(replace, text)


def embedding_hook(module, inputs, output):
    """Inject projected POI, traffic, and GPS features."""
    input_ids = inputs[0].to(module.weight.device)
    token_embeddings = output

    poi_start = module.poi_token_start_id
    traffic_start = getattr(module, "traffic_token_start_id", None)
    gps_start = getattr(module, "gps_token_start_id", None)

    is_poi = (
        (input_ids >= poi_start)
        & (input_ids < poi_start + module.num_poi_tokens)
    )

    if traffic_start is None:
        is_traffic = torch.zeros_like(input_ids, dtype=torch.bool)
    else:
        is_traffic = (
            (input_ids >= traffic_start)
            & (
                input_ids
                < traffic_start + module.num_traffic_tokens
            )
        )

    if gps_start is None:
        is_gps = torch.zeros_like(input_ids, dtype=torch.bool)
    else:
        is_gps = (
            (input_ids >= gps_start)
            & (input_ids < gps_start + module.num_gps_tokens)
        )

    if is_poi.any():
        poi_indices = input_ids[is_poi] - poi_start
        poi_features = module.poi_embedding_layer(poi_indices)
        projected = module.linear_mapping_layer(poi_features)
        token_embeddings[is_poi] = projected.to(token_embeddings.dtype)

    if (
        is_traffic.any()
        and hasattr(module, "traffic_pair_idx")
        and hasattr(module, "traffic_compact")
        and hasattr(module, "traffic_mapping_layer")
    ):
        batch_indices, token_indices = torch.nonzero(
            is_traffic,
            as_tuple=True,
        )
        compact_indices = module.traffic_pair_idx[
            batch_indices,
            token_indices,
        ].long()
        compact_size = int(module.traffic_compact.shape[0])
        valid = (
            (compact_indices >= 0)
            & (compact_indices < compact_size)
        )

        if valid.any():
            valid_batch = batch_indices[valid]
            valid_tokens = token_indices[valid]
            traffic_features = module.traffic_compact[
                compact_indices[valid]
            ]
            projected = module.traffic_mapping_layer(
                traffic_features
            )
            token_embeddings[valid_batch, valid_tokens] = projected.to(
                token_embeddings.dtype
            )

    if (
        is_gps.any()
        and hasattr(module, "geohash_embedding_layer")
        and hasattr(module, "gps_mapping_layer")
    ):
        gps_indices = input_ids[is_gps] - gps_start + 1
        gps_features = module.geohash_embedding_layer(gps_indices)
        projected = module.gps_mapping_layer(gps_features)
        token_embeddings[is_gps] = projected.to(token_embeddings.dtype)

    return token_embeddings


def apply_embedding_hook(
    model,
    poi_embedding_layer: nn.Embedding,
    poi_token_start_id: int,
    num_poi_tokens: int,
    poi_projector: nn.Module,
    traffic_compact: Optional[torch.Tensor] = None,
    traffic_token_start_id: Optional[int] = None,
    num_traffic_tokens: Optional[int] = None,
    traffic_projector: Optional[nn.Module] = None,
    geohash_embedding_layer: Optional[nn.Embedding] = None,
    gps_token_start_id: Optional[int] = None,
    num_gps_tokens: Optional[int] = None,
    gps_projector: Optional[nn.Module] = None,
) -> None:
    embedding = model.get_input_embeddings()

    embedding.poi_embedding_layer = poi_embedding_layer
    embedding.poi_token_start_id = poi_token_start_id
    embedding.num_poi_tokens = num_poi_tokens
    embedding.linear_mapping_layer = poi_projector

    if (
        traffic_compact is not None
        and traffic_token_start_id is not None
        and num_traffic_tokens is not None
        and traffic_projector is not None
    ):
        embedding.traffic_compact = traffic_compact.to(
            embedding.weight.device
        )
        embedding.traffic_token_start_id = traffic_token_start_id
        embedding.num_traffic_tokens = num_traffic_tokens
        embedding.traffic_mapping_layer = traffic_projector

    if (
        geohash_embedding_layer is not None
        and gps_token_start_id is not None
        and num_gps_tokens is not None
        and gps_projector is not None
    ):
        embedding.geohash_embedding_layer = (
            geohash_embedding_layer.to(embedding.weight.device)
        )
        embedding.gps_token_start_id = gps_token_start_id
        embedding.num_gps_tokens = num_gps_tokens
        embedding.gps_mapping_layer = gps_projector

    embedding.register_forward_hook(embedding_hook)


def extract_poi_id(text: str) -> Optional[str]:
    match = re.search(r"POI id\s+(\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    digits = re.findall(r"\d+", text)
    return digits[0] if digits else None


def evaluate_prediction(
    prediction: str,
    ground_truth: str,
) -> int:
    predicted_poi = extract_poi_id(prediction)
    actual_poi = extract_poi_id(ground_truth)
    return int(
        predicted_poi is not None
        and actual_poi is not None
        and predicted_poi == actual_poi
    )


def validate_required_args(args) -> None:
    if not os.path.isfile(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")

    if args.test_type in {"projector", "llm"}:
        if not args.poi_projector_path:
            raise ValueError(
                "--poi_projector_path is required for projector or LLM evaluation."
            )

    traffic_values = [
        args.prepacked_pt,
        args.traffic_compact_pt,
        args.traffic_projector_path,
    ]
    if any(traffic_values) and not all(traffic_values):
        raise ValueError(
            "Traffic evaluation requires --prepacked_pt, "
            "--traffic_compact_pt, and --traffic_projector_path."
        )

    gps_values = [
        args.gps_projector_path,
        args.geohash_vocab_path,
    ]
    if any(gps_values) and not all(gps_values):
        raise ValueError(
            "GPS evaluation requires --gps_projector_path and "
            "--geohash_vocab_path."
        )

    if args.test_type == "llm" and not args.output_dir:
        raise ValueError("--output_dir is required for test_type=llm.")

    if (
        args.gps_projector_path
        and args.geohash_embedding_ckpt is None
        and args.output_dir is None
    ):
        raise ValueError(
            "Provide --geohash_embedding_ckpt or --output_dir for GPS evaluation."
        )


def main(args) -> None:
    validate_required_args(args)
    set_seed()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.seq_len,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.flash_attn:
        try:
            from llama_attn_replace_sft import replace_llama_attn

            replace_llama_attn(inference=True)
        except Exception:
            print("[WARN] Flash attention replacement was skipped.")

    config = transformers.AutoConfig.from_pretrained(args.base_model)
    original_context = getattr(config, "max_position_embeddings", None)
    if original_context and args.context_size > original_context:
        factor = float(
            math.ceil(args.context_size / original_context)
        )
        config.rope_scaling = {"type": "linear", "factor": factor}

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    num_poi = np.load(args.poi_embedding_path).shape[0]
    poi_tokens = [f"<POI {index}>" for index in range(num_poi)]

    traffic_compact = None
    traffic_dim = None
    station_ids = []
    traffic_tokens = []
    if args.traffic_compact_pt:
        (
            traffic_compact,
            traffic_dim,
            station_ids,
        ) = load_traffic_compact(args.traffic_compact_pt)
        traffic_tokens = [
            f"<TRAFFIC {station_id}>" for station_id in station_ids
        ]

    num_gps_tokens = 0
    gps_tokens = []
    if args.geohash_vocab_path:
        with open(
            args.geohash_vocab_path,
            "r",
            encoding="utf-8",
        ) as file:
            geohash_vocab = json.load(file)
        geohash2id = geohash_vocab["geohash2id"]
        num_gps_tokens = max(
            int(value) for value in geohash2id.values()
        )
        gps_tokens = [
            f"<GPS {index}>"
            for index in range(1, num_gps_tokens + 1)
        ]

    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": (
                poi_tokens + traffic_tokens + gps_tokens
            )
        }
    )
    resize_model_embeddings(tokenizer, model)

    poi_token_start_id = infer_consecutive_start_id(
        tokenizer,
        poi_tokens,
    )
    traffic_token_start_id = (
        infer_consecutive_start_id(tokenizer, traffic_tokens)
        if traffic_tokens
        else None
    )
    gps_token_start_id = (
        infer_consecutive_start_id(tokenizer, gps_tokens)
        if gps_tokens
        else None
    )

    prepacked_pairs = None
    if args.prepacked_pt:
        prepacked = torch.load(args.prepacked_pt, map_location="cpu")
        packed_station_ids = list(prepacked["station_ids"])
        packed_start_id = int(prepacked["traffic_token_start_id"])

        if packed_station_ids != station_ids:
            raise ValueError(
                "Station IDs differ between prepacked and compact files."
            )
        if packed_start_id != traffic_token_start_id:
            raise ValueError(
                "Traffic token start ID differs from the prepacked file."
            )

        prepacked_pairs = prepacked["traffic_pair_idx"]
        tokenizer.truncation_side = "left"

    if tokenizer.eos_token is None:
        tokenizer.eos_token = DEFAULT_EOS_TOKEN

    reverse_map = load_reverse_map(args.traffic_id_map_json)

    if args.test_type in {"projector", "llm"}:
        poi_embeddings = load_poi_embeddings(
            args.poi_embedding_path
        )
        model_hidden_size = model.get_input_embeddings().weight.shape[1]

        poi_projector = load_projector(
            args.poi_projector_path,
            device,
        )
        if poi_projector[0].in_features != 128:
            raise ValueError("The POI projector input size must be 128.")
        if poi_projector[2].out_features != model_hidden_size:
            raise ValueError(
                "The POI projector output size does not match the model."
            )

        traffic_projector = None
        if args.traffic_projector_path:
            traffic_projector = load_projector(
                args.traffic_projector_path,
                device,
            )
            if traffic_projector[0].in_features != traffic_dim:
                raise ValueError(
                    "The traffic projector input size does not match "
                    "traffic_compact.pt."
                )
            if traffic_projector[2].out_features != model_hidden_size:
                raise ValueError(
                    "The traffic projector output size does not match "
                    "the model."
                )

        gps_projector = None
        geohash_embedding = None
        if args.gps_projector_path:
            gps_projector = load_projector(
                args.gps_projector_path,
                device,
            )
            if gps_projector[0].in_features != 128:
                raise ValueError(
                    "The GPS projector input size must be 128."
                )
            if gps_projector[2].out_features != model_hidden_size:
                raise ValueError(
                    "The GPS projector output size does not match "
                    "the model."
                )

            geohash_checkpoint = args.geohash_embedding_ckpt
            if geohash_checkpoint is None:
                geohash_checkpoint = os.path.join(
                    args.output_dir,
                    "geohash_embedding.pt",
                )
            geohash_embedding, _ = load_geohash_embedding(
                args.geohash_vocab_path,
                device,
                geohash_checkpoint,
            )

        poi_embedding_layer = nn.Embedding(
            len(poi_embeddings),
            128,
        ).to(device)
        with torch.no_grad():
            poi_embedding_layer.weight.copy_(
                poi_embeddings.to(device)
            )
        poi_embedding_layer.requires_grad_(False)

        apply_embedding_hook(
            model=model,
            poi_embedding_layer=poi_embedding_layer,
            poi_token_start_id=poi_token_start_id,
            num_poi_tokens=len(poi_embeddings),
            poi_projector=poi_projector,
            traffic_compact=traffic_compact,
            traffic_token_start_id=traffic_token_start_id,
            num_traffic_tokens=(
                len(station_ids) if station_ids else None
            ),
            traffic_projector=traffic_projector,
            geohash_embedding_layer=geohash_embedding,
            gps_token_start_id=gps_token_start_id,
            num_gps_tokens=(
                num_gps_tokens if num_gps_tokens > 0 else None
            ),
            gps_projector=gps_projector,
        )

        if reverse_map is None and station_ids:
            reverse_map = list(station_ids)

    if args.test_type == "llm":
        model = PeftModel.from_pretrained(
            model,
            args.output_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print(f"[INFO] Loaded LoRA adapter: {args.output_dir}")

    model.eval()

    generation_config = transformers.GenerationConfig(
        max_new_tokens=4,
        do_sample=True,
        temperature=0.6,
        top_k=40,
        top_p=0.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
    )

    with open(args.test_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    correct = 0
    valid_count = 0
    correct_indices = []
    prepacked_index = 0

    for line_index, line in tqdm(
        enumerate(lines),
        desc="Evaluating",
        total=len(lines),
    ):
        try:
            question, ground_truth = line.split("<answer>:", 1)

            answer_text = ground_truth.strip()
            prefix_match = re.match(
                r"^(At .*?, user \S+ will visit POI id )\d+\.\s*$",
                answer_text,
            )
            if prefix_match is None:
                continue

            generation_prefix = prefix_match.group(1)
        except ValueError:
            continue

        if reverse_map is not None:
            question = rewrite_traffic_ids(question, reverse_map)
        question = normalize_prompt_tokens(question)

        prompt = (
            question
            + "<answer>:"
            + generation_prefix
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.seq_len,
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        valid_count += 1

        if prepacked_pairs is not None:
            if prepacked_index >= len(prepacked_pairs):
                raise IndexError(
                    "The prepacked entries ended before the test samples."
                )

            raw_indices = prepacked_pairs[prepacked_index]
            prepacked_index += 1
            if isinstance(raw_indices, (list, tuple)):
                raw_indices = torch.tensor(
                    raw_indices,
                    dtype=torch.long,
                )
            raw_indices = raw_indices.view(-1)

            embedding = model.get_input_embeddings()
            traffic_start = getattr(
                embedding,
                "traffic_token_start_id",
                None,
            )
            traffic_count = getattr(
                embedding,
                "num_traffic_tokens",
                None,
            )

            aligned = torch.full(
                (inputs.input_ids.shape[1],),
                -1,
                dtype=torch.long,
            )

            if traffic_start is not None and traffic_count is not None:
                is_traffic = (
                    (inputs.input_ids >= traffic_start)
                    & (
                        inputs.input_ids
                        < traffic_start + traffic_count
                    )
                )
                traffic_positions = torch.nonzero(
                    is_traffic[0],
                    as_tuple=False,
                ).view(-1)
                compact_indices = raw_indices[raw_indices >= 0]
                match_count = min(
                    len(traffic_positions),
                    len(compact_indices),
                )
                if match_count > 0:
                    aligned[traffic_positions[-match_count:]] = (
                        compact_indices[-match_count:]
                    )

            embedding.traffic_pair_idx = aligned.unsqueeze(0).to(
                embedding.weight.device
            )

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
            )

        generated_tokens = outputs[
            :,
            inputs.input_ids.shape[1] :,
        ][0]
        prediction = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        if evaluate_prediction(prediction, ground_truth):
            correct += 1
            correct_indices.append(line_index)

    if valid_count == 0:
        raise ValueError("No valid test samples were found.")

    accuracy = correct / valid_count
    print(f"[RESULT] Valid samples: {valid_count}")
    print(f"[RESULT] ACC@1: {accuracy:.6f}")
    print(f"[RESULT] Correct indices: {correct_indices}")


if __name__ == "__main__":
    main(parse_args())
