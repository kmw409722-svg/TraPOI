import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

os.environ["TRANSFORMERS_NO_DEEPSPEED"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn import ModuleDict
from torch.utils.data import Subset
from transformers import BitsAndBytesConfig, Trainer

from llama_attn_replace_sft import replace_llama_attn


IGNORE_INDEX = -100


def set_seed(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="yukang/Llama-2-7b-longlora-32k-ft"
    )
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training QA JSON file."},
    )
    poi_embedding_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the 128-dimensional POI embedding file."},
    )
    prepacked_pt: Optional[str] = field(
        default=None,
        metadata={"help": "Path to prepacked_train.pt."},
    )
    traffic_compact_pt: Optional[str] = field(
        default=None,
        metadata={"help": "Path to traffic_compact.pt."},
    )
    geohash_vocab_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the geohash vocabulary JSON file."},
    )
    poi_geohash_json_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the POI-to-geohash JSON file."},
    )
    traffic_geohash_json_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the traffic-station-to-geohash JSON file."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=32768)
    use_flash_attn: bool = field(default=True)
    use_full_attn: bool = field(default=False)
    low_rank_training: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=8)
    lora_r: int = field(default=256)
    lora_alpha: float = field(default=512)
    save_total_limit: int = field(default=1)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer,
    model,
) -> None:
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens <= 0:
        return

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    input_average = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_average = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    input_embeddings[-num_new_tokens:] = input_average
    output_embeddings[-num_new_tokens:] = output_average


def load_poi_embeddings(path: str) -> torch.Tensor:
    return torch.tensor(np.load(path))


def load_traffic_compact(pt_path: str):
    blob = torch.load(pt_path, map_location="cpu")
    return (
        blob["traffic_compact"],
        int(blob["Dg"]),
        list(blob["station_ids"]),
    )


def initialize_embedding_layer_from_array(
    embedding_layer: nn.Embedding,
    array: torch.Tensor,
    device,
) -> None:
    with torch.no_grad():
        embedding_layer.to(device)
        embedding_layer.weight.data.copy_(array.to(device))


def embedding_hook(module, inputs, output):
    """Replace POI, traffic, and GPS token embeddings with projected features."""
    input_ids = inputs[0].to(module.weight.device)
    token_embeddings = output

    poi_start = module.poi_token_start_id
    traffic_start = module.traffic_token_start_id
    gps_start = module.gps_token_start_id

    is_poi = (
        (input_ids >= poi_start)
        & (input_ids < poi_start + module.num_poi_tokens)
    )
    is_traffic = (
        (input_ids >= traffic_start)
        & (input_ids < traffic_start + module.num_traffic_tokens)
    )
    is_gps = (
        (input_ids >= gps_start)
        & (input_ids < gps_start + module.num_gps_tokens)
    )

    if is_poi.any():
        poi_indices = input_ids[is_poi] - poi_start
        poi_features = module.poi_embedding_layer(poi_indices)
        mapped_poi = module.linear_mapping_layer(poi_features)
        token_embeddings[is_poi] = mapped_poi.to(token_embeddings.dtype)

    if (
        is_traffic.any()
        and hasattr(module, "traffic_pair_idx")
        and hasattr(module, "traffic_compact")
    ):
        batch_indices, token_indices = torch.nonzero(is_traffic, as_tuple=True)
        compact_indices = module.traffic_pair_idx[
            batch_indices, token_indices
        ].long()
        compact_size = int(module.traffic_compact.shape[0])
        valid = (compact_indices >= 0) & (compact_indices < compact_size)

        if valid.any():
            valid_batch = batch_indices[valid]
            valid_tokens = token_indices[valid]
            valid_compact = compact_indices[valid]
            traffic_features = module.traffic_compact[valid_compact]
            mapped_traffic = module.traffic_mapping_layer(traffic_features)
            token_embeddings[valid_batch, valid_tokens] = mapped_traffic.to(
                token_embeddings.dtype
            )

    if is_gps.any():
        gps_indices = input_ids[is_gps] - gps_start + 1
        gps_features = module.geohash_embedding_layer(gps_indices)
        mapped_gps = module.gps_mapping_layer(gps_features)
        token_embeddings[is_gps] = mapped_gps.to(token_embeddings.dtype)

    return token_embeddings


def apply_embedding_hook_prepacked(
    base_model,
    poi_embedding_layer: nn.Embedding,
    poi_token_start_id: int,
    num_poi_tokens: int,
    traffic_compact: torch.Tensor,
    traffic_token_start_id: int,
    num_traffic_tokens: int,
    gps_token_start_id: int,
    num_gps_tokens: int,
    linear_mapping_layer: nn.Module,
    traffic_mapping_layer: nn.Module,
    gps_mapping_layer: nn.Module,
    geohash_embedding_layer: nn.Embedding,
) -> None:
    embedding = base_model.get_input_embeddings()

    embedding.poi_embedding_layer = poi_embedding_layer
    embedding.poi_token_start_id = poi_token_start_id
    embedding.num_poi_tokens = num_poi_tokens
    embedding.linear_mapping_layer = linear_mapping_layer

    embedding.traffic_compact = traffic_compact.to(embedding.weight.device)
    embedding.traffic_token_start_id = traffic_token_start_id
    embedding.num_traffic_tokens = num_traffic_tokens
    embedding.traffic_mapping_layer = traffic_mapping_layer

    embedding.gps_token_start_id = gps_token_start_id
    embedding.num_gps_tokens = num_gps_tokens
    embedding.geohash_embedding_layer = geohash_embedding_layer.to(
        embedding.weight.device
    )
    embedding.gps_mapping_layer = gps_mapping_layer

    embedding.register_forward_hook(embedding_hook)


class PrepackedDataset(torch.utils.data.Dataset):
    def __init__(self, prepacked_path: str):
        blob = torch.load(prepacked_path, map_location="cpu")
        self.input_ids = blob["input_ids"]
        self.labels = blob["labels"]
        self.traffic_pair_idx = blob["traffic_pair_idx"]
        self.traffic_token_start_id = int(blob["traffic_token_start_id"])
        self.station_ids = list(blob["station_ids"])

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
            "traffic_pair_idx": self.traffic_pair_idx[index],
        }

    def nonempty_indices(self) -> List[int]:
        return [
            index
            for index, tensor in enumerate(self.input_ids)
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0
        ]


@dataclass
class DataCollatorPrepacked:
    pad_id: int
    max_len: int
    truncation_side: str = "right"

    def _truncate(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.size(0) <= self.max_len:
            return tensor
        if self.truncation_side == "right":
            return tensor[: self.max_len]
        return tensor[-self.max_len :]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        clean = [
            item
            for item in instances
            if isinstance(item.get("input_ids"), torch.Tensor)
            and item["input_ids"].numel() > 0
        ]

        if not clean:
            dummy = torch.tensor([self.pad_id], dtype=torch.long).unsqueeze(0)
            return {
                "input_ids": dummy,
                "labels": torch.full_like(dummy, IGNORE_INDEX),
                "attention_mask": dummy.ne(self.pad_id),
                "traffic_pair_idx": torch.full_like(dummy, -1),
            }

        input_ids_list = [self._truncate(item["input_ids"]) for item in clean]
        labels_list = [self._truncate(item["labels"]) for item in clean]
        pair_index_list = [
            self._truncate(item["traffic_pair_idx"]) for item in clean
        ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.pad_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        traffic_pair_idx = torch.nn.utils.rnn.pad_sequence(
            pair_index_list,
            batch_first=True,
            padding_value=-1,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.pad_id),
            "traffic_pair_idx": traffic_pair_idx,
        }


class TrafficPOIWrapper(nn.Module):
    def __init__(self, base_model: transformers.PreTrainedModel):
        super().__init__()
        self.base_model = base_model

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            return self.base_model.gradient_checkpointing_enable(**kwargs)
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable()
        return None

    def enable_input_require_grads(self):
        if hasattr(self.base_model, "enable_input_require_grads"):
            return self.base_model.enable_input_require_grads()
        return None

    @property
    def config(self):
        return getattr(self.base_model, "config", None)

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        if hasattr(self.base_model, "get_output_embeddings"):
            return self.base_model.get_output_embeddings()
        return None

    def tie_weights(self):
        if hasattr(self.base_model, "tie_weights"):
            return self.base_model.tie_weights()
        return None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        traffic_pair_idx=None,
        **kwargs,
    ):
        embedding = self.base_model.get_input_embeddings()
        if traffic_pair_idx is not None:
            embedding.traffic_pair_idx = traffic_pair_idx.to(
                embedding.weight.device
            )

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def _lora_items(obj):
    if obj is None:
        return []
    if isinstance(obj, (dict, ModuleDict)):
        return list(obj.items())
    try:
        return list(obj.items())
    except Exception:
        return []


def to_cpu_full_param(value):
    if value is None:
        return None
    if isinstance(value, torch.nn.Parameter):
        tensor = value
    else:
        tensor = torch.nn.Parameter(value, requires_grad=False)
    return tensor.data.detach().cpu().clone()


def _get_scale_for_tag(
    layer: nn.Module,
    tag: str,
    default_scale: float,
) -> float:
    try:
        scaling = getattr(layer, "scaling", None)
        if isinstance(scaling, dict) and tag in scaling:
            return float(scaling[tag])
        if isinstance(scaling, (int, float)):
            return float(scaling)

        lora_alpha = getattr(layer, "lora_alpha", None)
        rank = getattr(layer, "r", None)
        if (
            isinstance(lora_alpha, dict)
            and isinstance(rank, dict)
            and tag in lora_alpha
            and tag in rank
            and rank[tag]
        ):
            return float(lora_alpha[tag]) / float(rank[tag])
        if (
            isinstance(lora_alpha, (int, float))
            and isinstance(rank, (int, float))
            and rank
        ):
            return float(lora_alpha) / float(rank)
    except Exception:
        pass

    return float(default_scale)


def save_combined_linear_layer(
    module_list: nn.Sequential,
    output_path: str,
    filename: str,
    alpha: float,
    rank: int,
) -> None:
    os.makedirs(output_path, exist_ok=True)
    default_scale = float(alpha) / float(max(1, rank))
    combined = {}

    with torch.no_grad():
        for index, layer in enumerate(module_list):
            if isinstance(layer, nn.GELU):
                combined[f"layer_{index}_type"] = "GELU"
                continue

            is_linear = (
                isinstance(layer, nn.Linear)
                or layer.__class__.__name__.lower().endswith("linear")
            )
            if not is_linear:
                continue

            base_layer = getattr(layer, "base_layer", layer)
            weight = to_cpu_full_param(getattr(base_layer, "weight", None))
            if weight is None:
                continue

            bias = to_cpu_full_param(getattr(base_layer, "bias", None))
            merged_weight = weight.clone()

            lora_a = getattr(layer, "lora_A", None)
            lora_b = getattr(layer, "lora_B", None)
            tags_a = {tag for tag, _ in _lora_items(lora_a)}
            tags_b = {tag for tag, _ in _lora_items(lora_b)}

            for tag in sorted(tags_a & tags_b):
                a_module = dict(_lora_items(lora_a))[tag]
                b_module = dict(_lora_items(lora_b))[tag]
                a_weight = to_cpu_full_param(
                    getattr(a_module, "weight", None)
                )
                b_weight = to_cpu_full_param(
                    getattr(b_module, "weight", None)
                )
                if a_weight is None or b_weight is None:
                    continue

                scale = _get_scale_for_tag(layer, tag, default_scale)
                merged_weight.add_(
                    (scale * (b_weight @ a_weight)).to(merged_weight.dtype)
                )

            combined[f"layer_{index}_weight"] = merged_weight
            if bias is not None:
                combined[f"layer_{index}_bias"] = bias

    save_path = os.path.join(output_path, filename)
    torch.save(combined, save_path)
    print(f"[SAVE] Projector saved: {save_path}")


def save_split_projector_weights(
    module: nn.Sequential,
    output_dir: str,
    prefix: str,
) -> None:
    base_state = {}
    lora_state = {}

    with torch.no_grad():
        for index, layer in enumerate(module):
            is_linear = (
                isinstance(layer, nn.Linear)
                or layer.__class__.__name__.lower().endswith("linear")
            )
            if not is_linear:
                continue

            base_layer = getattr(layer, "base_layer", layer)
            weight = to_cpu_full_param(getattr(base_layer, "weight", None))
            bias = to_cpu_full_param(getattr(base_layer, "bias", None))
            if weight is not None:
                base_state[f"layer_{index}_weight"] = weight
            if bias is not None:
                base_state[f"layer_{index}_bias"] = bias

            for tag, lora_a in _lora_items(getattr(layer, "lora_A", None)):
                weight = to_cpu_full_param(getattr(lora_a, "weight", None))
                if weight is not None:
                    lora_state[
                        f"layer_{index}.lora_A.{tag}.weight"
                    ] = weight

            for tag, lora_b in _lora_items(getattr(layer, "lora_B", None)):
                weight = to_cpu_full_param(getattr(lora_b, "weight", None))
                if weight is not None:
                    lora_state[
                        f"layer_{index}.lora_B.{tag}.weight"
                    ] = weight

    base_filename = f"{prefix.replace('.', '_')}_base.pt"
    lora_filename = f"{prefix.replace('.', '_')}_lora.pt"
    torch.save(base_state, os.path.join(output_dir, base_filename))
    torch.save(lora_state, os.path.join(output_dir, lora_filename))
    print(
        f"[SAVE] Projector parts saved: {base_filename}, {lora_filename}"
    )


def _get_nested_module(root: nn.Module, path: str):
    module = root
    for name in path.split("."):
        module = getattr(module, name)
    return module


class POITrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        step = self.state.global_step
        checkpoint_name = f"checkpoint-{step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)

        try:
            torch.save(
                self.model.state_dict(),
                os.path.join(output_dir, "pytorch_model.bin"),
            )
        except Exception as error:
            print(f"[WARN] Could not save pytorch_model.bin: {error!r}")

        optimizer_path = os.path.join(output_dir, "optimizer.pt")
        scheduler_path = os.path.join(output_dir, "scheduler.pt")

        try:
            optimizer_state = (
                self.optimizer.state_dict()
                if self.optimizer is not None
                else {"_empty": True}
            )
            torch.save(optimizer_state, optimizer_path)
        except Exception as error:
            print(f"[WARN] Could not save optimizer state: {error!r}")
            torch.save({"_saved_via_stub": True}, optimizer_path)

        try:
            scheduler_state = (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else {"_empty": True}
            )
            torch.save(scheduler_state, scheduler_path)
        except Exception as error:
            print(f"[WARN] Could not save scheduler state: {error!r}")
            torch.save({"_saved_via_stub": True}, scheduler_path)

        try:
            self.state.save_to_json(
                os.path.join(output_dir, "trainer_state.json")
            )
        except Exception as error:
            print(f"[WARN] Could not save trainer state: {error!r}")

        try:
            with open(
                os.path.join(self.args.output_dir, "latest"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write(checkpoint_name)
        except Exception:
            pass

        try:
            self._rotate_checkpoints(
                use_mtime=True,
                output_dir=self.args.output_dir,
            )
        except Exception as error:
            print(f"[WARN] Could not rotate checkpoints: {error!r}")

        projector_paths = [
            "linear_mapping_layer",
            "traffic_mapping_layer",
            "gps_mapping_layer",
            "base_model.linear_mapping_layer",
            "base_model.traffic_mapping_layer",
            "base_model.gps_mapping_layer",
        ]

        saved_modules = set()
        for path in projector_paths:
            try:
                projector = _get_nested_module(self.model, path)
            except (AttributeError, TypeError):
                continue

            if not isinstance(projector, nn.Sequential):
                continue
            if id(projector) in saved_modules:
                continue
            saved_modules.add(id(projector))

            filename = f"{path.replace('.', '_')}_combined.pt"
            try:
                save_combined_linear_layer(
                    projector,
                    output_dir,
                    filename,
                    self.args.lora_alpha,
                    self.args.lora_r,
                )
            except Exception as error:
                print(
                    f"[WARN] Could not save merged projector "
                    f"{filename}: {error!r}"
                )

            try:
                save_split_projector_weights(projector, output_dir, path)
            except Exception as error:
                print(
                    f"[WARN] Could not save projector parts for "
                    f"{path}: {error!r}"
                )

        try:
            embedding = self.model.get_input_embeddings()
            geohash_layer = getattr(
                embedding,
                "geohash_embedding_layer",
                None,
            )
            if geohash_layer is not None:
                geohash_path = os.path.join(
                    output_dir,
                    "geohash_embedding.pt",
                )
                torch.save(geohash_layer.state_dict(), geohash_path)
                print(f"[SAVE] Geohash embedding saved: {geohash_path}")
            else:
                print("[WARN] Geohash embedding layer was not found.")
        except Exception as error:
            print(f"[WARN] Could not save geohash embedding: {error!r}")

        try:
            if isinstance(self.model.base_model, PeftModel):
                self.model.base_model.save_pretrained(output_dir)
                print(f"[SAVE] PEFT adapters saved: {output_dir}")
        except Exception as error:
            print(f"[WARN] Could not save PEFT adapters: {error!r}")


def _validate_consecutive_ids(
    token_ids: List[int],
    name: str,
) -> int:
    if not token_ids:
        raise ValueError(f"No {name} tokens were created.")

    start_id = min(token_ids)
    expected = list(range(start_id, start_id + len(token_ids)))
    if token_ids != expected:
        raise ValueError(
            f"{name} token IDs are not consecutive or are in the wrong order."
        )
    return start_id


def train() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    required_paths = {
        "--poi_embedding_path": data_args.poi_embedding_path,
        "--prepacked_pt": data_args.prepacked_pt,
        "--traffic_compact_pt": data_args.traffic_compact_pt,
        "--geohash_vocab_path": data_args.geohash_vocab_path,
    }
    missing = [name for name, value in required_paths.items() if not value]
    if missing:
        raise ValueError(
            "Missing required arguments: " + ", ".join(missing)
        )

    replace_llama_attn(
        training_args.use_flash_attn,
        training_args.use_full_attn,
    )

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    original_context_length = getattr(
        config,
        "max_position_embeddings",
        None,
    )
    if (
        original_context_length
        and training_args.model_max_length > original_context_length
    ):
        factor = float(
            math.ceil(
                training_args.model_max_length / original_context_length
            )
        )
        config.rope_scaling = {"type": "linear", "factor": factor}

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.resize_token_embeddings(len(tokenizer))

    device = base_model.device
    model_embedding_dim = base_model.get_input_embeddings().weight.shape[1]

    poi_embeddings = load_poi_embeddings(data_args.poi_embedding_path)
    num_poi_tokens = int(poi_embeddings.shape[0])

    traffic_compact, traffic_dim, station_ids = load_traffic_compact(
        data_args.traffic_compact_pt
    )
    num_traffic_tokens = len(station_ids)

    with open(
        data_args.geohash_vocab_path,
        "r",
        encoding="utf-8",
    ) as file:
        geohash_vocab = json.load(file)
    geohash2id = geohash_vocab["geohash2id"]
    num_gps_tokens = max(int(value) for value in geohash2id.values())

    poi_tokens = [f"<POI {index}>" for index in range(num_poi_tokens)]
    traffic_tokens = [
        f"<TRAFFIC {station_id}>" for station_id in station_ids
    ]
    gps_tokens = [
        f"<GPS {index}>" for index in range(1, num_gps_tokens + 1)
    ]

    smart_tokenizer_and_embedding_resize(
        {
            "additional_special_tokens": (
                poi_tokens + traffic_tokens + gps_tokens
            )
        },
        tokenizer,
        base_model,
    )

    poi_token_start_id = _validate_consecutive_ids(
        tokenizer.convert_tokens_to_ids(poi_tokens),
        "POI",
    )
    traffic_token_start_id = _validate_consecutive_ids(
        tokenizer.convert_tokens_to_ids(traffic_tokens),
        "traffic",
    )
    gps_token_start_id = _validate_consecutive_ids(
        tokenizer.convert_tokens_to_ids(gps_tokens),
        "GPS",
    )

    linear_mapping_layer = nn.Sequential(
        nn.Linear(128, model_embedding_dim),
        nn.GELU(),
        nn.Linear(model_embedding_dim, model_embedding_dim),
    ).to(device)
    traffic_mapping_layer = nn.Sequential(
        nn.Linear(traffic_dim, model_embedding_dim),
        nn.GELU(),
        nn.Linear(model_embedding_dim, model_embedding_dim),
    ).to(device)
    gps_mapping_layer = nn.Sequential(
        nn.Linear(128, model_embedding_dim),
        nn.GELU(),
        nn.Linear(model_embedding_dim, model_embedding_dim),
    ).to(device)

    base_model.add_module("linear_mapping_layer", linear_mapping_layer)
    base_model.add_module(
        "traffic_mapping_layer",
        traffic_mapping_layer,
    )
    base_model.add_module("gps_mapping_layer", gps_mapping_layer)

    poi_embedding_layer = nn.Embedding(
        num_embeddings=num_poi_tokens,
        embedding_dim=128,
    )
    initialize_embedding_layer_from_array(
        poi_embedding_layer,
        poi_embeddings,
        device,
    )
    for parameter in poi_embedding_layer.parameters():
        parameter.requires_grad = False

    geohash_embedding_layer = nn.Embedding(
        num_embeddings=num_gps_tokens + 1,
        embedding_dim=128,
    ).to(device)

    poi_projector = base_model.linear_mapping_layer
    traffic_projector = base_model.traffic_mapping_layer
    gps_projector = base_model.gps_mapping_layer

    for name, parameter in base_model.named_parameters():
        parameter.requires_grad = name.startswith(
            (
                "linear_mapping_layer",
                "traffic_mapping_layer",
                "gps_mapping_layer",
            )
        )

    if training_args.low_rank_training:
        lora_config = LoraConfig(
            r=max(1, int(training_args.lora_r)),
            lora_alpha=float(training_args.lora_alpha),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, lora_config)
        base_model.print_trainable_parameters()

    apply_embedding_hook_prepacked(
        base_model=base_model,
        poi_embedding_layer=poi_embedding_layer,
        poi_token_start_id=poi_token_start_id,
        num_poi_tokens=num_poi_tokens,
        traffic_compact=traffic_compact,
        traffic_token_start_id=traffic_token_start_id,
        num_traffic_tokens=num_traffic_tokens,
        gps_token_start_id=gps_token_start_id,
        num_gps_tokens=num_gps_tokens,
        linear_mapping_layer=poi_projector,
        traffic_mapping_layer=traffic_projector,
        gps_mapping_layer=gps_projector,
        geohash_embedding_layer=geohash_embedding_layer,
    )

    model = TrafficPOIWrapper(base_model)
    prepacked_dataset = PrepackedDataset(data_args.prepacked_pt)

    if (
        prepacked_dataset.traffic_token_start_id
        != traffic_token_start_id
    ):
        raise ValueError(
            "Traffic token start ID mismatch. The special-token order "
            "or count differs from the prepacking step."
        )

    if prepacked_dataset.station_ids != station_ids:
        raise ValueError(
            "Station IDs differ between prepacked_train.pt and "
            "traffic_compact.pt."
        )

    valid_indices = prepacked_dataset.nonempty_indices()
    if not valid_indices:
        raise ValueError("The prepacked dataset contains no valid samples.")

    train_dataset = Subset(prepacked_dataset, valid_indices)
    data_collator = DataCollatorPrepacked(
        pad_id=tokenizer.pad_token_id,
        truncation_side="left",
        max_len=training_args.model_max_length,
    )

    print(
        f"[INFO] Training with {len(valid_indices)} valid samples "
        f"out of {len(prepacked_dataset)}."
    )

    model.base_model.config.use_cache = False
    model.base_model.enable_input_require_grads()
    model.base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainer = POITrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("[INFO] Training started.")
    trainer.train(resume_from_checkpoint=False)

    geohash_output_path = os.path.join(
        training_args.output_dir,
        "geohash_embedding.pt",
    )
    torch.save(
        geohash_embedding_layer.state_dict(),
        geohash_output_path,
    )
    print(f"[SAVE] Geohash embedding saved: {geohash_output_path}")

    tokenizer.save_pretrained(training_args.output_dir)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    try:
        if isinstance(model.base_model, PeftModel):
            model.base_model.save_pretrained(training_args.output_dir)
            print(
                f"[SAVE] PEFT adapters saved: "
                f"{training_args.output_dir}"
            )
    except Exception as error:
        print(f"[WARN] Could not save PEFT adapters: {error!r}")

    lora_path = os.path.join(
        training_args.output_dir,
        "projectors_lora.pt",
    )
    torch.save(
        {
            key: value
            for key, value in model.state_dict().items()
            if "lora_" in key
        },
        lora_path,
    )

    save_combined_linear_layer(
        model.base_model.linear_mapping_layer,
        training_args.output_dir,
        "linear_mapping_layer_combined.pt",
        alpha=training_args.lora_alpha,
        rank=training_args.lora_r,
    )
    save_combined_linear_layer(
        model.base_model.traffic_mapping_layer,
        training_args.output_dir,
        "traffic_mapping_layer_combined.pt",
        alpha=training_args.lora_alpha,
        rank=training_args.lora_r,
    )
    save_combined_linear_layer(
        model.base_model.gps_mapping_layer,
        training_args.output_dir,
        "gps_mapping_layer_combined.pt",
        alpha=training_args.lora_alpha,
        rank=training_args.lora_r,
    )

    torch.cuda.empty_cache()
    print("[INFO] Training completed.")


if __name__ == "__main__":
    train()
