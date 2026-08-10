import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import transformers

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
BIST_ROOT = PROJECT_ROOT / "BiST"
BIST_SRC = BIST_ROOT / "src"

if not BIST_SRC.is_dir():
    raise FileNotFoundError(
        f"BiST source directory not found: {BIST_SRC}. "
        "Place the code and BiST directories under the same parent directory."
    )

sys.path.insert(0, str(BIST_ROOT))

from src.models.bist import BiST, MLP
from src.utils.args import get_config as base_get_config
from src.utils.dataloader import get_dataset_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = -100
GRID_MINUTES = 5
TRAFFIC_NUM_RE = re.compile(r"<\s*TRAFFIC\s+(\d+)\s*>", re.IGNORECASE)


def build_bist_args_for_inference(dataset: str, device_name: str):
    parser = base_get_config()
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--model_dim", type=int, default=32)
    parser.add_argument("--prompt_dim", type=int, default=32)
    parser.add_argument("--lrate", type=float, default=2e-3)
    parser.add_argument("--wdecay", type=float, default=1e-5)
    parser.add_argument("--clip_grad_value", type=float, default=5)
    args, _ = parser.parse_known_args([])

    args.model_name = "BiST"
    args.dataset = dataset
    args.years = "2012"
    args.input_dim = 3
    args.output_dim = 1
    args.seq_len = 12
    args.horizon = 12
    args.bs = 128
    args.device = device_name
    args.mode = "train"
    args.core = 1
    args.same = 0
    args.ct = 0
    args.max_epochs = 500
    args.patience = 20
    return args


def make_bist_model_from_args(
    args,
    device_obj: torch.device = device,
    ckpt_path: Optional[str] = None,
):
    _, tod_size, node_num = get_dataset_info(args.dataset)
    args.node_num = node_num

    base = MLP(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        num_layer=args.num_layer,
        model_dim=args.model_dim,
        prompt_dim=args.prompt_dim,
        tod_size=tod_size,
        kernel_size=args.kernel_size,
    )
    model_dim_total = args.model_dim + 3 * args.prompt_dim
    model = BiST(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        model_args=vars(args),
        stmodel=base,
        dim=[model_dim_total, model_dim_total],
        core=args.core,
    ).to(device_obj)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device_obj)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)

    model.eval()
    return model


def jload(path: str):
    with open(path, "r") as f:
        return json.load(f)


def adjust_ts(ts_str: str) -> str:
    dt = pd.to_datetime(str(ts_str).strip(), errors="raise")
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.tz_localize(None)
    dt = dt.floor(f"{GRID_MINUTES}min")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def find_t_idx(
    ts_str: str,
    ts2idx: Dict[str, int],
    uniq_dt: np.ndarray,
) -> int:
    if len(uniq_dt) == 0:
        raise ValueError("uniq_dt is empty. Cannot match timestamps.")
    if ts_str in ts2idx:
        return ts2idx[ts_str]

    target = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    matches = np.where(uniq_dt == target)[0]
    if len(matches) == 1:
        return int(matches[0])

    uniq64 = uniq_dt.astype("datetime64[s]")
    target64 = np.datetime64(target, "s")
    pos = int(np.searchsorted(uniq64, target64, side="right")) - 1
    return max(pos, 0)


def strip_prefix(value: str, prefix: str) -> str:
    if value is None:
        return ""
    stripped = value.strip()
    if stripped.lower().startswith(prefix.lower()):
        return stripped[len(prefix) :].strip()
    return stripped


def normalize_qa_items(raw_items: List[dict]) -> List[dict]:
    return [
        {
            "question": strip_prefix(item.get("question", ""), "<question>:"),
            "answer": strip_prefix(item.get("answer", ""), "<answer>:"),
        }
        for item in raw_items
    ]


def load_qa_items_any(path: str) -> List[dict]:
    if path.lower().endswith(".json"):
        items = jload(path)
        if not isinstance(items, list):
            raise ValueError(f"QA JSON must be a list, got {type(items)}")
        return normalize_qa_items(items)

    qa_items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            anchor = line.lower().find("<answer>:")
            if anchor == -1:
                continue
            question = strip_prefix(line[:anchor].rstrip(), "<question>:")
            answer = strip_prefix(line[anchor:].lstrip(), "<answer>:")
            qa_items.append({"question": question, "answer": answer})
    return qa_items


class TrafficOnDemand:
    def __init__(
        self,
        data_path: str,
        ckpt_path: str,
        train_time_csv: str,
        val_time_csv: str,
        test_time_csv: str,
        batch_size: int = 8,
        extra_shift_minutes: int = 0,
        bist_dataset: str = "CA",
    ):
        self.C = 1
        self.num_his = 12
        self.num_pred = 12

        bist_args = build_bist_args_for_inference(
            dataset=bist_dataset,
            device_name=str(device),
        )
        self.model = make_bist_model_from_args(
            bist_args,
            device_obj=device,
            ckpt_path=ckpt_path,
        )
        self.model.eval()
        self.C = getattr(bist_args, "input_dim", self.C)

        npz = np.load(data_path, allow_pickle=True)
        if "data" not in npz.files:
            raise ValueError(f"NPZ({data_path}) has no 'data' key.")
        if "stations" not in npz.files:
            raise ValueError(f"NPZ({data_path}) has no 'stations' key.")

        self.data_all = npz["data"].astype(np.float32)
        stations = npz["stations"]
        stations = stations.tolist() if hasattr(stations, "tolist") else list(stations)
        self.station_ids = [str(value).strip() for value in stations]
        self.N = len(self.station_ids)

        base_dir = os.path.dirname(data_path)
        self.idx_train = np.load(os.path.join(base_dir, "idx_train.npy"))
        self.idx_val = np.load(os.path.join(base_dir, "idx_val.npy"))
        self.idx_test = np.load(os.path.join(base_dir, "idx_test.npy"))

        total_steps = self.data_all.shape[0]

        def build_hist_idx(idx_global: np.ndarray, history: int) -> np.ndarray:
            hist_idx = np.zeros((len(idx_global), history), dtype=np.int64)
            for i, end_value in enumerate(idx_global):
                end_t = int(end_value)
                start_t = end_t - (history - 1)
                if start_t < 0 or end_t >= total_steps:
                    raise ValueError(
                        f"idx_global[{i}]={end_t} yields history [{start_t}, {end_t}] "
                        f"outside T_total={total_steps}."
                    )
                hist_idx[i] = np.arange(start_t, end_t + 1, dtype=np.int64)
            return hist_idx

        self.hist_idx_train = build_hist_idx(self.idx_train, self.num_his)
        self.hist_idx_val = build_hist_idx(self.idx_val, self.num_his)
        self.hist_idx_test = build_hist_idx(self.idx_test, self.num_his)

        def read_and_reindex_ts(csv_path: str) -> np.ndarray:
            df = pd.read_csv(csv_path)

            def pick_timestamp_col(frame: pd.DataFrame) -> str:
                columns = list(frame.columns)
                for column in columns:
                    if str(column).strip().lower() == "timestamp":
                        return column

                hints = ("timestamp", "time", "datetime", "date")
                hinted = [
                    column
                    for column in columns
                    if any(key in str(column).strip().lower() for key in hints)
                ]
                candidates = hinted if hinted else columns
                best_col = None
                best_score = -1.0
                for column in candidates:
                    parsed = pd.to_datetime(
                        frame[column].astype(str), errors="coerce", utc=False
                    )
                    score = parsed.notna().mean()
                    if score > best_score:
                        best_score = score
                        best_col = column
                if best_col is None or best_score < 0.5:
                    raise ValueError(
                        f"{csv_path}: no timestamp-like column found. "
                        f"columns={columns}, best parse rate={best_score:.3f}"
                    )
                return best_col

            raw = df[pick_timestamp_col(df)].astype(str)
            ts = pd.to_datetime(raw, errors="coerce").dropna()
            ts = ts.drop_duplicates().sort_values()
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_localize(None)
            ts = ts.dt.floor(f"{GRID_MINUTES}min")
            return ts.drop_duplicates().sort_values().to_numpy()

        train_ts = read_and_reindex_ts(train_time_csv)
        val_ts = read_and_reindex_ts(val_time_csv)
        test_ts = read_and_reindex_ts(test_time_csv)
        global_ts = pd.to_datetime(np.concatenate([train_ts, val_ts, test_ts]))

        formatted_ts = global_ts.strftime("%Y-%m-%d %H:%M:%S").tolist()
        self.ts2idx = {timestamp: i for i, timestamp in enumerate(formatted_ts)}
        self.uniq_dt = np.array(
            [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in formatted_ts]
        )

        self.t_to_loc: Dict[int, Tuple[str, int]] = {}

        def fill_t_to_loc(split_name: str, hist_idx_split: np.ndarray):
            last_h = hist_idx_split.shape[1] - 1
            for i in range(hist_idx_split.shape[0]):
                t_last = int(hist_idx_split[i, last_h])
                if t_last not in self.t_to_loc:
                    self.t_to_loc[t_last] = (split_name, i)

        fill_t_to_loc("train", self.hist_idx_train)
        fill_t_to_loc("val", self.hist_idx_val)
        fill_t_to_loc("test", self.hist_idx_test)

        self.batch_size = batch_size
        self.extra_shift_minutes = int(extra_shift_minutes)

    def forward_group(self, split_name: str, idx_list: List[int]) -> torch.Tensor:
        if not idx_list:
            return torch.empty(0)

        if split_name == "train":
            hist_idx_split = self.hist_idx_train
        elif split_name == "val":
            hist_idx_split = self.hist_idx_val
        else:
            hist_idx_split = self.hist_idx_test

        batch_size = len(idx_list)
        history = self.num_his
        values = np.zeros((batch_size, history, self.N, self.C), dtype=np.float32)
        for batch_idx, sample_idx in enumerate(idx_list):
            t_seq = hist_idx_split[sample_idx]
            values[batch_idx] = self.data_all[t_seq][:, : self.N, : self.C]

        inputs = torch.from_numpy(values).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        labels = torch.zeros(
            (batch_size, self.num_pred, self.N, self.C),
            dtype=torch.float32,
            device=device,
        )
        with torch.inference_mode():
            hidden, _, _ = self.model.stmodel(inputs, labels)

        return hidden.detach().cpu()


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    count = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if count > 0:
        input_weights = model.get_input_embeddings().weight.data
        output_weights = model.get_output_embeddings().weight.data
        input_weights[-count:] = input_weights[:-count].mean(dim=0, keepdim=True)
        output_weights[-count:] = output_weights[:-count].mean(dim=0, keepdim=True)


def rewrite_traffic_tokens_with_original_ids(text: str, reverse: List[str]) -> str:
    def replace(match):
        index = int(match.group(1))
        if 0 <= index < len(reverse):
            return f"<TRAFFIC {reverse[index]}>"
        return match.group(0)

    return TRAFFIC_NUM_RE.sub(replace, text)


def infer_sidecar_path(qa_json_path: str) -> str:
    if qa_json_path.endswith(".json"):
        return qa_json_path[:-5] + "_traffic_pairs.json"
    return qa_json_path + "_traffic_pairs.json"


def infer_map_path_from_dataset(qa_json_path: str) -> str:
    return os.path.join(os.path.dirname(qa_json_path), "traffic_id_map.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_json", required=True)
    parser.add_argument("--traffic_pairs_json", default=None)
    parser.add_argument("--traffic_id_map_json", default=None)
    parser.add_argument("--poi_embedding_npy", required=True)
    parser.add_argument(
        "--model_name", default="yukang/Llama-2-7b-longlora-32k-ft"
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--train_time_csv", required=True)
    parser.add_argument("--val_time_csv", required=True)
    parser.add_argument("--test_time_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--extra_shift_minutes", type=int, default=0)
    parser.add_argument("--bist_dataset", required=True)
    parser.add_argument("--geohash_vocab_path", required=True)
    args = parser.parse_args()

    if args.bist_dataset.strip().lower() != "ca":
        raise ValueError("--bist_dataset must be CA.")

    os.makedirs(args.out_dir, exist_ok=True)

    qa_items = load_qa_items_any(args.qa_json)
    traffic_pairs_per_qa = jload(
        args.traffic_pairs_json or infer_sidecar_path(args.qa_json)
    )
    if len(traffic_pairs_per_qa) != len(qa_items):
        count = min(len(traffic_pairs_per_qa), len(qa_items))
        qa_items = qa_items[:count]
        traffic_pairs_per_qa = traffic_pairs_per_qa[:count]

    map_path = args.traffic_id_map_json or infer_map_path_from_dataset(args.qa_json)
    reverse = jload(map_path).get("reverse", [])
    if not reverse:
        raise ValueError(f"reverse list missing in '{map_path}'.")

    if args.max_examples is not None and args.max_examples > 0:
        count = min(args.max_examples, len(qa_items))
        qa_items = qa_items[:count]
        traffic_pairs_per_qa = traffic_pairs_per_qa[:count]

    ondm = TrafficOnDemand(
        data_path=args.data_path,
        ckpt_path=args.ckpt,
        train_time_csv=args.train_time_csv,
        val_time_csv=args.val_time_csv,
        test_time_csv=args.test_time_csv,
        batch_size=args.batch_size,
        extra_shift_minutes=args.extra_shift_minutes,
        bist_dataset=args.bist_dataset,
    )

    station_ids = ondm.station_ids
    station_id2offset = {str(station_id): i for i, station_id in enumerate(station_ids)}

    num_poi = np.load(args.poi_embedding_npy).shape[0]
    base = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        padding_side="right",
    )

    with open(args.geohash_vocab_path, "r") as f:
        geohash2id = json.load(f)["geohash2id"]
    num_gps_tokens = max(int(value) for value in geohash2id.values())

    poi_tokens = [f"<POI {i}>" for i in range(num_poi)]
    traffic_tokens = [f"<TRAFFIC {station_id}>" for station_id in station_ids]
    gps_tokens = [f"<GPS {i}>" for i in range(1, num_gps_tokens + 1)]
    smart_tokenizer_and_embedding_resize(
        {"additional_special_tokens": poi_tokens + traffic_tokens + gps_tokens},
        tokenizer,
        base,
    )

    traffic_token_ids = tokenizer.convert_tokens_to_ids(traffic_tokens)
    traffic_start = min(traffic_token_ids)
    expected_ids = list(range(traffic_start, traffic_start + len(traffic_tokens)))
    if sorted(traffic_token_ids) != expected_ids:
        raise ValueError("Traffic token IDs are not consecutive.")

    used_pairs = set()
    for pairs in traffic_pairs_per_qa:
        for pair in pairs:
            index = int(pair.get("traffic_k"))
            if not 0 <= index < len(reverse):
                continue
            station = station_id2offset.get(str(reverse[index]))
            if station is None:
                continue

            dt_adj = datetime.strptime(
                adjust_ts(str(pair["ts"])),
                "%Y-%m-%d %H:%M:%S",
            )
            if ondm.extra_shift_minutes != 0:
                dt_adj += timedelta(minutes=ondm.extra_shift_minutes)
            timestamp = dt_adj.strftime("%Y-%m-%d %H:%M:%S")
            time_idx = find_t_idx(
                timestamp,
                ondm.ts2idx,
                ondm.uniq_dt,
            )
            if time_idx >= 0:
                used_pairs.add((time_idx, station))

    keys = sorted((time_idx, station) for time_idx, station in used_pairs if time_idx >= 0)
    pair2cid = {key: i for i, key in enumerate(keys)}
    if not pair2cid:
        raise ValueError("No matched (time, station) pairs.")

    by_sample: Dict[Tuple[str, int], List[Tuple[int, int]]] = defaultdict(list)
    zero_cids_per_station: Dict[int, List[int]] = defaultdict(list)

    for (time_idx, station), cid in pair2cid.items():
        location = ondm.t_to_loc.get(time_idx)
        if location is None:
            previous = time_idx - 1
            while previous >= 0:
                location = ondm.t_to_loc.get(previous)
                if location is not None:
                    break
                previous -= 1
            if location is None:
                zero_cids_per_station[station].append(cid)
                continue

        split, sample_idx = location
        by_sample[(split, sample_idx)].append((station, cid))

    probe_split, probe_idx = next(iter(by_sample.keys()))
    probe = ondm.forward_group(probe_split, [probe_idx])
    embedding_dim = int(probe.shape[-1])
    del probe

    compact = torch.empty((len(keys), embedding_dim), dtype=torch.float32)
    for split in ("train", "val", "test"):
        items = [
            (sample_idx, requests)
            for (sample_split, sample_idx), requests in by_sample.items()
            if sample_split == split
        ]
        for start in range(0, len(items), ondm.batch_size):
            chunk = items[start : start + ondm.batch_size]
            sample_indices = [sample_idx for sample_idx, _ in chunk]
            embeddings = ondm.forward_group(split, sample_indices)
            for batch_idx, (_, requests) in enumerate(chunk):
                for station, cid in requests:
                    compact[cid] = embeddings[batch_idx, 0, station, :]
            del embeddings

    if zero_cids_per_station:
        inputs = torch.zeros(
            (1, ondm.num_his, ondm.N, ondm.C),
            dtype=torch.float32,
            device=device,
        )
        labels = torch.zeros(
            (1, ondm.num_pred, ondm.N, ondm.C),
            dtype=torch.float32,
            device=device,
        )
        with torch.inference_mode():
            hidden, _, _ = ondm.model.stmodel(inputs, labels)
        zero_embeddings = hidden[0, 0, :, :].cpu()
        for station, cids in zero_cids_per_station.items():
            for cid in cids:
                compact[cid] = zero_embeddings[station]

    if not torch.isfinite(compact).all():
        raise ValueError("NaN or Inf detected in the compact embedding table.")
    if compact.numel() == 0 or compact.abs().sum().item() == 0.0:
        raise ValueError("The compact embedding table is empty or all zero.")

    del ondm.model
    del base
    torch.cuda.empty_cache()

    input_ids_list = []
    labels_list = []
    traffic_pair_idx_list = []
    eos = tokenizer.eos_token or "</s>"

    for example, pairs in zip(qa_items, traffic_pairs_per_qa):
        question = rewrite_traffic_tokens_with_original_ids(
            example["question"], reverse
        )
        source = "<question>:" + question
        target = "<answer>:" + example["answer"] + eos

        source_pack = tokenizer(source, return_tensors="pt", truncation=False)
        full_pack = tokenizer(source + target, return_tensors="pt", truncation=False)
        input_ids = full_pack.input_ids[0]
        source_ids = source_pack.input_ids[0]
        labels = input_ids.clone()
        source_len = source_ids.size(0)
        labels[:source_len] = IGNORE_INDEX

        occurrences = []
        for pair in pairs:
            index = int(pair.get("traffic_k", -1))
            if not 0 <= index < len(reverse):
                continue
            station = station_id2offset.get(str(reverse[index]))
            if station is None:
                continue

            dt_adj = datetime.strptime(
                adjust_ts(str(pair["ts"])),
                "%Y-%m-%d %H:%M:%S",
            )
            if ondm.extra_shift_minutes != 0:
                dt_adj += timedelta(minutes=ondm.extra_shift_minutes)
            timestamp = dt_adj.strftime("%Y-%m-%d %H:%M:%S")
            time_idx = find_t_idx(
                timestamp,
                ondm.ts2idx,
                ondm.uniq_dt,
            )
            occurrences.append((time_idx, station))

        queues = defaultdict(deque)
        for time_idx, station in occurrences:
            if time_idx >= 0 and (time_idx, station) in pair2cid:
                queues[station].append(pair2cid[(time_idx, station)])

        pair_indices = torch.full_like(input_ids, fill_value=-1)
        for token_idx in range(source_len):
            token_id = int(input_ids[token_idx])
            if traffic_start <= token_id < traffic_start + len(station_ids):
                station = token_id - traffic_start
                if queues[station]:
                    pair_indices[token_idx] = queues[station].popleft()

        input_ids_list.append(input_ids.cpu())
        labels_list.append(labels.cpu())
        traffic_pair_idx_list.append(pair_indices.cpu())

    torch.save(
        {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "traffic_pair_idx": traffic_pair_idx_list,
            "tokenizer_name": args.model_name,
            "station_ids": station_ids,
            "poi_count": int(num_poi),
            "traffic_token_start_id": int(traffic_start),
        },
        os.path.join(args.out_dir, "prepacked_train.pt"),
    )
    torch.save(
        {
            "traffic_compact": compact,
            "pair2cid": pair2cid,
            "Dg": int(embedding_dim),
            "station_ids": station_ids,
        },
        os.path.join(args.out_dir, "traffic_compact.pt"),
    )


if __name__ == "__main__":
    main()
