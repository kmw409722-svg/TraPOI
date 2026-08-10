import os
import json
import argparse
import io
import sys
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os.path as osp
import glob
import geohash2 as geohash
from functools import wraps
import torch
from types import SimpleNamespace as NS
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch import cat
import math
import inspect
from collections import defaultdict as dd, Counter

# Configure paths before running this script.
REPO_ROOT = "YOUR_SPATIO_TEMPORAL_HYPERGRAPH_MODEL_PATH"
TA_LLM_ROOT = "YOUR_TA_LLM_ROOT_PATH"


def resolve_required_directory(path_value: str, path_name: str) -> str:
    if not path_value or path_value.startswith("YOUR_"):
        raise ValueError(
            f"Set {path_name} to a valid directory before running this script."
        )

    resolved = os.path.abspath(os.path.expanduser(path_value))
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"{path_name} directory not found: {resolved}")

    return resolved


REPO_ROOT = resolve_required_directory(REPO_ROOT, "REPO_ROOT")
TA_LLM_ROOT = resolve_required_directory(TA_LLM_ROOT, "TA_LLM_ROOT")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from contextlib import contextmanager


@contextmanager
def in_repo():
    prev = os.getcwd()
    os.chdir(REPO_ROOT)

    try:
        yield
    finally:
        os.chdir(prev)

# Import project-specific preprocessing, dataset, and model components.
with in_repo():
    from preprocess import preprocess
    from utils import seed_torch, set_logger, Cfg, count_parameters, test_step, save_model
    from layer import NeighborSampler
    from dataset import LBSNDataset
    from model import STHGCN, SequentialTransformer

try:
    from layer import HypergraphTransformer
except ImportError:
    from model.layer import HypergraphTransformer


def encode_geohash(lat, lon, precision=7):
    if geohash is None:
        raise ImportError('Install Geohash library.')
    return geohash.encode(float(lat), float(lon), precision=precision)


# Format ranked POI candidates for downstream prompts.
def format_candidates_as_hint(candidates, gps_token_for_poi=None, max_items=None, prefix=None):
    if not candidates:
        return ''

    try:
        sorted_cands = sorted(candidates, key=lambda x: (x[1] is not None, x[1]), reverse=True)
    except Exception:
        sorted_cands = candidates

    if max_items is not None:
        sorted_cands = sorted_cands[:max_items]
    pieces = []

    for pid, _ in sorted_cands:
        try:
            pid_int = int(pid)
        except Exception:
            pid_int = pid
        gps_part = ''
        if gps_token_for_poi is not None:
            try:
                gps_tok = gps_token_for_poi(pid_int)
            except Exception:
                gps_tok = ''
            if gps_tok:
                gps_part = f', located at {gps_tok}'
        pieces.append(f'POI id {pid_int} <POI {pid_int}>{gps_part}')

    body = ', '.join(pieces)
    if prefix is None:
        prefix = 'Here are POI candidates that may help with the answer (sorted by score, descending). '

    return prefix + body + '.'


# Convert traffic station identifiers to contiguous model indices.
def build_contiguous_traffic_id_map(poi2traffic: dict):
    seen = set()
    ids = []

    for _, info in poi2traffic.items():
        if isinstance(info, dict):
            tids = [info.get('traffic_id', None)]
        elif isinstance(info, list):
            tids = [x.get('traffic_id', None) for x in info if isinstance(x, dict)]
        else:
            continue
        for tid in tids:
            if tid is None:
                continue
            tid_str = str(tid)
            if tid_str not in seen:
                seen.add(tid_str)
                ids.append(tid_str)

    def sort_key(x):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)

    ids = sorted(ids, key=sort_key)
    id_map = {orig: i for i, orig in enumerate(ids)}
    reverse = ids[:]

    return (id_map, reverse)


def save_traffic_id_map_json(path, id_map, reverse):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump({'map': id_map, 'reverse': reverse}, f, ensure_ascii=False)

    print(f'[SAVE] contiguous traffic id map -> {path} (size={len(id_map)})')


def load_poi_label_encoder(label_encoding_pkl_path):
    if not label_encoding_pkl_path or not os.path.exists(label_encoding_pkl_path):
        return (None, None)

    try:
        with open(label_encoding_pkl_path, 'rb') as f:
            items = pickle.load(f)
        poi_id_le = items[0]
        padding_poi_id = items[5] if len(items) >= 6 else 0
        return (poi_id_le, padding_poi_id)
    except Exception as e:
        print(f'[WARN] Failed to load label encoder from {label_encoding_pkl_path}: {e}')
        return (None, None)


def decode_poi_from_idx(poi_id, poi_id_le, padding_poi_id):
    if poi_id_le is None or pd.isna(poi_id):
        return poi_id

    try:
        pid = int(poi_id)
    except Exception:
        return poi_id

    classes = getattr(poi_id_le, 'classes_', None)
    if classes is None:
        return poi_id
    n = len(classes)

    try:
        if padding_poi_id is not None and int(padding_poi_id) == 0:
            if pid == 0:
                return None
            if 1 <= pid <= n:
                return classes[pid - 1]
        if 0 <= pid < n:
            return classes[pid]
    except Exception:
        pass

    return poi_id


def canonical_poi_key(s):
    if s is None:
        return ''
    s = str(s).strip()
    if s == '' or s.lower() == 'nan':
        return ''

    try:
        f = float(s)
        if float(f).is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return s


def _canon_series(s):
    return s.map(canonical_poi_key)


def normalize_poi2traffic_keys(poi2traffic):
    norm = {}

    for raw_k, info in poi2traffic.items():
        reps = set()
        try:
            reps.add(str(raw_k))
        except Exception:
            pass
        try:
            f = float(raw_k)
            if f.is_integer():
                reps.add(str(int(f)))
                reps.add(f'{f:.1f}')
            else:
                reps.add(str(f))
        except Exception:
            pass
        rep_list = list(reps)
        rep_list.sort(key=lambda x: (0, len(x)) if x.isdigit() else (1, len(x)))
        rep = canonical_poi_key(rep_list[0]) if rep_list else str(raw_k)
        merged = [rep] + [r for r in rep_list if r != rep]
        for rk in merged:
            if rk not in norm:
                norm[rk] = info

    return norm


def get_mapped_traffic_id_for_poi(poi_id, poi2traffic, traffic_id_map, poi_id_le=None, padding_poi_id=None):
    if poi2traffic is None:
        return []
    k1 = canonical_poi_key(poi_id)
    info = poi2traffic.get(k1)

    if info is None and poi_id_le is not None:
        decoded = decode_poi_from_idx(poi_id, poi_id_le, padding_poi_id)
        if decoded is not None:
            k2 = canonical_poi_key(decoded)
            info = poi2traffic.get(k2)
    if info is None:
        return []
    if isinstance(info, dict):
        raw_tids = [info.get('traffic_id')]
    elif isinstance(info, list):
        raw_tids = [x.get('traffic_id') for x in info if isinstance(x, dict)]
    else:
        return []
    mapped_ids = []

    for raw_tid in raw_tids:
        if raw_tid is None:
            continue
        mid = traffic_id_map.get(str(raw_tid))
        if mid is not None:
            mapped_ids.append(mid)

    mapped_ids = list(dict.fromkeys(mapped_ids))

    return mapped_ids

EARTH_RADIUS_M = 6371000.0
MAX_TRAFFIC_DISTANCE_M = 2000.0


# Build geohash mappings for POI coordinates.
def build_poi_geohash_file(train_data: pd.DataFrame, test_data: pd.DataFrame, out_path: str, precision: int=7):
    required = {'PoiId', 'Latitude', 'Longitude'}

    for need in required:
        if need not in train_data.columns or need not in test_data.columns:
            raise KeyError(f"build_poi_geohash_file: '{need}' column is required in both train/test CSVs.")

    cols = ['PoiId', 'Latitude', 'Longitude']
    poi_df = pd.concat([train_data[cols], test_data[cols]], ignore_index=True).drop_duplicates().dropna()
    mapping = {}

    for _, row in poi_df.iterrows():
        pid_raw = row['PoiId']
        lat = row['Latitude']
        lon = row['Longitude']
        pid = canonical_poi_key(pid_raw)
        try:
            gh = encode_geohash(lat, lon, precision=precision)
            mapping[pid] = gh
        except Exception as e:
            print(f'[WARN] Failed to encode geohash for POI {pid_raw} ({lat}, {lon}): {e}')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f'[GeoHash][SAVE] POI geohash (precision={precision}) -> {out_path} (size={len(mapping)})')


# Reconstruct the configuration and dataset used during training.
def build_training_like_env(args, MODEL_DEVICE):
    with in_repo():
        yaml_path = os.path.join(REPO_ROOT, 'conf', 'best_conf', f'{args.dataset_name}.yml')
        cfg = Cfg(yaml_path)
        if hasattr(cfg.model_args, 'sizes') and isinstance(cfg.model_args.sizes, str):
            cfg.model_args.sizes = [int(i) for i in cfg.model_args.sizes.split('-')]
        preprocess(cfg)
        lbsn_dataset = LBSNDataset(cfg)
        if not hasattr(cfg, 'dataset_args') or cfg.dataset_args is None:
            cfg.dataset_args = NS()
        cfg.dataset_args.spatial_slots = lbsn_dataset.spatial_slots
        cfg.dataset_args.num_user = lbsn_dataset.num_user
        cfg.dataset_args.num_poi = lbsn_dataset.num_poi
        cfg.dataset_args.num_category = lbsn_dataset.num_category
        cfg.dataset_args.padding_poi_id = lbsn_dataset.padding_poi_id
        cfg.dataset_args.padding_user_id = lbsn_dataset.padding_user_id
        cfg.dataset_args.padding_poi_category = lbsn_dataset.padding_poi_category
        cfg.dataset_args.padding_hour_id = lbsn_dataset.padding_hour_id
        cfg.dataset_args.padding_weekday_id = lbsn_dataset.padding_weekday_id
        if not hasattr(cfg, 'run_args') or cfg.run_args is None:
            cfg.run_args = NS()
        if not hasattr(cfg.run_args, 'num_workers'):
            cfg.run_args.num_workers = 0

    return (cfg, lbsn_dataset)


def _get_row_col(edge_index):
    from torch_sparse import SparseTensor
    if isinstance(edge_index, SparseTensor):
        return (edge_index.storage.row(), edge_index.storage.col())
    if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2 and (edge_index.size(0) == 2):
        return (edge_index[0], edge_index[1])
    raise TypeError(f'_get_row_col: unsupported edge_index type {type(edge_index)}')


def _get_src_dst(edge_index):
    from torch_sparse import SparseTensor

    if isinstance(edge_index, SparseTensor):
        dst = edge_index.storage.row()
        src = edge_index.storage.col()
        return src, dst

    if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2 and edge_index.size(0) == 2:
        return edge_index[0], edge_index[1]

    raise TypeError(f'_get_src_dst: unsupported edge_index type {type(edge_index)}')


def _split_is_traj_left(edge_index_ci2traj, split_idx):
    src, dst = _get_row_col(edge_index_ci2traj)
    traj_max = split_idx

    try:
        left_is_traj = (src <= traj_max).all().item() and (dst > traj_max).all().item()
        right_is_traj = (dst <= traj_max).all().item() and (src > traj_max).all().item()
        if left_is_traj:
            return True
        if right_is_traj:
            return False
    except Exception:
        pass

    return True


# Capture attention weights from the hypergraph transformer.
def _patch_hgt_for_alpha(layer_cls):
    if getattr(layer_cls, '_alpha_patched', False):
        return
    orig_forward = layer_cls.forward
    orig_message = layer_cls.message

    @wraps(orig_forward)
    def patched_forward(self, x, edge_index, edge_time_embed, edge_dist_embed, edge_type_embed, edge_attr_embed):
        try:
            self._last_edge_index = edge_index.detach().to('cpu')
        except Exception:
            self._last_edge_index = edge_index

        return orig_forward(self, x, edge_index, edge_time_embed, edge_dist_embed, edge_type_embed, edge_attr_embed)

    @wraps(orig_message)
    def patched_message(self, x, edge_attr_embed, edge_time_embed, edge_dist_embed, edge_type_embed, index, ptr, have_query_feature, size_i):
        x_j, x_i = x
        if self.trans_flag:
            if have_query_feature:
                x_i = self.lin_trans_x(x_i)
            x_j_raw = self.lin_trans_x(x_j)
            x_j = self.lin_trans_x(x_j)
        else:
            x_j_raw = x_j
        if edge_type_embed is not None:
            x_j = self.rel_transform(x_j, edge_type_embed)
        if self.time_fusion_mode == 'concat':
            x_j = cat([x_j, edge_time_embed, edge_dist_embed], dim=-1)
        elif self.time_fusion_mode == 'add':
            x_j = x_j + edge_time_embed + edge_dist_embed
        if edge_attr_embed is not None:
            if self.edge_fusion_mode == 'concat':
                x_j = cat([x_j, edge_attr_embed], dim=-1)
            else:
                x_j = x_j + edge_attr_embed
        key = self.lin_key(x_j).view(-1, self.attn_heads, self.out_channels)
        if not have_query_feature:
            query = self.att_r
            alpha = (key * query).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
        else:
            query = self.lin_query(x_i).view(-1, self.attn_heads, self.out_channels)
            alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)

        try:
            self._last_alpha = alpha.mean(dim=1).detach().to('cpu')
        except Exception:
            self._last_alpha = None

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if self.message_mode == 'node_edge':
            out = self.lin_value(x_j).view(-1, self.attn_heads, self.out_channels)
        else:
            out = self.lin_value(x_j_raw).view(-1, self.attn_heads, self.out_channels)
        out *= alpha.view(-1, self.attn_heads, 1)

        return out

    try:
        patched_message.__signature__ = inspect.signature(orig_message)
        patched_forward.__signature__ = inspect.signature(orig_forward)
    except Exception:
        pass

    layer_cls.forward = patched_forward
    layer_cls.message = patched_message
    layer_cls._alpha_patched = True


# Aggregate trajectory attention into ranked POI scores.
def _alpha_edges_to_poi_scores_tj2tj_neighbors(alpha_tj2tj, edge_index_tj2tj, ci2traj_edge_index, split_idx, poi_col_view, target_traj_idx, center_local_id=None, global2local=None):
    if alpha_tj2tj is None:
        return []

    src, dst = _get_src_dst(edge_index_tj2tj)
    center_id = int(target_traj_idx)

    if center_local_id is not None:
        center_id = int(center_local_id)
    elif global2local is not None and int(target_traj_idx) in global2local:
        center_id = int(global2local[int(target_traj_idx)])

    incoming_mask = (
        (dst == center_id)
        & (src <= split_idx)
        & (src != center_id)
    )
    neigh = []

    if incoming_mask.sum().item() > 0:
        neigh += list(
            zip(
                src[incoming_mask].detach().cpu().tolist(),
                alpha_tj2tj[incoming_mask].detach().cpu().numpy().tolist(),
            )
        )
    if not neigh:
        return []
    tj_weight = {}

    for tj, w in neigh:
        tj_weight[tj] = tj_weight.get(tj, 0.0) + float(w)

    src, dst = _get_row_col(ci2traj_edge_index)
    left_is_traj = _split_is_traj_left(ci2traj_edge_index, split_idx)
    tj_side, ci_side = (src, dst) if left_is_traj else (dst, src)
    tj2ci = dd(list)

    for a, b in zip(tj_side.detach().cpu().tolist(), ci_side.detach().cpu().tolist()):
        if a <= split_idx and b > split_idx:
            tj2ci[a].append(b)

    scores = Counter()

    for tj, w in tj_weight.items():
        for ci_gidx in tj2ci.get(tj, []):
            try:
                pid = int(poi_col_view[ci_gidx].item())
                scores[pid] += float(w)
            except Exception:
                continue

    return list(scores.most_common())


# Map raw trajectory identifiers to graph node indices.
def load_traj_mappers(dataset_name: str, repo_root: str=REPO_ROOT):
    prep_dir = os.path.join(repo_root, 'data', dataset_name, 'preprocessed')
    pkl_path = os.path.join(prep_dir, 'traj_raw2encoded.pkl')
    csv_path = os.path.join(prep_dir, 'traj_encoded2graph.csv')
    raw2enc = {}
    enc2graph = {}

    try:
        with open(pkl_path, 'rb') as f:
            raw2enc = pickle.load(f)
        raw2enc = {int(k): int(v) for k, v in raw2enc.items()}
    except Exception as e:
        print(f'[WARN] raw2enc not loaded from {pkl_path}: {e}')

    try:
        df = pd.read_csv(csv_path)
        c1 = 'encoded_traj_id'
        c2 = 'graph_traj_index'
        if c1 not in df.columns or c2 not in df.columns:
            raise ValueError(f"CSV must contain columns '{c1}', '{c2}'")
        enc2graph = {int(r[c1]): int(r[c2]) for _, r in df.iterrows()}
    except Exception as e:
        print(f'[WARN] enc2graph not loaded from {csv_path}: {e}')

    def raw_or_encoded_to_graph(traj_id_any):
        try:
            tid = int(traj_id_any)
        except Exception:
            raise ValueError(f'Invalid traj_id {traj_id_any!r}')

        enc = raw2enc.get(tid, None)
        if enc is None:
            raise ValueError(f'Encoded ID not found for traj_id {tid}, aborting operation.')
        g = enc2graph.get(enc, None)
        if g is None:
            raise ValueError(f'Graph traj index not found for encoded traj_id {enc} (raw={tid}). Check traj_encoded2graph.csv.')

        return g

    return (raw2enc, enc2graph, raw_or_encoded_to_graph)


# Load the trained STHGCN model and apply the attention patch.
def _load_sthgcn_and_patch(checkpoint_path, yaml_path=None, dataset_overrides: dict=None):
    try:
        with in_repo():
            _patch_hgt_for_alpha(HypergraphTransformer)
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            cfg = ckpt.get('cfg', None)
            if cfg is None and yaml_path is not None:
                cfg = Cfg(yaml_path)
                if hasattr(cfg.model_args, 'sizes') and isinstance(cfg.model_args.sizes, str):
                    cfg.model_args.sizes = [int(i) for i in cfg.model_args.sizes.split('-')]
            if cfg is None:
                print('[WARN] Missing model config. Attention candidates disabled.')
                return None
            if not hasattr(cfg, 'dataset_args') or cfg.dataset_args is None:
                cfg.dataset_args = NS()
            if dataset_overrides:
                for k, v in dataset_overrides.items():
                    setattr(cfg.dataset_args, k, v)
            if not hasattr(cfg, 'run_args') or cfg.run_args is None:
                cfg.run_args = NS()
            if not hasattr(cfg.run_args, 'gpu'):
                cfg.run_args.gpu = -1
            if not hasattr(cfg.run_args, 'device'):
                try:
                    g = int(getattr(cfg.run_args, 'gpu', -1))
                    cfg.run_args.device = f'cuda:{g}' if g >= 0 and torch.cuda.is_available() else 'cpu'
                except Exception:
                    cfg.run_args.device = 'cpu'
            model = STHGCN(cfg)
            model.load_state_dict(state, strict=False)
            model.eval()
            return model
    except Exception as e:
        print(f'[WARN] Model load failed: {e}')
        return None


def _safe_get_split_index(adjs_t):
    ei = adjs_t[1]
    if ei is None:
        raise ValueError('[split] adjs_t[1] is None; cannot compute split_index like run.py')
    if not hasattr(ei, 'storage'):
        raise TypeError('[split] adjs_t[1] must be a SparseTensor with .storage.row() like in run.py')
    return ei.storage.row().max().tolist()


# Build neighbor-sampling batches for target trajectories.
def make_batch_sampler(cfg, lbsn_dataset, target_traj_indices: list, MODEL_DEVICE, num_workers: int=0, mode: str='train'):
    sizes = cfg.model_args.sizes if isinstance(cfg.model_args.sizes, list) else [int(i) for i in str(cfg.model_args.sizes).split('-')]
    node_idx_list, max_time_list, label_list, kept_centers = ([], [], [], [])

    for t_idx in target_traj_indices:
        tid = int(t_idx)
        try:
            sidx, nidx, mxt, lbl, split_name = _pick_aligned_sample_for_traj(lbsn_dataset, target_traj_idx=tid, mode=mode)
            node_idx_list.append(int(nidx[0].item()))
            max_time_list.append(int(mxt[0].item()))
            label_list.append(lbl[0].view(1, -1))
            kept_centers.append(tid)
        except Exception as e:
            print(f'[WARN] center {tid}: _pick_aligned_sample_for_traj failed ({e}), skip this center.')
            continue

    if len(node_idx_list) == 0:
        raise ValueError('[make_batch_sampler] No centers provided. Check that your target_traj_indices is non-empty.')
    B = len(node_idx_list)
    sample_idx = torch.arange(B, dtype=torch.long)
    node_idx = torch.tensor(node_idx_list, dtype=torch.long).contiguous()
    max_time = torch.tensor(max_time_list, dtype=torch.long).contiguous()
    label = _pad_labels_to_common_width(label_list, pad_value=0).contiguous()
    sampler = NeighborSampler(lbsn_dataset.x, lbsn_dataset.edge_index, lbsn_dataset.edge_attr, intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold, inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold, edge_t=lbsn_dataset.edge_t, edge_delta_t=lbsn_dataset.edge_delta_t, edge_type=lbsn_dataset.edge_type, sizes=sizes, sample_idx=sample_idx, node_idx=node_idx, edge_delta_s=lbsn_dataset.edge_delta_s, max_time=max_time, label=label, batch_size=B, num_workers=num_workers, shuffle=False, pin_memory=False)

    return (sampler, kept_centers)


def _pick_aligned_sample_for_traj(ds, target_traj_idx: int, mode: str='train'):
    mode = (mode or 'train').lower()

    if mode not in ('train', 'test'):
        raise ValueError(f"mode must be 'train' or 'test', got {mode!r}")

    def _to1d_long(x):
        t = torch.as_tensor(x)

        if t.ndim == 0:
            t = t.view(1)

        return t.long().contiguous()

    def _try_split(split_name: str):
        if split_name == 'train':
            node_idx = getattr(ds, 'node_idx_train', None)
            sample_idx = getattr(ds, 'sample_idx_train', None)
            max_time = getattr(ds, 'max_time_train', None)
            label = getattr(ds, 'label_train', None)
        else:
            node_idx = getattr(ds, 'node_idx_test', None)
            sample_idx = getattr(ds, 'sample_idx_test', None)
            max_time = getattr(ds, 'max_time_test', None)
            label = getattr(ds, 'label_test', None)

        if node_idx is None:
            return None

        ni = _to1d_long(node_idx)
        hits = (ni == int(target_traj_idx)).nonzero(
            as_tuple=False
        ).view(-1)

        if hits.numel() == 0:
            return None

        pos = int(hits[-1].item())

        s_all = _to1d_long(sample_idx)
        t_all = _to1d_long(max_time)
        y_all = torch.as_tensor(label)

        if y_all.ndim == 0:
            y_all = y_all.view(1, 1)
        elif y_all.ndim == 1:
            y_all = y_all.view(-1, 1)
        elif y_all.ndim > 2:
            y_all = y_all.view(y_all.size(0), -1)

        lens = {
            'node': len(ni),
            'sample': len(s_all),
            'time': len(t_all),
            'label': len(y_all),
        }
        N = min(lens.values())

        if N <= 0:
            return None

        if pos >= N:
            pos = 0

        sidx = s_all[pos:pos + 1].contiguous()
        nidx = ni[pos:pos + 1].contiguous()
        mxt = t_all[pos:pos + 1].contiguous()
        lbl = y_all[pos:pos + 1]

        if lbl.ndim == 1:
            lbl = lbl.view(-1, 1)

        return sidx, nidx, mxt, lbl, split_name

    preferred = [mode] + [
        m for m in ('train', 'test')
        if m != mode
    ]

    for split_name in preferred:
        out = _try_split(split_name)

        if out is not None:
            return out

    raise ValueError(
        f'[_pick_aligned_sample_for_traj] target_traj_idx '
        f'{target_traj_idx} not found in any of train/test splits.'
    )

def _pad_labels_to_common_width(label_list, pad_value=0):
    widths = [lbl.view(1, -1).size(1) for lbl in label_list]
    Cmax = max(widths) if widths else 1
    out = []

    for lbl in label_list:
        t = lbl.view(1, -1)
        if t.size(1) < Cmax:
            pad = torch.full((1, Cmax - t.size(1)), pad_value, dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        out.append(t)

    return torch.cat(out, dim=0)


# Generate attention-based POI candidates for target trajectories.
def attention_candidates_from_target_trajs_batch(model, cfg, lbsn_dataset, target_traj_indices: list, topk_2hop: int=10, num_workers: int=0, mode: str='train', min_tj_edges: int=1):
    DEVICE = next(model.parameters()).device
    all_centers = [int(c) for c in target_traj_indices]
    results = {c: [] for c in all_centers}

    try:
        sampler, kept_centers = make_batch_sampler(cfg, lbsn_dataset, all_centers, DEVICE, num_workers=num_workers, mode=mode)
    except Exception:
        return results

    per_center_results = {int(t): [] for t in kept_centers}

    try:
        for data in sampler:
            split_index = _safe_get_split_index(data.adjs_t)
            if split_index is None and hasattr(data, 'split_index'):
                split_index = int(data.split_index)
            if split_index is None:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            centers = list(map(int, kept_centers))
            try:
                centers_local = {}
                if hasattr(data, 'n_id') and data.n_id is not None:
                    n_id = data.n_id.detach().cpu().tolist()
                    global2local = {int(g): int(i) for i, g in enumerate(n_id)}
                elif hasattr(data, 'seed_n_id') and data.seed_n_id is not None:
                    seeds = data.seed_n_id.detach().cpu().tolist()
                    global2local = {int(g): int(i) for i, g in enumerate(seeds)}
                else:
                    global2local = {}
                for c in centers:
                    centers_local[c] = global2local.get(int(c))
            except Exception:
                centers_local = {}
            data = data.to(DEVICE)
            input_data = {'x': data.x, 'edge_index': data.adjs_t, 'edge_attr': data.edge_attrs, 'split_index': split_index, 'delta_ts': data.edge_delta_ts, 'delta_ss': data.edge_delta_ss, 'edge_type': data.edge_types}
            if hasattr(torch, 'inference_mode'):
                ctx = torch.inference_mode()
            else:
                ctx = torch.no_grad()
            with ctx:
                _logits, _loss = model(input_data, label=data.y[:, 0])
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            if not (getattr(model, 'do_traj2traj', False) and len(model.conv_list) > 1):
                for conv in getattr(model, 'conv_list', []):
                    if hasattr(conv, '_last_alpha'):
                        conv._last_alpha = None
                    if hasattr(conv, '_last_edge_index'):
                        conv._last_edge_index = None
                del input_data, data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                results.update(per_center_results)
                return results
            last_conv = model.conv_list[-1]
            alpha_tj = getattr(last_conv, '_last_alpha', None)
            eidx_tj = getattr(last_conv, '_last_edge_index', None)
            eidx_ci = None
            best_cross = -1
            for ei in input_data['edge_index']:
                if ei is None:
                    continue
                try:
                    if hasattr(ei, 'storage'):
                        row, col = (ei.storage.row(), ei.storage.col())
                    elif isinstance(ei, torch.Tensor) and ei.ndim == 2 and (ei.size(0) == 2):
                        row, col = (ei[0], ei[1])
                    else:
                        continue
                    crosses = (row <= split_index) & (col > split_index) | (col <= split_index) & (row > split_index)
                    num_cross = int(crosses.sum().item())
                    if num_cross > best_cross:
                        best_cross = num_cross
                        eidx_ci = ei
                except Exception:
                    pass
            poi_col_view = None
            try:
                poi_col_view = data.x[:, 1].long()
            except Exception:
                poi_col_view = None
            src_tj, dst_tj = (None, None)
            if alpha_tj is not None and eidx_tj is not None:
                try:
                    src_tj, dst_tj = _get_src_dst(eidx_tj)
                except Exception:
                    pass
            for center in centers:
                cl = centers_local.get(int(center), None)
                if cl is None:
                    per_center_results[int(center)] = []
                    continue
                if src_tj is not None and dst_tj is not None:
                    incoming_mask = (
                        (dst_tj == cl)
                        & (src_tj <= split_index)
                        & (src_tj != cl)
                    )
                    num_tj_touch = int(incoming_mask.sum().item())
                    if num_tj_touch < int(min_tj_edges):
                        per_center_results[int(center)] = []
                        continue
                if alpha_tj is None or eidx_tj is None or eidx_ci is None or (poi_col_view is None):
                    per_center_results[int(center)] = []
                    continue
                cand = _alpha_edges_to_poi_scores_tj2tj_neighbors(alpha_tj, eidx_tj, eidx_ci, split_index, poi_col_view, int(cl))
                cand = cand[:topk_2hop]
                per_center_results[int(center)] = cand
            for c in centers[:8]:
                cand = per_center_results.get(int(c), [])
            last_conv._last_alpha = None
            last_conv._last_edge_index = None
            for conv in getattr(model, 'conv_list', []):
                if hasattr(conv, '_last_alpha'):
                    conv._last_alpha = None
                if hasattr(conv, '_last_edge_index'):
                    conv._last_edge_index = None
            del input_data, data, alpha_tj, eidx_tj, eidx_ci, poi_col_view, centers_local
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception:
        for c in kept_centers:
            per_center_results[int(c)] = []

    results.update(per_center_results)

    return results


# Build geohash mappings for traffic stations.
def build_traffic_geohash_file(meta_dir: str, traffic_npz_path: str, out_path: str, precision: int=7):
    traffic_loc = _load_station_coordinates_from_metadata(meta_dir, traffic_npz_path)
    mapping = {}

    for sid, (lat, lon) in traffic_loc.items():
        try:
            gh = encode_geohash(lat, lon, precision=precision)
            mapping[str(sid)] = gh
        except Exception as e:
            print(f'[WARN] Failed to encode geohash for station {sid} ({lat}, {lon}): {e}')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f'[GeoHash][SAVE] Traffic geohash (precision={precision}) -> {out_path} (size={len(mapping)})')


# Load traffic station coordinates from CSV or text metadata.
def _load_station_coordinates_from_metadata(meta_path: str, traffic_npz_path: str=None) -> dict:
    station_set = None
    if traffic_npz_path and osp.exists(traffic_npz_path):
        try:
            import numpy as np
            npz = np.load(traffic_npz_path, allow_pickle=True)
            if 'stations' in npz:
                stations = [str(s) for s in npz['stations']]
                station_set = set(stations)
            else:
                print(f"[TrafficMap] No 'stations' array in {traffic_npz_path}; continuing without filter.")
        except Exception as e:
            print(f'[TrafficMap] Failed to load {traffic_npz_path}: {e}; continuing without filter.')
    if osp.isfile(meta_path) and meta_path.lower().endswith('.csv'):
        meta = pd.read_csv(meta_path)
        id_candidates = ['지점번호', 'station_id', 'StationID', 'ID', 'id']
        lat_candidates = ['위도', 'Latitude', 'lat', 'LAT']
        lon_candidates = ['경도', 'Longitude', 'lon', 'LON', 'lng', 'LNG']

        def _pick_col(df, cands):
            for c in cands:
                if c in df.columns:
                    return c
            return None
        c_id = _pick_col(meta, id_candidates)
        c_lat = _pick_col(meta, lat_candidates)
        c_lon = _pick_col(meta, lon_candidates)
        if not (c_id and c_lat and c_lon):
            raise ValueError(f'CSV metadata must contain id/lat/lon columns. Found columns={list(meta.columns)} | Need one of id={id_candidates}, lat={lat_candidates}, lon={lon_candidates}')
        meta[c_id] = meta[c_id].astype(str).str.strip()
        meta[c_lat] = pd.to_numeric(meta[c_lat], errors='coerce')
        meta[c_lon] = pd.to_numeric(meta[c_lon], errors='coerce')
        meta = meta.dropna(subset=[c_lat, c_lon])
        if station_set is not None:
            meta = meta[meta[c_id].isin(station_set)]
        coords = {str(r[c_id]).strip(): (float(r[c_lat]), float(r[c_lon])) for _, r in meta.iterrows()}
        if not coords:
            raise RuntimeError('No station coordinates found in the CSV metadata.')
        return coords
    if not osp.isdir(meta_path):
        raise FileNotFoundError(f'Metadata path not found: {meta_path}')
    pattern = osp.join(meta_path, '*.txt')
    paths = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(f'Metadata text files not found: {meta_path}')
    meta_list = []

    for p in paths:
        df = pd.read_csv(p, sep=None, engine='python')
        meta_list.append(df)

    meta = pd.concat(meta_list, ignore_index=True)
    required_cols = ['ID', 'Type', 'Latitude', 'Longitude']

    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Required metadata column missing: '{col}'")

    meta['ID'] = meta['ID'].astype(str).str.strip()
    type_series = meta['Type'].astype(str).str.upper().str.strip()
    lat_numeric = pd.to_numeric(meta['Latitude'], errors='coerce')
    lng_numeric = pd.to_numeric(meta['Longitude'], errors='coerce')
    has_lat = lat_numeric.notna()
    has_lng = lng_numeric.notna()
    is_mainline = type_series == 'ML'
    valid_mask = is_mainline & has_lat & has_lng

    if station_set is not None:
        valid_mask = valid_mask & meta['ID'].isin(station_set)
    meta_valid = meta.loc[valid_mask]
    coords = {}

    for _, row in meta_valid.iterrows():
        sid = str(row['ID']).strip()
        coords[sid] = (float(row['Latitude']), float(row['Longitude']))

    if not coords:
        raise RuntimeError('No valid station coordinates found.')

    return coords
import ast


def normalize_category(cat):
    if isinstance(cat, str):
        s = cat.strip()
        if s.startswith('[') and s.endswith(']') or (s.startswith('{') and s.endswith('}')):
            try:
                cat = json.loads(s)
            except Exception:
                try:
                    cat = ast.literal_eval(s)
                except Exception:
                    return s
        else:
            return s
    if isinstance(cat, list):
        names = []
        for item in cat:
            if isinstance(item, dict) and 'name' in item:
                names.append(str(item['name']))
            else:
                names.append(str(item))
        return ', '.join(names)
    if isinstance(cat, dict):
        return str(cat.get('name', cat))
    return '' if cat is None or (isinstance(cat, float) and pd.isna(cat)) else str(cat)


def find_unmapped_pois_in_df(df, poi2traffic, id_map, poi_id_le=None, padding_poi_id=None):
    unmapped = set()

    for pid in pd.unique(df['PoiId']):
        k1 = canonical_poi_key(pid)
        info1 = poi2traffic.get(k1)
        info2 = None
        if info1 is None and poi_id_le is not None:
            decoded = decode_poi_from_idx(pid, poi_id_le, padding_poi_id)
            if decoded is not None:
                k2 = canonical_poi_key(decoded)
                info2 = poi2traffic.get(k2)
        info = info1 if info1 is not None else info2
        if info is None:
            unmapped.add(k1)
            continue
        if isinstance(info, dict):
            raw_tids = [info.get('traffic_id')]
        elif isinstance(info, list):
            raw_tids = [x.get('traffic_id') for x in info if isinstance(x, dict)]
        else:
            unmapped.add(k1)
            continue
        ok = False
        for raw_tid in raw_tids:
            if raw_tid is None:
                continue
            if str(raw_tid) in id_map:
                ok = True
                break
        if not ok:
            unmapped.add(k1)

    return unmapped


def drop_trajectories_with_poi(df: pd.DataFrame, bad_poi):
    if 'pseudo_session_trajectory_id' not in df.columns:
        raise KeyError('pseudo_session_trajectory_id column is required in the dataframe.')
    bad_key = canonical_poi_key(bad_poi)
    traj_col = 'pseudo_session_trajectory_id'
    poi_norm = df['PoiId'].map(canonical_poi_key)
    bad_trajs = set(df.loc[poi_norm == bad_key, traj_col])
    n_before = len(df)
    df_new = df[~df[traj_col].isin(bad_trajs)].copy()
    n_after = len(df_new)

    return (df_new, len(bad_trajs), n_before - n_after)


def _center_traj_idx_from_current_trajectory(user_trajectory_data, traj_id_mapper):
    uniq = user_trajectory_data['pseudo_session_trajectory_id'].unique()
    if len(uniq) == 0:
        raise ValueError('user_trajectory_data has no pseudo_session_trajectory_id')
    raw_traj_id = int(uniq[0])
    if callable(traj_id_mapper):
        return int(traj_id_mapper(raw_traj_id))
    return int(raw_traj_id)


# Map each POI to its nearest traffic stations.
def _save_poi_to_nearest_station_mapping(current_df: pd.DataFrame, station_metadata_csv: str, traffic_train_csv: str, save_dir: str, filename: str='poi_to_nearest_traffic_node.pkl', k: int=3) -> str:
    try:
        import numpy as np
        from sklearn.neighbors import BallTree
    except ImportError as e:
        raise ImportError('Install numpy and scikit-learn for traffic mapping.') from e

    required = {'PoiId', 'Latitude', 'Longitude'}
    missing = required - set(current_df.columns)

    if missing:
        raise KeyError(f'Required columns missing: {missing}')
    traffic_loc = _load_station_coordinates_from_metadata(station_metadata_csv, traffic_train_csv)
    if not traffic_loc:
        raise RuntimeError('Traffic station coordinates are empty.')
    poi_df = current_df[['PoiId', 'Latitude', 'Longitude']].drop_duplicates().dropna()
    if poi_df.empty:
        raise RuntimeError('POI coordinates are empty.')
    station_ids = list(traffic_loc.keys())
    stations_latlon = np.array([traffic_loc[sid] for sid in station_ids], dtype=float)
    stations_rad = np.radians(stations_latlon[:, [0, 1]])
    poi_ids_raw = poi_df['PoiId'].tolist()
    pois_latlon = poi_df[['Latitude', 'Longitude']].to_numpy(dtype=float)
    pois_rad = np.radians(pois_latlon[:, [0, 1]])
    k_eff = max(1, min(int(k), len(station_ids)))
    tree = BallTree(stations_rad, metric='haversine')
    d_rad, idx = tree.query(pois_rad, k=k_eff)
    d_m = (d_rad * EARTH_RADIUS_M).astype(float)

    def _to_str(x):
        try:
            return str(int(float(x)))
        except Exception:
            return str(x)

    mapping = {}

    for i, pid in enumerate(poi_ids_raw):
        pid_str = _to_str(pid)
        neighbors = []
        for j in range(k_eff):
            station_id = str(station_ids[idx[i, j]])
            dist_m_ij = float(d_m[i, j])
            if dist_m_ij <= MAX_TRAFFIC_DISTANCE_M:
                neighbors.append({'traffic_id': station_id, 'distance': dist_m_ij})
        if not neighbors:
            nearest_station_id = str(station_ids[idx[i, 0]])
            nearest_dist = float(d_m[i, 0])
            neighbors.append({'traffic_id': nearest_station_id, 'distance': nearest_dist})
        mapping[pid_str] = neighbors

    os.makedirs(save_dir, exist_ok=True)
    out_path = osp.join(save_dir, filename)

    with open(out_path, 'wb') as f:
        pickle.dump(mapping, f)

    print(f'[TrafficMap][SAVE] nearest mapping -> {out_path} ({len(mapping):,}) (k={k_eff}, max_dist={MAX_TRAFFIC_DISTANCE_M}m)')

    return out_path


def infer_poi_max_value(main_data, historical_data=None, poi_id_le=None, padding_poi_id=None):
    if poi_id_le is not None and hasattr(poi_id_le, 'classes_'):
        n = len(getattr(poi_id_le, 'classes_', []))
        if n > 0:
            if padding_poi_id is not None:
                try:
                    if int(padding_poi_id) == 0:
                        return int(n)
                except Exception:
                    pass
            return int(n - 1 if n > 0 else 0)
    frames = [main_data]
    if historical_data is not None:
        frames.append(historical_data)
    combined = pd.concat(frames, ignore_index=True)
    s = pd.to_numeric(combined['PoiId'], errors='coerce')
    s = s.dropna()

    if not s.empty:
        try:
            return int(s.max())
        except Exception:
            pass
    unique_cnt = combined['PoiId'].nunique(dropna=True)

    return max(int(unique_cnt - 1), 0)


# Generate QA pairs with POI, traffic, GPS, and model candidate tokens.
def generate_qa_pairs(main_data, kqt=None, historical_data=None, args=None, poi2traffic=None, traffic_id_map=None, poi_id_le=None, padding_poi_id=None, poi_geohash=None, traffic_geohash=None, geohash2id=None, traffic_id_reverse=None, model=None, cfg=None, lbsn_dataset=None, traj_id_mapper=None, mode=None, topk_candidates=10, topk_geo_candidates=20):

    def _gps_token_for_poi(poi_id):
        if poi_geohash is None or geohash2id is None:
            return ''
        key = canonical_poi_key(poi_id)
        gh = poi_geohash.get(key)

        if gh is None:
            return ''
        gh_id = geohash2id.get(gh)
        if gh_id is None:
            return ''

        return f'<GPS {int(gh_id)}>'

    def _gps_token_for_traffic_sid(contiguous_id):
        if traffic_geohash is None or geohash2id is None:
            return ''
        if traffic_id_reverse is None:
            return ''

        try:
            cid = int(contiguous_id)
        except Exception:
            return ''

        if cid < 0 or cid >= len(traffic_id_reverse):
            return ''
        orig_sid = traffic_id_reverse[cid]
        gh = traffic_geohash.get(str(orig_sid))

        if gh is None:
            return ''
        gh_id = geohash2id.get(gh)
        if gh_id is None:
            return ''

        return f'<GPS {int(gh_id)}>'

    main_data = main_data.sort_values(by=['UserId', 'pseudo_session_trajectory_id', 'UTCTimeOffsetEpoch'])
    qa_pairs = []
    traffic_pairs_list = []
    poi_count_bins = defaultdict(int)
    value_max = infer_poi_max_value(main_data, historical_data, poi_id_le, padding_poi_id)
    geo_index = None

    if topk_geo_candidates and topk_geo_candidates > 0:
        try:
            import numpy as np
            from sklearn.neighbors import BallTree
            frames = []
            if set(['PoiId', 'Latitude', 'Longitude']).issubset(main_data.columns):
                frames.append(main_data[['PoiId', 'Latitude', 'Longitude']])
            if historical_data is not None and set(['PoiId', 'Latitude', 'Longitude']).issubset(historical_data.columns):
                frames.append(historical_data[['PoiId', 'Latitude', 'Longitude']])
            if frames:
                poi_coord_df = pd.concat(frames, ignore_index=True).dropna()
                poi_coord_df = poi_coord_df.drop_duplicates(subset=['PoiId', 'Latitude', 'Longitude'])
                poi_coord_df['Latitude'] = pd.to_numeric(poi_coord_df['Latitude'], errors='coerce')
                poi_coord_df['Longitude'] = pd.to_numeric(poi_coord_df['Longitude'], errors='coerce')
                poi_coord_df = poi_coord_df.dropna(subset=['Latitude', 'Longitude'])
                if len(poi_coord_df) > 0:
                    poi_ids = poi_coord_df['PoiId'].tolist()
                    coords = poi_coord_df[['Latitude', 'Longitude']].to_numpy(dtype=float)
                    coords_rad = np.radians(coords[:, [0, 1]])
                    tree = BallTree(coords_rad, metric='haversine')
                    poiid2index = {canonical_poi_key(pid): i for i, pid in enumerate(poi_ids)}
                    geo_index = (tree, poi_ids, coords_rad, poiid2index, topk_geo_candidates)
        except Exception as e:
            print(f'[WARN] failed to build POI geo-neighbor index: {e}')
            geo_index = None

    def _fmt_time_no_tz(x):
        try:
            ts = pd.to_datetime(x, errors='coerce')
            if pd.isna(ts):
                s = str(x)
                return s.split('+')[0].strip()
            return ts.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            s = str(x)
            return s.split('+')[0].strip()

    def get_geo_candidates_for_prev_poi(prev_poi_id):
        if geo_index is None:
            return []
        from math import isfinite
        tree, poi_ids, coords_rad, poiid2index, k_geo = geo_index
        key = canonical_poi_key(prev_poi_id)
        idx = poiid2index.get(key)

        if idx is None:
            return []
        k_query = min(k_geo + 1, len(poi_ids))
        d_rad, ind = tree.query(coords_rad[idx:idx + 1], k=k_query)
        d_rad = d_rad[0]
        ind = ind[0]
        candidates = []

        for dist_rad, j in zip(d_rad, ind):
            if j == idx:
                continue
            if not isfinite(float(dist_rad)):
                continue
            pid_cand = poi_ids[j]
            dist_m = float(dist_rad * EARTH_RADIUS_M)
            score = -dist_m
            candidates.append((pid_cand, score))
            if len(candidates) >= k_geo:
                break

        return candidates

    attn_cache = {}
    key2center = {}
    centers_to_query = []
    seen_centers = set()
    local_mode = (mode or getattr(args, 'mode', 'train')).lower() if args is not None else (mode or 'train').lower()
    attn_bs = int(getattr(args, 'attn_batch_size', 32)) if args is not None else 32
    topk_attn = getattr(args, 'topk_attn_2hop', topk_candidates) if args is not None else topk_candidates
    num_workers = getattr(args, 'num_workers', 0) if args is not None else 0
    min_tj_edges = getattr(args, 'min_tj_edges', 1) if args is not None else 1

    for user in main_data['UserId'].unique():
        user_data = main_data[main_data['UserId'] == user]
        for traj_id in user_data['pseudo_session_trajectory_id'].unique():
            user_trajectory_data = user_data[user_data['pseudo_session_trajectory_id'] == traj_id]
            center_idx = _center_traj_idx_from_current_trajectory(user_trajectory_data=user_trajectory_data, traj_id_mapper=traj_id_mapper)
            center_idx = int(center_idx)
            key = (int(user), int(traj_id))
            key2center[key] = center_idx
            if model is not None and cfg is not None and (lbsn_dataset is not None):
                if center_idx not in seen_centers:
                    seen_centers.add(center_idx)
                    centers_to_query.append(center_idx)

    if model is not None and cfg is not None and (lbsn_dataset is not None) and centers_to_query:
        for i in range(0, len(centers_to_query), attn_bs):
            batch_centers = centers_to_query[i:i + attn_bs]
            try:
                per_center = attention_candidates_from_target_trajs_batch(model, cfg, lbsn_dataset, batch_centers, topk_2hop=topk_attn, num_workers=num_workers, mode=local_mode, min_tj_edges=min_tj_edges)
                attn_cache.update(per_center)
            except Exception as e:
                print(f'[WARN] batch attention failed for centers[{i}:{i + len(batch_centers)}]: {e}')

    for user in tqdm(main_data['UserId'].unique()):
        user_data = main_data[main_data['UserId'] == user]
        for traj_id in user_data['pseudo_session_trajectory_id'].unique():
            user_trajectory_data = user_data[user_data['pseudo_session_trajectory_id'] == traj_id]
            start_time = user_trajectory_data['UTCTimeOffsetEpoch'].min()
            num_traj = user_trajectory_data.shape[0]
            early_historical_data = pd.DataFrame()
            if num_traj <= 600 and historical_data is not None:
                early_historical_data = historical_data[(historical_data['UserId'] == user) & (historical_data['UTCTimeOffsetEpoch'] < start_time)].tail(600)
            user_trajectory_data.reset_index(drop=True, inplace=True)
            if num_traj > 500:
                user_trajectory_data = user_trajectory_data.tail(500)
                num_traj = 500
            current_body = user_trajectory_data.iloc[:-1]
            num_current_traj = len(current_body)
            num_early_traj = len(early_historical_data)
            total_traj_count = num_current_traj + num_early_traj

            if total_traj_count > 100:
                remaining = 100

                K = min(num_current_traj, remaining)
                final_current_data = current_body.tail(K).to_dict("records")
                remaining -= K

                J = min(num_early_traj, remaining)
                final_early_data = early_historical_data.tail(J).to_dict("records")
            else:
                final_current_data = current_body.to_dict("records")
                final_early_data = early_historical_data.to_dict("records")

            total_traj_count = len(final_current_data) + len(final_early_data)
            question_parts = ['<question>:  Traffic representation: the token `<TRAFFIC k>` encodes nearby traffic aggregated over the previous hour.']
            traffic_pairs_for_this_qa = []

            def _append_line(row, uid_for_line):
                tid_mapped_list = get_mapped_traffic_id_for_poi(row['PoiId'], poi2traffic, traffic_id_map, poi_id_le, padding_poi_id)
                poi_tok = f"<POI {row['PoiId']}>"
                gps_poi_tok = _gps_token_for_poi(row['PoiId'])
                poi_part = f"visited POI id {row['PoiId']} represented as {poi_tok}"
                if gps_poi_tok:
                    poi_part += f', located at {gps_poi_tok}'
                traf_pieces = []

                if tid_mapped_list:
                    for t_id in tid_mapped_list:
                        traf_tok = f'<TRAFFIC {int(t_id)}>'
                        gps_traf_tok = _gps_token_for_traffic_sid(t_id)
                        if gps_traf_tok:
                            traf_pieces.append(f'{traf_tok}, located at {gps_traf_tok}')
                        else:
                            traf_pieces.append(f'{traf_tok}')
                traffic_part = ''

                if traf_pieces:
                    traffic_part = ', and nearby traffic nodes are ' + ' ; '.join(traf_pieces)
                cat_norm = normalize_category(row.get('PoiCategoryName', ''))
                category_part = f", and the CATEGORY is {cat_norm} with category id {row['PoiCategoryId']}"
                ts_str = _fmt_time_no_tz(row['UTCTimeOffset'])
                question_parts.append(f'At {ts_str}, user {uid_for_line} {poi_part}{traffic_part}{category_part}.')
                if tid_mapped_list:
                    ts_no_tz = _fmt_time_no_tz(row['UTCTimeOffset'])
                    for t_id in tid_mapped_list:
                        try:
                            traffic_pairs_for_this_qa.append({'ts': ts_no_tz, 'traffic_k': int(t_id)})
                        except Exception:
                            pass

            if final_early_data:
                question_parts.append(f'There is earlier history trajectory data for user {user}:')
                for row in final_early_data:
                    _append_line(row, user)

            question_parts.append(f'The following is the current trajectory of user {user}:')
            for row in final_current_data:
                _append_line(row, user)

            hint_line = ''
            key = (int(user), int(traj_id))
            center_idx = key2center.get(key, None)
            attn_2hop = []
            if center_idx is not None and model is not None and (cfg is not None) and (lbsn_dataset is not None):
                attn_2hop = attn_cache.get(int(center_idx), [])
            geo_cand = []
            traj_len = len(user_trajectory_data)
            if traj_len > 1 and geo_index is not None:
                prev_poi_id = user_trajectory_data.iloc[-2]['PoiId']
                geo_cand = get_geo_candidates_for_prev_poi(prev_poi_id)
            if args is not None and getattr(args, 'dataset_name', '').lower() in ('ca', 'tky'):

                def _filter_zero(seq):
                    out = []
                    for pid, sc in seq:
                        try:
                            if int(pid) == 0:
                                continue
                        except Exception:
                            pass
                        out.append((pid, sc))
                    return out
                attn_2hop = _filter_zero(attn_2hop)
                geo_cand = _filter_zero(geo_cand)
            attn_hint = ''
            geo_hint = ''
            if attn_2hop:
                attn_hint = format_candidates_as_hint(attn_2hop, gps_token_for_poi=_gps_token_for_poi, max_items=topk_attn, prefix='POI embedding-based candidates: ')
            if geo_cand and (topk_geo_candidates is None or topk_geo_candidates > 0):
                geo_hint = format_candidates_as_hint(geo_cand, gps_token_for_poi=_gps_token_for_poi, max_items=topk_geo_candidates, prefix='Distance-based candidates: ')
            if attn_hint or geo_hint:
                lines = ['Here are POI candidates that may help with the answer (sorted by score, descending).']
                if attn_hint:
                    lines.append(attn_hint)
                if geo_hint:
                    lines.append(geo_hint)
                hint_line = ' '.join(lines)

            if hint_line:
                question_parts.append(hint_line)

            next_poi_id = user_trajectory_data.iloc[-1]['PoiId']
            _ = get_mapped_traffic_id_for_poi(next_poi_id, poi2traffic, traffic_id_map, poi_id_le, padding_poi_id)
            question = ' '.join(question_parts)
            ts_next = _fmt_time_no_tz(user_trajectory_data.iloc[-1]['UTCTimeOffset'])
            question += f' Based on this data, predict the location user {user} will visit next. At {ts_next}, user {user} will visit a POI. Which POI id will user {user} visit? Note that POI id is an integer in the range from 0 to {value_max}.'
            answer = f"<answer>: At {user_trajectory_data.iloc[-1]['UTCTimeOffset']}, user {user} will visit POI id {next_poi_id}."
            bin_index = total_traj_count // 100 * 100
            
            if bin_index <= 800:
                poi_count_bins[f'{bin_index}-{bin_index + 100}'] += 1
            else:
                poi_count_bins['800+'] += 1
            qa_pairs.append((question, answer))
            traffic_pairs_list.append(list(traffic_pairs_for_this_qa))

    return (qa_pairs, poi_count_bins, traffic_pairs_list)


def assert_no_unmapped_pois(dfs, poi2traffic, id_map, poi_id_le=None, padding_poi_id=None, sample_show=20, allow_unmapped=True):
    unmapped = set()

    for df in dfs:
        for pid in pd.unique(df['PoiId']):
            k1 = canonical_poi_key(pid)
            info1 = poi2traffic.get(k1)
            info2 = None
            if info1 is None and poi_id_le is not None:
                orig_pid = decode_poi_from_idx(pid, poi_id_le, padding_poi_id)
                if orig_pid is not None:
                    k2 = canonical_poi_key(orig_pid)
                    info2 = poi2traffic.get(k2)
            info = info1 if info1 is not None else info2
            if info is None:
                unmapped.add(k1)
                continue
            if isinstance(info, dict):
                raw_tids = [info.get('traffic_id')]
            elif isinstance(info, list):
                raw_tids = [x.get('traffic_id') for x in info if isinstance(x, dict)]
            else:
                unmapped.add(k1)
                continue
            ok = False
            for raw_tid in raw_tids:
                if raw_tid is None:
                    continue
                if str(raw_tid) in id_map:
                    ok = True
                    break
            if not ok:
                unmapped.add(k1)

    if unmapped:
        examples = ', '.join(sorted(list(unmapped))[:sample_show])
        level = '[WARN]' if allow_unmapped else '[ERROR]'
        print(f'{level} Unmapped POI detected: {len(unmapped)} POI(s) have no traffic mapping.')
        print(f'       Examples: {examples}')
        if not allow_unmapped:
            print('       Fix your poi→traffic mapping (pkl) or lower the distance threshold, then rerun.')
            sys.exit(1)


# Parse inputs and run the end-to-end preprocessing pipeline.
def main():
    parser = argparse.ArgumentParser(description='Build contiguous traffic IDs and generate QA pairs in one go.')
    parser.add_argument('-dataset_name', type=str, choices=['ca', 'hd'], required=True)
    parser.add_argument('-poi2traffic_pkl', type=str, default=None, help='Path to poi→traffic mapping pickle (poi_to_traffic_node_mapping_*.pkl).')
    parser.add_argument('-label_encoding_pkl', type=str, default=None, help='Path to label_encoding.pkl with poi label encoder.')
    parser.add_argument('-out_map_json', type=str, default=None, help='Where to save contiguous traffic id map JSON (default: <dataset>/preprocessed/traffic_id_map.json)')
    parser.add_argument('-out_train_json', type=str, default=None, help='Where to save train QA JSON')
    parser.add_argument('-out_test_txt', type=str, default=None, help='Where to save test QA TXT')
    parser.add_argument('-allow_unmapped', type=str, choices=['True', 'False'], default='False', help='If False, exit on unmapped POIs. If True, just warn and continue.')
    parser.add_argument('-traffic_meta_dir', type=str, required=True, help='traffic metadata txt(ID,Type,Latitude,Longitude) folder path')
    parser.add_argument('-traffic_npz_path', type=str, required=True)
    parser.add_argument('-k_nearest_traffic', type=int, default=5)
    parser.add_argument('--use_attn_candidates', type=str, choices=['True', 'False'], default='True')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to the pretrained STHGCN checkpoint.'
    )
    parser.add_argument('--topk_attn_2hop', type=int, default=20)
    parser.add_argument('--attn_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--min_tj_edges', type=int, default=1)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--topk_geo_candidates', type=int, default=20)
    args = parser.parse_args()

    default_base = osp.join(
        TA_LLM_ROOT,
        'datasets',
        args.dataset_name,
        'preprocessed'
    )

    if args.label_encoding_pkl is None:
        args.label_encoding_pkl = osp.join(default_base, 'label_encoding.pkl')
    if args.poi2traffic_pkl is None:
        args.poi2traffic_pkl = osp.join(default_base, 'maps', 'poi_to_nearest_traffic_node_1tok_sep_candidate_attn30_nearby30.pkl')
    allow_unmapped = args.allow_unmapped == 'True'
    base_path = default_base + '/'

    MODEL_DEVICE = torch.device('cpu')

    cfg, lbsn_dataset = build_training_like_env(args, MODEL_DEVICE)
    _, _, file_based_mapper = load_traj_mappers(args.dataset_name, repo_root=REPO_ROOT)
    sample_csv = os.path.join(base_path, 'sample.csv')

    if not os.path.exists(sample_csv):
        raise FileNotFoundError(f'sample.csv not found at: {sample_csv}')
    sample_df = pd.read_csv(sample_csv)
    if 'pseudo_session_trajectory_id' not in sample_df.columns:
        raise KeyError("sample.csv must contain 'pseudo_session_trajectory_id' column to map QA trajectories to graph traj nodes.")
    valid_traj_ids = set(sample_df['pseudo_session_trajectory_id'].dropna().astype(int).tolist())

    def traj_id_mapper(tid):
        try:
            raw_tid = int(tid)
        except Exception:
            msg = f'traj_id_mapper: invalid traj id {tid!r}'
            print(f'[ERROR] {msg}')
            sys.exit(1)

        if raw_tid not in valid_traj_ids:
            msg = f'traj_id_mapper: QA traj_id {raw_tid} not found in sample.csv; cannot map to a trajectory node in the graph.'
            print(f'[ERROR] {msg}')
            sys.exit(1)

        try:
            graph_idx = file_based_mapper(raw_tid)
        except Exception as e:
            msg = f'traj_id_mapper: failed to map raw traj id {raw_tid} to graph_traj_index: {e}'
            print(f'[ERROR] {msg}')
            sys.exit(1)

        return int(graph_idx)

    model = None
    if args.use_attn_candidates == 'True':
        dataset_meta = dict(spatial_slots=lbsn_dataset.spatial_slots, num_user=lbsn_dataset.num_user, num_poi=lbsn_dataset.num_poi, num_category=lbsn_dataset.num_category, padding_poi_id=lbsn_dataset.padding_poi_id, padding_user_id=lbsn_dataset.padding_user_id, padding_poi_category=lbsn_dataset.padding_poi_category, padding_hour_id=lbsn_dataset.padding_hour_id, padding_weekday_id=lbsn_dataset.padding_weekday_id)
        model = _load_sthgcn_and_patch(args.checkpoint_path, yaml_path=os.path.join(REPO_ROOT, 'conf', 'best_conf', f'{args.dataset_name}.yml'), dataset_overrides=dataset_meta)
        if model is not None:
            model.to(MODEL_DEVICE)
    out_map_json = args.out_map_json or os.path.join(base_path, 'traffic_id_map_bist_sep_candidate_attn30_nearby30.json')
    out_train_json = args.out_train_json or os.path.join(base_path, f'train_qa_pairs_kqt_200items_{args.dataset_name}_projector_TRAFFIC_poi_without_sim_100_bist_separate_1tok_candidate_attn30_nearby30.json')
    out_test_txt = args.out_test_txt or os.path.join(base_path, f'test_qa_pairs_kqt_200items_{args.dataset_name}_projector_TRAFFIC_poi_without_sim_100_bist_separate_1tok_candidate_attn30_nearby30.txt')
    train_csv = os.path.join(base_path, 'train_sample.csv')
    test_csv = os.path.join(base_path, 'test_sample_with_traj.csv')
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    for df in (train_data, test_data):
        df['PoiId'] = _canon_series(df['PoiId'])

    poi_id_le, padding_poi_id = load_poi_label_encoder(args.label_encoding_pkl)
    geohash_precision = 7
    poi_geohash_json = os.path.join(base_path, f'poi_geohash_p{geohash_precision}_sep_1tok_candidate_attn30_nearby30.json')
    traffic_geohash_json = os.path.join(base_path, f'traffic_geohash_p{geohash_precision}_sep_1tok_candidate_attn30_nearby30.json')
    build_poi_geohash_file(train_data=train_data, test_data=test_data, out_path=poi_geohash_json, precision=geohash_precision)
    build_traffic_geohash_file(meta_dir=args.traffic_meta_dir, traffic_npz_path=args.traffic_npz_path, out_path=traffic_geohash_json, precision=geohash_precision)

    with open(poi_geohash_json, 'r') as f:
        poi_gh = json.load(f)

    with open(traffic_geohash_json, 'r') as f:
        traffic_gh = json.load(f)

    all_codes = sorted(set(poi_gh.values()) | set(traffic_gh.values()))
    geohash2id = {gh: i + 1 for i, gh in enumerate(all_codes)}
    id2geohash = {i: gh for gh, i in geohash2id.items()}
    geohash_vocab_path = os.path.join(base_path, f'geohash_vocab_p{geohash_precision}_sep_1tok_candidate_attn30_nearby30.json')

    with open(geohash_vocab_path, 'w') as f:
        json.dump({'geohash2id': geohash2id, 'id2geohash': id2geohash}, f, ensure_ascii=False, indent=2)

    print(f'[GeoHash][SAVE] shared geohash vocab -> {geohash_vocab_path} (size={len(geohash2id)})')
    if not os.path.exists(args.poi2traffic_pkl):
        if args.dataset_name not in ('ca', 'hd', 'sydney'):
            raise FileNotFoundError(f'poi2traffic_pkl not found: {args.poi2traffic_pkl} (dataset={args.dataset_name}). Provide the mapping file path.')
        station_metadata_dir = args.traffic_meta_dir
        traffic_npz_path = args.traffic_npz_path
        save_dir = os.path.dirname(args.poi2traffic_pkl)
        filename = os.path.basename(args.poi2traffic_pkl)
        cols_needed = ['PoiId', 'Latitude', 'Longitude']
        for need in cols_needed:
            if need not in train_data.columns or need not in test_data.columns:
                raise KeyError(f"'{need}' column is required in both train/test CSVs to build nearest mapping.")
        current_df = pd.concat([train_data[cols_needed], test_data[cols_needed]], ignore_index=True).drop_duplicates()
        print(f"[TrafficMap] '{args.poi2traffic_pkl}' not found → generating nearest mapping (dataset={args.dataset_name})...")
        created_path = _save_poi_to_nearest_station_mapping(current_df=current_df, station_metadata_csv=station_metadata_dir, traffic_train_csv=traffic_npz_path, save_dir=save_dir, filename=filename, k=args.k_nearest_traffic)
        args.poi2traffic_pkl = created_path

    with open(args.poi2traffic_pkl, 'rb') as f:
        poi2traffic_raw = pickle.load(f)

    poi2traffic = normalize_poi2traffic_keys(poi2traffic_raw)
    id_map, reverse = build_contiguous_traffic_id_map(poi2traffic)
    save_traffic_id_map_json(out_map_json, id_map, reverse)
    unmapped_all = set()
    unmapped_all |= find_unmapped_pois_in_df(train_data, poi2traffic, id_map, poi_id_le, padding_poi_id)
    unmapped_all |= find_unmapped_pois_in_df(test_data, poi2traffic, id_map, poi_id_le, padding_poi_id)

    if not len(unmapped_all) == 0:
        if len(unmapped_all) == 1:
            bad = next(iter(unmapped_all))
            print(f'[CLEANUP] Exactly 1 unmapped POI detected: {bad}. Dropping affected trajectories in train/test before QA.')
            train_data, tr_traj_removed, tr_rows_removed = drop_trajectories_with_poi(train_data, bad)
            test_data, te_traj_removed, te_rows_removed = drop_trajectories_with_poi(test_data, bad)
            print(f'[CLEANUP] train: removed {tr_traj_removed} traj / {tr_rows_removed} rows | test: removed {te_traj_removed} traj / {te_rows_removed} rows.')
            unmapped_left = set()
            unmapped_left |= find_unmapped_pois_in_df(train_data, poi2traffic, id_map, poi_id_le, padding_poi_id)
            unmapped_left |= find_unmapped_pois_in_df(test_data, poi2traffic, id_map, poi_id_le, padding_poi_id)
            if not len(unmapped_left) == 0:
                if allow_unmapped:
                    print(f'[WARN] Unexpected unmapped POIs remain after cleanup: {sorted(list(unmapped_left))[:10]}')
                else:
                    print(f'[ERROR] Unexpected unmapped POIs remain after cleanup: {sorted(list(unmapped_left))[:10]}')
                    sys.exit(1)
        elif allow_unmapped:
            print(f'[WARN] Found {len(unmapped_all)} unmapped POIs (>=2). Samples: {sorted(list(unmapped_all))[:10]}')
        else:
            print(f'[ERROR] Found {len(unmapped_all)} unmapped POIs (>=2). Samples: {sorted(list(unmapped_all))[:10]}')
            sys.exit(1)
    assert_no_unmapped_pois([train_data, test_data], poi2traffic, id_map, poi_id_le, padding_poi_id, sample_show=20, allow_unmapped=allow_unmapped)
    qa_pairs_train, bins_train, train_pairs = generate_qa_pairs(main_data=train_data, kqt=None, historical_data=train_data, args=args, poi2traffic=poi2traffic, traffic_id_map=id_map, poi_id_le=poi_id_le, padding_poi_id=padding_poi_id, poi_geohash=poi_gh, traffic_geohash=traffic_gh, geohash2id=geohash2id, traffic_id_reverse=reverse, model=model, cfg=cfg, lbsn_dataset=lbsn_dataset, traj_id_mapper=traj_id_mapper, mode='train', topk_candidates=10, topk_geo_candidates=args.topk_geo_candidates)
    qa_pairs_test, bins_test, test_pairs = generate_qa_pairs(main_data=test_data, kqt=None, historical_data=train_data, args=args, poi2traffic=poi2traffic, traffic_id_map=id_map, poi_id_le=poi_id_le, padding_poi_id=padding_poi_id, poi_geohash=poi_gh, traffic_geohash=traffic_gh, geohash2id=geohash2id, traffic_id_reverse=reverse, model=model, cfg=cfg, lbsn_dataset=lbsn_dataset, traj_id_mapper=traj_id_mapper, mode='test', topk_candidates=10, topk_geo_candidates=args.topk_geo_candidates)
    os.makedirs(os.path.dirname(out_train_json), exist_ok=True)

    with open(out_train_json, 'w') as jf:
        json.dump([{'question': q, 'answer': a} for q, a in qa_pairs_train], jf)

    print(f'[SAVE] train QA -> {out_train_json}')
    os.makedirs(os.path.dirname(out_test_txt), exist_ok=True)

    with open(out_test_txt, 'w') as tf:
        for q, a in qa_pairs_test:
            tf.write(q + a + '\n')

    print(f'[SAVE] test QA -> {out_test_txt}')
    train_pairs_path = out_train_json[:-5] + '_traffic_pairs.json' if out_train_json.endswith('.json') else out_train_json + '_traffic_pairs.json'
    test_pairs_path = out_test_txt[:-4] + '_traffic_pairs.json' if out_test_txt.endswith('.txt') else out_test_txt + '_traffic_pairs.json'

    with open(train_pairs_path, 'w') as f:
        json.dump(train_pairs, f, ensure_ascii=False)

    print(f'[SAVE] train traffic pairs -> {train_pairs_path} (items={len(train_pairs)})')

    with open(test_pairs_path, 'w') as f:
        json.dump(test_pairs, f, ensure_ascii=False)

    print(f'[SAVE] test traffic pairs -> {test_pairs_path} (items={len(test_pairs)})')

if __name__ == '__main__':
    main()
