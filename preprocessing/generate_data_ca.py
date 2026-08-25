import os
import glob
import argparse
import json
import numpy as np
import pandas as pd


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_filtered_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if 'Timestamp' not in df.columns:
        raise ValueError(f"Missing 'Timestamp' column: {path}")
    if df.shape[1] < 2:
        raise ValueError(f"Expected a Timestamp column and at least one station column: {path}")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    bad_mask = df['Timestamp'].isna()
    n_bad = bad_mask.sum()
    if n_bad > 0:
        print(f"[WARN] {os.path.basename(path)}: dropping {n_bad} invalid timestamp rows.")
        df = df.loc[~bad_mask].copy()

    if df.empty:
        raise ValueError(f"No valid Timestamp rows found: {path}")

    df = df.sort_values('Timestamp')
    df = df.set_index('Timestamp')

    start_ts = df.index.min()
    end_ts = df.index.max()
    full_index = pd.date_range(start=start_ts, end=end_ts, freq='5T')

    df = df.reindex(full_index)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0)

    print(f"[load_filtered_csv] {os.path.basename(path)}: "
          f"{start_ts} to {end_ts}, {df.shape[0]} steps, {df.shape[1]} stations")

    return df


def load_mainline_ids_from_dir(meta_dir: str):
    if not os.path.isdir(meta_dir):
        raise NotADirectoryError(f"Metadata directory not found: {meta_dir}")

    pattern = os.path.join(meta_dir, "*.txt")
    paths = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No metadata .txt files found in: {meta_dir}")

    meta_list = []
    print(f"[INFO] Loading metadata files:")
    for p in paths:
        print(" -", os.path.basename(p))
        df = pd.read_csv(p, sep=None, engine="python")
        meta_list.append(df)

    meta = pd.concat(meta_list, ignore_index=True)

    required_cols = ["ID", "Type", "Latitude", "Longitude"]
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Missing required metadata column: {col}")

    type_series = meta["Type"].astype(str).str.upper().str.strip()
    is_mainline = type_series == "ML"

    lat_numeric = pd.to_numeric(meta["Latitude"], errors="coerce")
    lng_numeric = pd.to_numeric(meta["Longitude"], errors="coerce")
    has_lat = lat_numeric.notna()
    has_lng = lng_numeric.notna()

    # Use mainline stations only.
    valid_mask = is_mainline & has_lat & has_lng

    # Alternative: use all station types.
    # valid_mask = has_lat & has_lng
    mainline_ids = meta.loc[valid_mask, "ID"].astype(str).unique().tolist()

    print(f"[INFO] Total sensors: {meta.shape[0]}")
    print(f"[INFO] Mainline sensors: {is_mainline.sum()}")
    print(f"[INFO] Mainline sensors with coordinates: {len(mainline_ids)}")

    return set(mainline_ids)


def filter_to_mainline(df: pd.DataFrame, mainline_ids: set):
    cols_keep = [c for c in df.columns if str(c) in mainline_ids]

    if not cols_keep:
        raise ValueError("No station columns match the mainline and coordinate filters.")

    df_filtered = df[cols_keep]
    print(f"[INFO] Station columns: {df.shape[1]} -> {df_filtered.shape[1]} mainline columns")
    return df_filtered


def generate_data_and_idx(df, x_offsets, y_offsets,
                          add_time_of_day, add_day_of_week, add_month_of_year):
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")

    if df.index.isna().any():
        n_bad = df.index.isna().sum()
        print(f"[WARN] DatetimeIndex contains {n_bad} invalid rows. Dropping them.")
        df = df[~df.index.isna()]

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)  # (T, N, 1)

    feature_list = [data]

    if add_time_of_day:
        print('add time-of-day feature')
        time_ind = (df.index - df.index.normalize()) / pd.Timedelta(days=1)
        time_ind = time_ind.values  # (T,)
        time_of_day = np.tile(time_ind[:, None, None], (1, num_nodes, 1))  # (T, N, 1)
        feature_list.append(time_of_day)
    else:
        print('skip time-of-day feature')

    if add_day_of_week:
        print('add day-of-week feature')
        dow = df.index.dayofweek.values  # (T,)
        dow_tiled = np.tile(dow[:, None, None], (1, num_nodes, 1))  # (T, N, 1)
        day_of_week = dow_tiled / 7
        feature_list.append(day_of_week)
    else:
        print('skip day-of-week feature')

    if add_month_of_year:
        print('add month-of-year feature')
        moy = df.index.month.values  # (T,)
        moy_tiled = np.tile(moy[:, None, None], (1, num_nodes, 1))  # (T, N, 1)
        month_of_year = moy_tiled / 12
        feature_list.append(month_of_year)
    else:
        print('skip month-of-year feature')

    data = np.concatenate(feature_list, axis=-1)

    min_t = abs(min(x_offsets))
    max_t = num_samples - abs(max(y_offsets))  # Exclusive
    print('index min and max:', min_t, max_t)
    idx = np.arange(min_t, max_t, 1)
    return data, idx


def generate_train_val_test(args):
    # Select mainline stations.
    mainline_ids = load_mainline_ids_from_dir(args.meta_dir)

    # Load split CSV files.
    train_df = load_filtered_csv(args.train_csv)
    valid_df = load_filtered_csv(args.valid_csv)
    test_df  = load_filtered_csv(args.test_csv)

    print('train shape after 5-minute reindex:', train_df.shape)
    print('validation shape after 5-minute reindex:', valid_df.shape)
    print('test shape after 5-minute reindex:', test_df.shape)

    # Keep mainline stations with coordinates.
    train_df = filter_to_mainline(train_df, mainline_ids)
    valid_df = filter_to_mainline(valid_df, mainline_ids)
    test_df  = filter_to_mainline(test_df,  mainline_ids)

    print('mainline train shape:', train_df.shape)
    print('mainline validation shape:', valid_df.shape)
    print('mainline test shape:', test_df.shape)

    # Check station order and build the mapping.
    cols_train = [str(c) for c in train_df.columns]
    cols_valid = [str(c) for c in valid_df.columns]
    cols_test  = [str(c) for c in test_df.columns]

    if cols_train != cols_valid or cols_train != cols_test:
        print("[ERROR] Station columns or order differ across splits.")
        print("  train[:10] =", cols_train[:10])
        print("  validation[:10] =", cols_valid[:10])
        print("  test[:10] =", cols_test[:10])
        raise ValueError("Station column order differs across splits.")

    # This order defines BiST node and embedding indices.
    station_ids_order = np.array(cols_train, dtype=object)
    id2idx = {sid: i for i, sid in enumerate(cols_train)}

    used_station_ids = set(cols_train)
    print(f"[INFO] Selected mainline stations: {len(used_station_ids)}")
    # Station mapping is complete.

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)

    # Generate data and indices for each split.
    print('\n=== Generate train data ===')
    train_data, idx_train = generate_data_and_idx(
        train_df, x_offsets, y_offsets, args.tod, args.dow, args.moy
    )

    print('\n=== Generate validation data ===')
    valid_data, idx_valid = generate_data_and_idx(
        valid_df, x_offsets, y_offsets, args.tod, args.dow, args.moy
    )

    print('\n=== Generate test data ===')
    test_data, idx_test = generate_data_and_idx(
        test_df, x_offsets, y_offsets, args.tod, args.dow, args.moy
    )

    print('\ntrain data shape:', train_data.shape, 'train index shape:', idx_train.shape)
    print('validation data shape:', valid_data.shape, 'validation index shape:', idx_valid.shape)
    print('test data shape:', test_data.shape,  'test index shape:', idx_test.shape)

    # Normalize using training statistics.
    x_train_for_scaler = train_data[:, :, 0]
    scaler = StandardScaler(mean=x_train_for_scaler.mean(),
                            std=x_train_for_scaler.std())

    train_data[..., 0] = scaler.transform(train_data[..., 0])
    valid_data[..., 0] = scaler.transform(valid_data[..., 0])
    test_data[..., 0] = scaler.transform(test_data[..., 0])

    # Merge split data and convert indices to global positions.
    # data: (T_total, N, F)
    data_all = np.concatenate([train_data, valid_data, test_data], axis=0)

    offset_train = 0
    offset_valid = train_data.shape[0]
    offset_test  = train_data.shape[0] + valid_data.shape[0]

    idx_train_global = idx_train + offset_train
    idx_valid_global = idx_valid + offset_valid
    idx_test_global  = idx_test + offset_test

    # Save data, indices, and the station mapping.
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    npz_path = os.path.join(out_dir, 'his.npz')
    np.savez_compressed(
        npz_path,
        data=data_all,        # (T_total, N, F)
        mean=scaler.mean,
        std=scaler.std,
        stations=station_ids_order,  # Station ID order
    )

    np.save(os.path.join(out_dir, 'idx_train.npy'), idx_train_global)
    np.save(os.path.join(out_dir, 'idx_val.npy'),   idx_valid_global)
    np.save(os.path.join(out_dir, 'idx_test.npy'),  idx_test_global)

    # Save the station-to-index mapping.
    map_path = os.path.join(out_dir, 'station_id2idx.json')
    with open(map_path, 'w') as f:
        json.dump({"id2idx": id2idx, "stations": cols_train}, f, indent=2)
    print(f"\nstation-to-index mapping saved: {map_path}")

    print('\n=== Save complete ===')
    print(f'his.npz path: {npz_path}')
    print(f'data shape: {data_all.shape}')
    print(f'train index shape: {idx_train_global.shape}')
    print(f'validation index shape: {idx_valid_global.shape}')
    print(f'test index shape: {idx_test_global.shape}')
    print(f'scaler mean: {scaler.mean}, std: {scaler.std}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_csv',
        type=str,
        required=True,
        help='Path to the filtered training CSV containing a Timestamp column.'
    )
    parser.add_argument(
        '--valid_csv',
        type=str,
        required=True,
        help='Path to the filtered validation CSV containing a Timestamp column.'
    )
    parser.add_argument(
        '--test_csv',
        type=str,
        required=True,
        help='Path to the filtered test CSV containing a Timestamp column.'
    )

    parser.add_argument(
        '--meta_dir',
        type=str,
        required=True,
        help='Directory containing the station metadata text files.'
    )

    parser.add_argument(
        '--seq_length_x',
        type=int,
        default=12,
        help='Input sequence length.'
    )
    parser.add_argument(
        '--seq_length_y',
        type=int,
        default=12,
        help='Output sequence length.'
    )
    parser.add_argument(
        '--tod',
        type=int,
        default=1,
        help='Add the time-of-day feature (0 or 1).'
    )
    parser.add_argument(
        '--dow',
        type=int,
        default=1,
        help='Add the day-of-week feature (0 or 1).'
    )
    parser.add_argument(
        '--moy',
        type=int,
        default=0,
        help='Add the month-of-year feature (0 or 1).'
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default='ca/2012_hour',
        help='Output directory for his.npz and idx_*.npy files.'
    )

    args = parser.parse_args()
    generate_train_val_test(args)
