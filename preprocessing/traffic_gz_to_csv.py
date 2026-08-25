import glob
import gzip
import multiprocessing
import os
from datetime import datetime

import pandas as pd


class Traffic5MinProcessor:
    """Convert district traffic files into 5-minute train, validation, and test CSV files."""

    def __init__(self, base_folder_path, district_ids):
        self.base_folder_path = base_folder_path
        self.metadata_folder_path = os.path.join(base_folder_path, "metadata")
        self.district_ids = district_ids

        os.makedirs(self.base_folder_path, exist_ok=True)
        os.makedirs(self.metadata_folder_path, exist_ok=True)
        for district_id in self.district_ids:
            os.makedirs(
                os.path.join(self.base_folder_path, f"district{district_id}"),
                exist_ok=True,
            )

        self.fixed_start = datetime(2009, 3, 14, 23)
        self.fixed_end = datetime(2010, 10, 23, 1)
        self.all_5min_timestamps = pd.date_range(
            start=self.fixed_start,
            end=self.fixed_end,
            freq="5min",
        )

        self.train_start = pd.Timestamp("2009-03-15 20:11:32").floor("5min")
        self.train_end = pd.Timestamp("2010-08-27 23:54:34").ceil("5min")
        self.valid_start = pd.Timestamp("2010-08-27 23:55:36").floor("5min")
        self.valid_end = pd.Timestamp("2010-09-24 22:47:59").ceil("5min")
        self.test_start = pd.Timestamp("2010-09-24 22:57:43").floor("5min")
        self.test_end = pd.Timestamp("2010-10-22 18:43:39").ceil("5min")

    def process_all_districts(self):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(self._process_district_wrapper, self.district_ids)

    def _process_district_wrapper(self, district_id):
        self.process_district(district_id)

    def process_district(self, district_id):
        print(f"Processing district {district_id}...")
        folder_path = os.path.join(
            self.base_folder_path,
            f"district{district_id}",
        )
        all_rows = []

        for filename in os.listdir(folder_path):
            if not filename.endswith(".gz"):
                continue

            file_path = os.path.join(folder_path, filename)
            with gzip.open(file_path, "rt", encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split(",")
                    if len(parts) < 10:
                        continue

                    try:
                        timestamp = datetime.strptime(
                            parts[0],
                            "%m/%d/%Y %H:%M:%S",
                        )
                        if not self.fixed_start <= timestamp <= self.fixed_end:
                            continue

                        observed = float(parts[8])
                        total_flow = float(parts[9])
                    except Exception:
                        continue

                    all_rows.append(
                        [timestamp, parts[1], observed, total_flow]
                    )

        district_df = pd.DataFrame(
            all_rows,
            columns=[
                "Timestamp",
                "Station",
                "% Observed",
                "Total Flow",
            ],
        )
        if district_df.empty:
            print(f"No valid data found for district {district_id}.")
            return

        district_df["Station"] = (
            district_df["Station"].astype(str).str.strip()
        )
        district_df = district_df.sort_values(
            ["Station", "Timestamp"]
        )
        self._merge_metadata_and_filter_save(
            district_df,
            district_id,
        )

    def _merge_metadata_and_filter_save(self, district_df, district_id):
        district_id_str = f"{district_id:02d}"
        prefix = f"d{district_id_str}_text_meta_"
        metadata_files = [
            filename
            for filename in os.listdir(self.metadata_folder_path)
            if filename.startswith(prefix) and filename.endswith(".txt")
        ]

        if not metadata_files:
            print(f"No metadata file found for district {district_id}.")
            return

        metadata_path = os.path.join(
            self.metadata_folder_path,
            sorted(metadata_files)[-1],
        )

        try:
            metadata_df = pd.read_csv(
                metadata_path,
                sep="\t",
                dtype=str,
            )[["ID", "Latitude", "Longitude"]].copy()
            metadata_df["ID"] = (
                metadata_df["ID"].astype(str).str.strip()
            )
            metadata_df["Latitude"] = pd.to_numeric(
                metadata_df["Latitude"],
                errors="coerce",
            )
            metadata_df["Longitude"] = pd.to_numeric(
                metadata_df["Longitude"],
                errors="coerce",
            )
            metadata_df = metadata_df.drop_duplicates(
                subset=["ID"],
                keep="last",
            )
        except Exception as error:
            print(f"Failed to read metadata: {metadata_path}\n{error}")
            return

        merged_df = (
            district_df.merge(
                metadata_df,
                left_on="Station",
                right_on="ID",
                how="left",
            )
            .rename(
                columns={
                    "Latitude": "latitude",
                    "Longitude": "longitude",
                }
            )
            .drop(columns=["ID"])
        )
        merged_df = merged_df.dropna(
            subset=["latitude", "longitude"]
        )
        self._filter_reindex_and_save(
            merged_df,
            district_id,
        )

    def _filter_reindex_and_save(self, district_df, district_id):
        grouped = district_df.groupby("Station")
        total_expected = len(self.all_5min_timestamps)

        missing_stats = pd.DataFrame(
            {
                "missing_timestamp_count": (
                    total_expected
                    - grouped["Timestamp"].nunique()
                ),
                "missing_flow_count": grouped["Total Flow"].apply(
                    lambda values: values.isna().sum()
                ),
            }
        ).fillna(0).astype(int)

        threshold = total_expected * 0.80
        valid_stations = [
            station
            for station, row in missing_stats.iterrows()
            if (
                row["missing_timestamp_count"]
                + row["missing_flow_count"]
            )
            <= threshold
        ]
        valid_df = district_df[
            district_df["Station"].isin(valid_stations)
        ]

        reindexed_frames = []
        for station, group in valid_df.groupby("Station"):
            station_df = (
                group.set_index("Timestamp")
                .reindex(self.all_5min_timestamps)
            )
            station_df["Station"] = station
            station_df = station_df.reset_index().rename(
                columns={"index": "Timestamp"}
            )
            reindexed_frames.append(station_df)

        if not reindexed_frames:
            print(f"No valid stations found for district {district_id}.")
            return

        valid_df = pd.concat(
            reindexed_frames,
            ignore_index=True,
        )
        output_path = os.path.join(
            self.base_folder_path,
            f"district{district_id:02d}_5min.csv",
        )
        valid_df.to_csv(output_path, index=False)
        print(
            f"Saved district {district_id}: "
            f"{len(valid_stations)} stations -> {output_path}"
        )

    def merge_all_districts_to_splits(
        self,
        merged_output_filename=None,
    ):
        district_frames = []

        for district_id in self.district_ids:
            district_path = os.path.join(
                self.base_folder_path,
                f"district{district_id:02d}_5min.csv",
            )
            if not os.path.exists(district_path):
                print(f"Missing district file: {district_path}")
                continue

            district_df = pd.read_csv(district_path)
            district_df["Timestamp"] = pd.to_datetime(
                district_df["Timestamp"],
                errors="coerce",
            )
            district_df = district_df.sort_values(
                ["Station", "Timestamp"]
            ).reset_index(drop=True)

            filled_frames = []
            for _, station_df in district_df.groupby("Station"):
                station_df = (
                    station_df.sort_values("Timestamp")
                    .set_index("Timestamp")
                )

                station_df["Total Flow"] = (
                    station_df["Total Flow"]
                    .fillna(station_df["Total Flow"].mean())
                    .clip(lower=0)
                )
                station_df["% Observed"] = (
                    station_df["% Observed"]
                    .fillna(station_df["% Observed"].mean())
                    .clip(lower=0, upper=100)
                )
                filled_frames.append(station_df.reset_index())

            if not filled_frames:
                print(f"No valid stations in district {district_id}.")
                continue

            district_frames.append(
                pd.concat(filled_frames, ignore_index=True)
            )

        if not district_frames:
            print("No district data to merge.")
            return

        merged_df = (
            pd.concat(district_frames, ignore_index=True)
            .sort_values(["Station", "Timestamp"])
            .reset_index(drop=True)
        )
        merged_df["Timestamp"] = pd.to_datetime(
            merged_df["Timestamp"],
            errors="coerce",
        )

        pivot_df = (
            merged_df.pivot(
                index="Timestamp",
                columns="Station",
                values="Total Flow",
            )
            .sort_index()
        )

        train_df = pivot_df.loc[self.train_start:self.train_end]
        valid_df = pivot_df.loc[self.valid_start:self.valid_end]
        test_df = pivot_df.loc[self.test_start:self.test_end]

        train_path = os.path.join(
            self.base_folder_path,
            "train_5min.csv",
        )
        valid_path = os.path.join(
            self.base_folder_path,
            "valid_5min.csv",
        )
        test_path = os.path.join(
            self.base_folder_path,
            "test_5min.csv",
        )

        train_df.to_csv(train_path)
        valid_df.to_csv(valid_path)
        test_df.to_csv(test_path)

        print(
            "Saved train_5min.csv, valid_5min.csv, "
            "and test_5min.csv."
        )
        print(
            f"Train: {train_df.shape}, "
            f"Valid: {valid_df.shape}, "
            f"Test: {test_df.shape}"
        )

        if merged_output_filename:
            merged_path = os.path.join(
                self.base_folder_path,
                merged_output_filename,
            )
            pivot_df.to_csv(merged_path)
            print(f"Saved merged data: {merged_path}")

    @staticmethod
    def generate_metadata_csv(metadata_folder_path):
        metadata_files = glob.glob(
            os.path.join(
                metadata_folder_path,
                "d*_text_meta_*.txt",
            )
        )
        metadata_frames = []

        for metadata_file in metadata_files:
            try:
                metadata_df = pd.read_csv(
                    metadata_file,
                    sep="\t",
                    dtype=str,
                )
                required_columns = {
                    "ID",
                    "Latitude",
                    "Longitude",
                }
                if not required_columns.issubset(metadata_df.columns):
                    continue

                selected_df = metadata_df[
                    ["ID", "Latitude", "Longitude"]
                ].copy()
                selected_df.columns = [
                    "Station",
                    "latitude",
                    "longitude",
                ]
                selected_df["Station"] = (
                    selected_df["Station"]
                    .astype(str)
                    .str.strip()
                )
                selected_df["latitude"] = pd.to_numeric(
                    selected_df["latitude"],
                    errors="coerce",
                )
                selected_df["longitude"] = pd.to_numeric(
                    selected_df["longitude"],
                    errors="coerce",
                )
                metadata_frames.append(selected_df)
            except Exception as error:
                print(
                    f"Failed to read metadata file "
                    f"{os.path.basename(metadata_file)}: {error}"
                )

        if not metadata_frames:
            print("No valid metadata found.")
            return

        merged_metadata = pd.concat(
            metadata_frames,
            ignore_index=True,
        )
        merged_metadata.dropna(
            subset=["Station", "latitude", "longitude"],
            inplace=True,
        )
        merged_metadata["Station"] = (
            merged_metadata["Station"].astype(str).str.strip()
        )
        merged_metadata = merged_metadata.drop_duplicates(
            subset=["Station"],
            keep="last",
        )

        output_path = os.path.join(
            metadata_folder_path,
            "metadata_5min.csv",
        )
        merged_metadata.to_csv(output_path, index=False)
        print(
            f"Saved metadata: {len(merged_metadata)} rows "
            f"-> {output_path}"
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()

    base_folder_path = "dataset/CA_TRAFFIC_DATASET"
    district_ids = [3, 4, 5, 6, 7, 8, 10, 11, 12]

    processor = Traffic5MinProcessor(
        base_folder_path,
        district_ids,
    )
    processor.process_all_districts()
    processor.merge_all_districts_to_splits(
        merged_output_filename=None
    )
    Traffic5MinProcessor.generate_metadata_csv(
        os.path.join(base_folder_path, "metadata")
    )
