## Accepted at CIKM 2026

# LLM Meets Traffic: Traffic-Aware Large Language Models for Next POI Recommendation in Vehicle Navigation Interfaces

This repository contains the implementation of **TraPOI**, a traffic-aware LLM framework for next POI recommendation. TraPOI incorporates POI, GPS, and traffic representations into an LLM and uses multi-view candidate retrieval to provide both spatially feasible and preference-relevant POI candidates.

## Data Preparation

Download the TraPOI dataset from Google Drive:

- [TraPOI Dataset](https://drive.google.com/drive/folders/1xXarJKVwjUZdMI-_juGhHE2iSWu0Tbz-?usp=sharing)

Due to the large size of the raw traffic dataset, only the data for District 03 are provided in the Google Drive folder.

The raw traffic data for the remaining districts (Districts 04, 05, 06, 07, 08, 10, 11, and 12) can be downloaded from the Caltrans Performance Measurement System (PeMS):

- [Caltrans PeMS](http://pems.dot.ca.gov)

POI check-in data are preprocessed following [LLM4POI](https://github.com/neolifer/LLM4POI).

The POI and traffic encoders are trained using the official implementations of [STHGCN](https://github.com/alipay/Spatio-Temporal-Hypergraph-Model) and [BiST](https://github.com/PoorOtterBob/BiST), respectively.

### Traffic Data Preparation

The traffic preprocessing scripts are located in the `preprocessing` directory.

#### Raw Traffic Preprocessing

`preprocessing/traffic_gz_to_csv.py` converts the raw district-level traffic `.gz` files into 5-minute traffic CSV files.

```bash
python preprocessing/traffic_gz_to_csv.py
```

The script first processes each district separately and then generates the following merged outputs:

```text
train_5min.csv
valid_5min.csv
test_5min.csv
metadata_5min.csv
```

#### BiST Input Generation

`preprocessing/generate_data_ca.py` converts the processed traffic CSV files into the input format used by BiST.

```bash
python -u preprocessing/generate_data_ca.py \
  --train_csv <YOUR_TRAIN_TRAFFIC_CSV_PATH> \
  --valid_csv <YOUR_VALIDATION_TRAFFIC_CSV_PATH> \
  --test_csv <YOUR_TEST_TRAFFIC_CSV_PATH> \
  --meta_dir <YOUR_TRAFFIC_METADATA_DIRECTORY> \
  --seq_length_x 12 \
  --seq_length_y 12 \
  --tod 1 \
  --dow 1 \
  --moy 0 \
  --out_dir <YOUR_BIST_DATA_OUTPUT_DIRECTORY>
```

## Candidate Retrieval and Prompt Generation

`build_prompt_with_candidates.py` constructs the multimodal prompts used by TraPOI.

### Run Prompt Generation

```bash
python -u build_prompt_with_candidates.py \
  -dataset_name ca \
  -traffic_meta_dir <YOUR_TRAFFIC_METADATA_DIRECTORY> \
  -traffic_npz_path <YOUR_BIST_HIS_NPZ_PATH> \
  -k_nearest_traffic 5 \
  --checkpoint_path <YOUR_STHGCN_CHECKPOINT_PATH>
```

The script outputs next-POI prompts containing historical and current trajectories, nearby traffic information, GPS tokens, and distance- and embedding-based POI candidates.

## Traffic Embedding Extraction

`extract_and_prepack_traffic_emb.py` extracts traffic representations from a pretrained BiST model and prepares the tokenized data used for multimodal LLM training.

### Run Traffic Embedding Extraction

```bash
CUDA_VISIBLE_DEVICES=<YOUR_GPU_ID> python -u extract_and_prepack_traffic_emb.py \
  --qa_json <YOUR_QA_JSON_PATH> \
  --traffic_pairs_json <YOUR_TRAFFIC_PAIRS_JSON_PATH> \
  --traffic_id_map_json <YOUR_TRAFFIC_ID_MAP_JSON_PATH> \
  --poi_embedding_npy <YOUR_POI_EMBEDDING_NPY_PATH> \
  --out_dir <YOUR_OUTPUT_DIRECTORY> \
  --data_path <YOUR_BIST_HIS_NPZ_PATH> \
  --ckpt <YOUR_BIST_CHECKPOINT_PATH> \
  --train_time_csv <YOUR_TRAIN_TIMESTAMP_CSV_PATH> \
  --val_time_csv <YOUR_VALIDATION_TIMESTAMP_CSV_PATH> \
  --test_time_csv <YOUR_TEST_TIMESTAMP_CSV_PATH> \
  --bist_dataset CA \
  --geohash_vocab_path <YOUR_GEOHASH_VOCAB_PATH>
```

The script generates:

```text
prepacked_train.pt
traffic_compact.pt
```

`prepacked_train.pt` contains the tokenized question-answer examples and the compact traffic-embedding index assigned to each matched traffic-token position.

`traffic_compact.pt` contains the extracted BiST traffic representations and the mapping from each requested time-station pair to its compact embedding index.

## LLM Training

`finetune_poi_traffic_gps_sep_LLM_with_projector.py` trains the LLM using the prepacked question-answer examples, POI embeddings, traffic embeddings, and GPS embeddings.

### Run Training

```bash
CUDA_VISIBLE_DEVICES=<YOUR_GPU_ID> python -u finetune_poi_traffic_gps_sep_LLM_with_projector.py \
  --model_name_or_path yukang/Llama-2-7b-longlora-32k-ft \
  --output_dir <YOUR_MODEL_OUTPUT_DIRECTORY> \
  --poi_embedding_path <YOUR_POI_EMBEDDING_NPY_PATH> \
  --prepacked_pt <YOUR_PREPACKED_TRAIN_PT_PATH> \
  --traffic_compact_pt <YOUR_TRAFFIC_COMPACT_PT_PATH> \
  --geohash_vocab_path <YOUR_GEOHASH_VOCAB_PATH> \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_steps 20 \
  --lr_scheduler_type constant_with_warmup \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 400 \
```

The training script saves the LoRA adapters, tokenizer, multimodal projectors, geohash embedding, and trainer state under the specified output directory.

## Evaluation

`test_next_poi_traffic_gps_sep_LLM.py` evaluates the trained TraPOI model on the next-POI prediction task.

### Prepare Test Traffic Embeddings

Before evaluation, run `extract_and_prepack_traffic_emb.py` on the test QA data to generate test-specific `prepacked_train.pt` and `traffic_compact.pt` files.

```bash
CUDA_VISIBLE_DEVICES=<YOUR_GPU_ID> python -u extract_and_prepack_traffic_emb.py \
  --qa_json <YOUR_TEST_QA_JSON_PATH> \
  --traffic_pairs_json <YOUR_TEST_TRAFFIC_PAIRS_JSON_PATH> \
  --traffic_id_map_json <YOUR_TRAFFIC_ID_MAP_JSON_PATH> \
  --poi_embedding_npy <YOUR_POI_EMBEDDING_NPY_PATH> \
  --out_dir <YOUR_TEST_TRAFFIC_EMBEDDING_OUTPUT_DIRECTORY> \
  --data_path <YOUR_BIST_HIS_NPZ_PATH> \
  --ckpt <YOUR_BIST_CHECKPOINT_PATH> \
  --train_time_csv <YOUR_TRAIN_TIMESTAMP_CSV_PATH> \
  --val_time_csv <YOUR_VALIDATION_TIMESTAMP_CSV_PATH> \
  --test_time_csv <YOUR_TEST_TIMESTAMP_CSV_PATH> \
  --bist_dataset CA \
  --geohash_vocab_path <YOUR_GEOHASH_VOCAB_PATH>
```

### Run Evaluation

```bash
CUDA_VISIBLE_DEVICES=<YOUR_GPU_ID> python -u test_next_poi_traffic_gps_sep_LLM.py \
  --base_model yukang/Llama-2-7b-longlora-32k-ft \
  --test_type llm \
  --test_file <YOUR_TEST_QA_TEXT_PATH> \
  --output_dir <YOUR_TRAINED_CHECKPOINT_DIRECTORY> \
  --poi_embedding_path <YOUR_POI_EMBEDDING_NPY_PATH> \
  --poi_projector_path <YOUR_POI_PROJECTOR_PATH> \
  --prepacked_pt <YOUR_TEST_PREPACKED_PT_PATH> \
  --traffic_compact_pt <YOUR_TEST_TRAFFIC_COMPACT_PT_PATH> \
  --traffic_projector_path <YOUR_TRAFFIC_PROJECTOR_PATH> \
  --traffic_id_map_json <YOUR_TRAFFIC_ID_MAP_JSON_PATH> \
  --gps_projector_path <YOUR_GPS_PROJECTOR_PATH> \
  --geohash_vocab_path <YOUR_GEOHASH_VOCAB_PATH>
```
