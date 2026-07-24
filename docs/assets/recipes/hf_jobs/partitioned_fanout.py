# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
# ]
# ///
"""Partitioned Multi-Job Generation Recipe

Scale a seed-driven Data Designer workload horizontally by splitting the seed dataset into N
partitions with `PartitionBlock` and running one job per partition. Every job runs this same
script with a different `--partition-index`; a small launcher loop submits all partitions to
Hugging Face Jobs, but any scheduler that can run a Python script works the same way.

Partition selection uses Data Designer's native seed selection strategies (see the Seed Datasets
guide). The jobs are fully independent — each reads only its own partition of the seed dataset and
writes its own output shard — so there is no cross-job coordination, and a failed partition is
retried by resubmitting that single index.

The seed dataset can be:
    - the bundled demo seed (topic/audience rows written by the script itself; the default),
    - a local file via `--seed-path` (CSV, Parquet, or JSON), or
    - a dataset on the Hugging Face Hub via `--hf-seed-path`.

With ordered sampling, generation cycles through the selected partition if `--num-records` exceeds
the partition size. To make exactly one pass over the seed data, set `--num-records` to the number
of rows in the partition (dataset size divided by `--num-partitions`).

Prerequisites:
    - NVIDIA_API_KEY environment variable for the default "nvidia-text" model alias.
    - For Hugging Face Jobs runs: `huggingface_hub` installed locally (provides the `hf` CLI)
      and authentication via `hf auth login`.

Run:
    # Locally, over the bundled demo seed (single partition)
    uv run partitioned_fanout.py --num-records 10

    # Locally, partition 0 of 4
    uv run partitioned_fanout.py --partition-index 0 --num-partitions 4 --num-records 18

    # Fan out across 4 Hugging Face Jobs, collecting shards in one Hub dataset
    for k in 0 1 2 3; do
        hf jobs uv run --detach --flavor cpu-basic --timeout 1h --secrets HF_TOKEN \
            partitioned_fanout.py \
            --partition-index $k --num-partitions 4 --num-records 18 \
            --endpoint https://router.huggingface.co/v1 --model openai/gpt-oss-20b --api-key-env HF_TOKEN \
            --push-to-hub <your-username>/partitioned-fanout-demo
    done
"""

from __future__ import annotations

import tempfile
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults

CUSTOM_PROVIDER_NAME = "custom-endpoint"
CUSTOM_MODEL_ALIAS = "custom-endpoint-model"

DEMO_TOPICS = [
    "tidal pools",
    "cast iron cooking",
    "city pigeons",
    "old maps",
    "lighthouses",
    "sourdough starters",
    "morse code",
    "alpine trains",
    "library card catalogs",
    "weather balloons",
    "fountain pens",
    "kelp forests",
    "canal locks",
    "beekeeping",
    "shorthand writing",
    "star charts",
    "tide tables",
    "letterpress printing",
    "ham radio",
    "bird migration",
    "root cellars",
    "windmills",
    "knot tying",
    "paper marbling",
]
DEMO_AUDIENCES = ["curious kids", "curious adults", "subject-matter experts"]


def write_demo_seed(directory: Path) -> Path:
    """Write a small demo seed dataset (72 topic/audience rows), used when no seed is provided."""
    rows = [{"topic": topic, "audience": audience} for topic in DEMO_TOPICS for audience in DEMO_AUDIENCES]
    path = directory / "demo_seed.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def resolve_seed_source(
    seed_path: str | None, hf_seed_path: str | None
) -> dd.LocalFileSeedSource | dd.HuggingFaceSeedSource:
    if seed_path is not None and hf_seed_path is not None:
        raise ValueError("Provide at most one of --seed-path and --hf-seed-path.")
    if hf_seed_path is not None:
        return dd.HuggingFaceSeedSource(path=hf_seed_path)
    if seed_path is not None:
        return dd.LocalFileSeedSource(path=seed_path)
    return dd.LocalFileSeedSource(path=str(write_demo_seed(Path(tempfile.mkdtemp(prefix="partitioned_fanout_")))))


def resolve_selection_strategy(
    partition_index: int | None,
    num_partitions: int | None,
) -> dd.PartitionBlock | None:
    if partition_index is None and num_partitions is None:
        return None
    if partition_index is None or num_partitions is None:
        raise ValueError("--partition-index and --num-partitions must be provided together.")
    return dd.PartitionBlock(index=partition_index, num_partitions=num_partitions)


def build_config(
    model_alias: str,
    seed_source: dd.LocalFileSeedSource | dd.HuggingFaceSeedSource,
    selection_strategy: dd.PartitionBlock | None,
    model_configs: list[dd.ModelConfig] | None = None,
) -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    config_builder.with_seed_dataset(seed_source, selection_strategy=selection_strategy)

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="explainer",
            model_alias=model_alias,
            prompt=(
                "Write a clear, engaging two-paragraph explainer about {{ topic }} for {{ audience }}. "
                "Respond with only the explainer."
            ),
        )
    )

    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="quiz_question",
            model_alias=model_alias,
            prompt=(
                "Write one quiz question that tests whether {{ audience }} understood the key idea of "
                "the following explainer about {{ topic }}.\n\n{{ explainer }}\n\n"
                "Respond with only the question."
            ),
        )
    )

    return config_builder


def build_custom_model(
    endpoint: str,
    model: str,
    api_key_env: str | None,
    max_parallel_requests: int,
) -> tuple[dd.ModelProvider, dd.ModelConfig]:
    """Build a provider and model config for an arbitrary OpenAI-compatible endpoint."""
    provider = dd.ModelProvider(
        name=CUSTOM_PROVIDER_NAME,
        endpoint=endpoint,
        api_key=api_key_env,
    )
    model_config = dd.ModelConfig(
        alias=CUSTOM_MODEL_ALIAS,
        model=model,
        provider=CUSTOM_PROVIDER_NAME,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
            max_parallel_requests=max_parallel_requests,
        ),
        # Not every OpenAI-compatible gateway implements the model-listing endpoint used by the health check.
        skip_health_check=True,
    )
    return provider, model_config


def push_partition_to_hub(
    results: DatasetCreationResults,
    *,
    repo_id: str,
    partition_index: int,
    num_partitions: int,
    private: bool,
) -> None:
    """Upload this partition's output as one parquet shard of a shared Hub dataset."""
    from huggingface_hub import HfApi

    shard_name = f"partition-{partition_index:05d}-of-{num_partitions:05d}.parquet"
    export_path = results.export(Path(tempfile.mkdtemp(prefix="partitioned_fanout_out_")) / shard_name)
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=export_path,
        path_in_repo=f"data/{shard_name}",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Shard uploaded: https://huggingface.co/datasets/{repo_id}/tree/main/data")


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model-alias", type=str, default="nvidia-text")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Optional OpenAI-compatible endpoint URL. Overrides --model-alias when provided with --model.",
    )
    parser.add_argument("--model", type=str, default=None, help="Model identifier to request from --endpoint.")
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Name of the environment variable holding the API key for --endpoint (e.g. HF_TOKEN).",
    )
    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        default=16,
        help="Concurrent requests to --endpoint. See the Architecture & Performance guide for tuning.",
    )
    parser.add_argument(
        "--seed-path",
        type=str,
        default=None,
        help="Optional local seed file (CSV, Parquet, or JSON). Defaults to a bundled demo seed.",
    )
    parser.add_argument(
        "--hf-seed-path",
        type=str,
        default=None,
        help="Optional Hugging Face Hub seed path, e.g. 'datasets/<user>/<dataset>/data/*.parquet'.",
    )
    parser.add_argument(
        "--partition-index",
        type=int,
        default=None,
        help="Zero-based index of the seed partition this job should process.",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=None,
        help="Total number of partitions the seed dataset is split into.",
    )
    parser.add_argument("--num-records", type=int, default=5)
    parser.add_argument("--artifact-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="partitioned_fanout")
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Optional Hugging Face dataset repo id; each partition uploads one parquet shard to it.",
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub dataset repo as private.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if (args.endpoint is None) != (args.model is None):
        raise ValueError("--endpoint and --model must be provided together.")

    seed_source = resolve_seed_source(args.seed_path, args.hf_seed_path)
    selection_strategy = resolve_selection_strategy(args.partition_index, args.num_partitions)

    model_providers: list[dd.ModelProvider] | None = None
    model_configs: list[dd.ModelConfig] | None = None
    model_alias = args.model_alias
    if args.endpoint is not None:
        provider, model_config = build_custom_model(
            endpoint=args.endpoint,
            model=args.model,
            api_key_env=args.api_key_env,
            max_parallel_requests=args.max_parallel_requests,
        )
        model_providers = [provider]
        model_configs = [model_config]
        model_alias = CUSTOM_MODEL_ALIAS

    config_builder = build_config(
        model_alias=model_alias,
        seed_source=seed_source,
        selection_strategy=selection_strategy,
        model_configs=model_configs,
    )
    data_designer = DataDesigner(artifact_path=args.artifact_path, model_providers=model_providers)
    results = data_designer.create(config_builder, num_records=args.num_records, dataset_name=args.dataset_name)

    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")
    results.display_sample_record()

    if args.push_to_hub is not None:
        push_partition_to_hub(
            results,
            repo_id=args.push_to_hub,
            partition_index=args.partition_index or 0,
            num_partitions=args.num_partitions or 1,
            private=args.private,
        )


if __name__ == "__main__":
    main()
