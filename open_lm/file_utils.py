import copy
import io
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
from itertools import cycle, islice

import braceexpand
import fsspec
import numpy as np
import torch
import webdataset as wds

from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from open_lm.data import detshuffle2
from open_lm.distributed import is_master


def remote_sync_s3(local_dir, remote_dir):
    # skip epoch_latest which can change during sync.
    result = subprocess.run(
        ["aws", "s3", "sync", local_dir, remote_dir, "--exclude", "*epoch_latest.pt"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        logging.error(f"Error: Failed to sync with S3 bucket {result.stderr.decode('utf-8')}")
        return False

    logging.info(f"Successfully synced with S3 bucket")
    return True


def remote_sync_fsspec(local_dir, remote_dir):
    # FIXME currently this is slow and not recommended. Look into speeding up.
    a = fsspec.get_mapper(local_dir)
    b = fsspec.get_mapper(remote_dir)

    for k in a:
        # skip epoch_latest which can change during sync.
        if "epoch_latest.pt" in k:
            continue

        logging.info(f"Attempting to sync {k}")
        if k in b and len(a[k]) == len(b[k]):
            logging.debug(f"Skipping remote sync for {k}.")
            continue

        try:
            logging.info(f"Successful sync for {k}.")
            b[k] = a[k]
        except Exception as e:
            logging.info(f"Error during remote sync for {k}: {e}")
            return False

    return True


def remote_sync(local_dir, remote_dir, protocol):
    logging.info("Starting remote sync.")
    if protocol == "s3":
        return remote_sync_s3(local_dir, remote_dir)
    elif protocol == "fsspec":
        return remote_sync_fsspec(local_dir, remote_dir)
    else:
        logging.error("Remote protocol not known")
        return False


def keep_running_remote_sync(sync_every, local_dir, remote_dir, protocol):
    while True:
        time.sleep(sync_every)
        remote_sync(local_dir, remote_dir, protocol)


def start_sync_process(sync_every, local_dir, remote_dir, protocol):
    p = multiprocessing.Process(
        target=keep_running_remote_sync,
        args=(sync_every, local_dir, remote_dir, protocol),
    )
    return p


def terminate_sync_process(p: multiprocessing.Process):
    if p is not None and p.is_alive():
        logging.info(f"Terminating remote sync process.")
        p.terminate()


# Note: we are not currently using this save function.
def pt_save(pt_obj, file_path):
    of = fsspec.open(file_path, "wb")
    with of as f:
        torch.save(pt_obj, file_path)


def _pt_load_s3_cp(file_path, map_location=None):
    cmd = f"aws s3 cp {file_path} -"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise Exception(f"Failed to fetch model from s3. stderr: {stderr.decode()}")
    return torch.load(io.BytesIO(stdout), map_location=map_location)


def pt_load(file_path, map_location=None):
    if file_path.startswith("s3"):
        logging.info("Loading remote checkpoint, which may take a bit.")
        return _pt_load_s3_cp(file_path, map_location)
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out


def check_exists(file_path):
    try:
        with fsspec.open(file_path):
            pass
    except FileNotFoundError:
        return False
    return True


def get_metadata_file(path):
    of = fsspec.open(path, "rb")
    with of as f:
        out = f.read()
    out = [json.loads(o) for o in out.decode("utf-8").split("\n")[:-1]]
    return out


def convert_metadata_to_dict(manifest):
    manifest_dict = {}
    for m in manifest:
        try:
            manifest_dict[m["shard"]] = m["num_sequences"]
        except:
            manifest_dict[m["shard"]] = m["num_chunks"]
    return manifest_dict


def get_shards_for_chunk(num_samples, chunk, path):
    """Function to get a chunk of shards to train on.

    Chunks are groups of shards with samples roughly equal to the number of samples
    that will be seen during training. This function uses the dataset manifest
    to split the shards into chunks, and assign shards to each chunk.
    """
    metadata = get_metadata_file(path)
    shard_list = []
    curr_shard_list = []
    chunk_count_list = []
    curr_chunk_count = 0
    for m in metadata:
        try:
            curr_chunk_count += m["num_sequences"]
        except KeyError:
            curr_chunk_count += m["num_chunks"]

        curr_shard_list.append(m["shard"])
        if curr_chunk_count >= num_samples:
            shard_list.append(curr_shard_list)
            chunk_count_list.append(curr_chunk_count)
            curr_shard_list = []
            curr_chunk_count = 0

    # Append remaining shards
    if len(curr_shard_list) > 0:
        shard_list.append(curr_shard_list)
        chunk_count_list.append(curr_chunk_count)

    return (
        shard_list[chunk % len(shard_list)],
        chunk_count_list[chunk % len(chunk_count_list)],
    )


def enough_shards(shard_lists: List[List[str]], min_shards_needed: int):
    for sl in shard_lists:
        if len(sl) < min_shards_needed:
            return False
    return True


def enough_samples(num_samples_per_source: List[List[int]], needed_samples_per_source: List[int]):
    for i, number_per_shard in enumerate(num_samples_per_source):
        if sum(number_per_shard) < needed_samples_per_source[i]:
            return False
    return True


def source_exhausted(paths, shard_list_per_source):
    for i, source in enumerate(paths):
        data = get_metadata_file(source)
        if len(data) < len(shard_list_per_source[i]):
            return True
    return False


def count_small_shards(path, ratio=0.9):
    """Count the number of shards with significantly fewer sequences than the largest shard.

    Small shards are defined as those that have size less than a ratio (default 90%) of the size of the largest shard.
    """
    shard_sizes = []
    data = get_metadata_file(path)
    for item in data:
        try:
            shard_sizes.append(item["num_sequences"])
        except KeyError:
            shard_sizes.append(item["num_chunks"])

    shard_sizes = np.array(shard_sizes)

    return np.sum(shard_sizes < ratio * max(shard_sizes))


def are_sources_imbalanced_with_each_other(paths, ratio=2):
    median_shard_size_per_source = []
    for p in paths:
        shard_sizes = []
        data = get_metadata_file(p)
        for item in data:
            try:
                shard_sizes.append(item["num_sequences"])
            except KeyError:
                shard_sizes.append(item["num_chunks"])

        median_shard_size_per_source.append(np.median(shard_sizes))

    return max(median_shard_size_per_source) > ratio * min(median_shard_size_per_source)


def log_num_checkpoints(total_steps, args):
    """Log the number of checkpoints that will be made.

    This function counts the number of checkpoints to be made, and logs that number, printing out a warning if that
    number is different than expected.
    """

    assert args.dataset_manifest is not None, "log_num_checkpoints is needed only in the sampling w/o replacement case."

    steps_done = 0
    tokens_seen = 0
    next_shard_per_source = [0 for _ in range(len(args.dataset_manifest))]
    manifests = [get_metadata_file(m) for m in args.dataset_manifest]
    manifests = [convert_metadata_to_dict(m) for m in manifests]
    num_sources = len(manifests)
    num_global_workers = args.world_size * args.workers
    checkpoints_made = 0

    if is_master(args):
        logging.info("Precounting number of steps / tokens seen per checkpoint:")

    while steps_done < total_steps:
        shard_strings_per_source, _, next_shard_per_source = get_string_for_epoch(
            args.train_num_samples,
            next_shard_per_source,
            args.dataset_manifest,
            args.train_data_mix_weights,
            args.workers,
            args.world_size,
        )

        shard_ids_per_source = [[Path(url.split("/")[-1]).with_suffix("").name for url in braceexpand.braceexpand(shard_string)] for shard_string in shard_strings_per_source]
        num_samples_per_shard_per_source = [[manifests[i][shard_id] for shard_id in shard_ids_per_source[i]] for i in range(len(manifests))]

        num_samples_per_global_worker_per_source = [[0 for _ in range(num_global_workers)] for _ in range(num_sources)]

        # wds.split_by_node and wds.split_by_worker work in conjunction by assigning tars to global workers
        # we can simulate this by first computing the samples that will be seen per source and per global worker

        for source_id in range(num_sources):
            for i, elem in enumerate(num_samples_per_shard_per_source[source_id]):
                num_samples_per_global_worker_per_source[source_id][i % num_global_workers] += elem
        
        # we then find the batches that each worker will produce
        num_batches_per_global_worker_per_source = [np.array([n // args.global_batch_size for n in nsamples]) for nsamples in num_samples_per_global_worker_per_source]
        num_batches_per_global_worker = sum(num_batches_per_global_worker_per_source)

        # The way wds.split_by_node and wds.split_by_worker are set up, each global worker in the above sequence
        # is indexed by its local worker id and its gpu proc id, WITH THE SECOND CHANGING FASTER
        # this means that global_worker_id mod world size is the correct assignment of the worker to a gpu
        num_batches_per_gpu = [0 for _ in range(args.world_size)]
        for worker_id in range(num_global_workers):
            gpu_id = worker_id % args.world_size 
            num_batches_per_gpu[gpu_id] += num_batches_per_global_worker[worker_id]

        # Each gpu will serve as many batches as it can, and the checkpoint will end when one proc runs out of data.
        steps_epoch = min(num_batches_per_gpu)

        steps_done += steps_epoch
        if steps_done > total_steps:
            steps_done = total_steps
        tokens_seen = steps_done * args.global_batch_size * args.seq_len
        checkpoints_made += 1

        if is_master(args):
            logging.info(f"==> Checkpoint {checkpoints_made}, steps {steps_done}, tokens seen {tokens_seen}")

    if is_master(args):
        logging.info(
            f"Number of checkpoints to be made: {checkpoints_made}."
            f"Number will be greater in case of unexpected failures leading to the use of more shards"
        )

        if checkpoints_made != args.epochs:
            logging.warning(
                f"{args.epochs} were requested, but {checkpoints_made} will be made. This behavior is a best effort in "
                f"checkpointing for the desired amount of epochs, and depends on the number of workers and gpus used, "
                f"as well as the size of the shards themselves."
            )

    return


def get_string_for_epoch(
    num_samples: int,
    starting_points: List[int],
    paths: List[str],
    weights: Optional[List[float]],
    num_workers_per_gpu: int,
    world_size: int,
    multi_epoch=False,
):
    """See _single_epoch_string for full docstring."""
    if multi_epoch:
        raise NotImplementedError("Multiple passes over the dataset not fully supported yet.")
    else:
        return _single_epoch_string(num_samples, starting_points, paths, weights, num_workers_per_gpu, world_size)


def _multi_epoch_string(num_samples, starting_chunk, paths, weights, min_shards_needed):
    """Multi epoch string training."""

    raise NotImplementedError("Function not fully supported yet.")

    if weights is None:
        weights = [1.0 / len(paths) for _ in range(len(paths))]
    needed_samples_per_source = [int(np.ceil(weights[i] * num_samples / sum(weights))) for i in range(len(weights))]
    shard_strings_per_source = []
    next_chunk = starting_chunk
    shard_list_per_source = [[] for _ in range(len(paths))]
    num_samples_per_source = [0 for _ in range(len(paths))]
    while not enough_shards(shard_list_per_source, min_shards_needed) or not enough_samples(
        num_samples_per_source, needed_samples_per_source
    ):
        for i, source_path in enumerate(paths):
            shard_list_source, num_samples_source = get_shards_for_chunk(
                needed_samples_per_source[i], next_chunk, source_path
            )
            shard_list_per_source[i].extend(shard_list_source)
            num_samples_per_source[i] += num_samples_source
        next_chunk += 1
        if source_exhausted(paths, shard_list_per_source):
            logging.warning(
                "Number of shards requested for a single epoch is more than the number of shards available. "
                "Consider lowering the number of workers and / or the number of GPUs, to avoid continuous "
                "reuse of samples."
            )

    for i, source_path in enumerate(paths):
        shard_list_source = shard_list_per_source[i]
        num_samples_source = num_samples_per_source[i]
        shard_root_source = "/".join(source_path.split("/")[:-1]) + "/"
        if len(shard_list_source) == 1:
            shard_string_source = shard_root_source + shard_list_source[0] + ".tar"
        else:
            shard_string_source = shard_root_source + "{" + ",".join(shard_list_source) + "}.tar"
        if source_path.startswith("s3"):
            shard_string_source = f"pipe:aws s3 cp {shard_string_source} -"
        shard_strings_per_source.append(shard_string_source)

    return shard_strings_per_source, num_samples_per_source, next_chunk


def _single_epoch_string(
    num_samples: int,
    starting_shard_per_source: List[int],
    paths: List[str],
    weights: Optional[List[float]],
    num_workers_per_gpu: int,
    world_size: int,
):
    """Retrieve shards to train on for a particular checkpoint.

    Currently only a single source is fully supported yet.

    Args:
        num_samples: Total number of samples required.
        starting_shard_per_source: First shard per source that has not been consumed yet.
        paths: Paths to source manifests.
        weights: Weighting between sources. If None, it is assumed to be uniform.
        num_workers_per_gpu: Number of workers per gpu process.
        world_size: Total number of gpus used for training.
    """

    num_sources = len(paths)

    if num_sources > 1:
        logging.warning(
            "Multiple sources are not supported fully as of now. It is advised to combine the data into a single "
            "source, by using datapreprocess/ray/tokenize_shuffle.py. Best effort will be done to mix data at the "
            "desired ratio."
        )
        if are_sources_imbalanced_with_each_other(paths):
            logging.warning(
                "Sources contain highly imbalanced shards (largest median shard size of a source is >2x the smallest "
                "median size of a source). This will lead to deteriorated performance (less frequent checkpoints, "
                "data being skipped, and inaccurate mixing). It is STRONGLY advised to combine into one source."
            )

    for path in paths:
        num_small_shards = count_small_shards(path)
        if num_small_shards > 0:
            logging.warning(
                f"Source defined by {path} contains {num_small_shards} shards that are smaller than 90% the size of "
                f"the largest shard. These shards might cause deterioration in performance, with more samples being "
                f"skipped than necessary. It is advised to make the shards more uniform."
            )

    if weights is None:
        weights = [1.0 / num_sources for _ in range(num_sources)]

    assert len(weights) == num_sources, "One weight is needed per source."

    needed_samples_per_source = [int(np.ceil(weights[i] * num_samples / sum(weights))) for i in range(num_sources)]

    manifests = [get_metadata_file(path) for path in paths]
    shard_strings_per_source = []
    next_shard_per_source = copy.deepcopy(starting_shard_per_source)
    shard_list_per_source = [[] for _ in range(num_sources)]
    num_samples_per_source = [[] for _ in range(num_sources)]

    total_num_workers = num_workers_per_gpu * world_size
    while not enough_shards(shard_list_per_source, total_num_workers) or not enough_samples(
        num_samples_per_source, needed_samples_per_source
    ):
        try:
            for i in range(num_sources):
                # Add shards incrementally
                shard_name = manifests[i][next_shard_per_source[i]]["shard"]
                try:
                    num_samples_shard = manifests[i][next_shard_per_source[i]]["num_sequences"]
                except KeyError:
                    num_samples_shard = manifests[i][next_shard_per_source[i]]["num_chunks"]

                shard_list_per_source[i].append(shard_name)
                num_samples_per_source[i].append(num_samples_shard)

                next_shard_per_source[i] += 1

        except IndexError as e:
            logging.error(
                "Number of shards requested for a single epoch is more than the number of shards available. This means "
                "that the amount of data requested to train on is more than the dataloader can serve. This can either "
                "happen because there are not enough data to begin with, or data being skipped due to rounding errors. "
                "To alleviate the latter, consider making more uniform shards, and using less workers/GPUs. This will "
                "allow for better use of the dataset."
            )
            raise e

    for i in range(num_sources):
        # Ensure the number of shards is a multiple of number of workers, so each worker has the same
        # number of shards.
        #
        # This is a heuristic to minimize how much data we discard when trying to ensure each worker has
        # the same number of samples. Shards tend to have similar number of samples, so an extra shard
        # in a worker will likely get discarded.
        num_multiples = len(shard_list_per_source[i]) // total_num_workers

        shard_list_per_source[i] = shard_list_per_source[i][: num_multiples * total_num_workers]
        num_samples_per_source[i] = num_samples_per_source[i][: num_multiples * total_num_workers]

        # Put back unused shards.
        next_shard_per_source[i] = starting_shard_per_source[i] + len(shard_list_per_source[i])

    num_samples_per_source = [sum(n) for n in num_samples_per_source]

    for i, source_path in enumerate(paths):
        # Combine into a single shard string for training
        shard_list_source = shard_list_per_source[i]
        shard_root_source = "/".join(source_path.split("/")[:-1]) + "/"
        if len(shard_list_source) == 1:
            shard_string_source = shard_root_source + shard_list_source[0] + ".tar"
        else:
            shard_string_source = shard_root_source + "{" + ",".join(shard_list_source) + "}.tar"
        if source_path.startswith("s3"):
            shard_string_source = f"pipe:aws s3 cp {shard_string_source} -"
        shard_strings_per_source.append(shard_string_source)

    return shard_strings_per_source, num_samples_per_source, next_shard_per_source
