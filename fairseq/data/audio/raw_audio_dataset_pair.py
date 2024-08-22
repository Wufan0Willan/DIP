# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
import io

import numpy as np
import torch
import torch.nn.functional as F

from .. import FairseqDataset
from ..data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel
import random
import pdb

logger = logging.getLogger(__name__)


class RawAudioDatasetPair(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        compute_mask_indices=False,
        **mask_compute_kwargs,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize
        self.compute_mask_indices = compute_mask_indices
        if self.compute_mask_indices:
            self.mask_compute_kwargs = mask_compute_kwargs
            self._features_size_map = {}
            self._C = mask_compute_kwargs["encoder_embed_dim"]
            self._conv_feature_layers = eval(mask_compute_kwargs["conv_feature_layers"])

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    def _compute_mask_indices(self, dims, padding_mask):
        B, T, C = dims
        mask_indices, mask_channel_indices = None, None
        if self.mask_compute_kwargs["mask_prob"] > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_compute_kwargs["mask_prob"],
                self.mask_compute_kwargs["mask_length"],
                self.mask_compute_kwargs["mask_selection"],
                self.mask_compute_kwargs["mask_other"],
                min_masks=2,
                no_overlap=self.mask_compute_kwargs["no_mask_overlap"],
                min_space=self.mask_compute_kwargs["mask_min_space"],
            )
            mask_indices = torch.from_numpy(mask_indices)
        if self.mask_compute_kwargs["mask_channel_prob"] > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_compute_kwargs["mask_channel_prob"],
                self.mask_compute_kwargs["mask_channel_length"],
                self.mask_compute_kwargs["mask_channel_selection"],
                self.mask_compute_kwargs["mask_channel_other"],
                no_overlap=self.mask_compute_kwargs["no_mask_channel_overlap"],
                min_space=self.mask_compute_kwargs["mask_channel_min_space"],
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            )

        return mask_indices, mask_channel_indices

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value=value)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        #pdb.set_trace()
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        targets = [s["target"] for s in samples]
        source_sizes = [len(s) for s in sources]
        target_sizes = [len(s) for s in targets]

        if self.pad:
            source_size = min(max(source_sizes), self.max_sample_size)
        else:
            source_size = min(min(source_sizes), self.max_sample_size)

        if self.pad:
            target_size = min(max(target_sizes), self.max_sample_size)
        else:
            target_size = min(min(target_sizes), self.max_sample_size)


        collated_sources = sources[0].new_zeros(len(sources), source_size)
        collated_targets = targets[0].new_zeros(len(targets), target_size)
        #print("pad:{0}".format(self.pad))
        source_padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        target_padding_mask = (
            torch.BoolTensor(collated_targets.shape).fill_(False) if self.pad else None
        )

        for i, (source, size) in enumerate(zip(sources, source_sizes)):
            diff = size - source_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                source_padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, source_size)

        for i, (target, size) in enumerate(zip(targets, target_sizes)):
            diff = size - target_size
            if diff == 0:
                collated_targets[i] = target
            elif diff < 0:
                assert self.pad
                collated_targets[i] = torch.cat(
                    [target, target.new_full((-diff,), 0.0)]
                )
                target_padding_mask[i, diff:] = True
            else:
                collated_targets[i] = self.crop_to_max_size(target, target_size)

        #print(collated_indexs)
        #print("************************")
        input = {"source": collated_sources, "target": collated_targets}
        #pdb.set_trace()
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)
        #pdb.set_trace()
        if self.compute_mask_indices:
            B = input["source"].size(0)
            T = self._get_mask_indices_dims(input["source"].size(-1))
            padding_mask_reshaped = input["padding_mask"].clone()
            extra = padding_mask_reshaped.size(1) % T
            if extra > 0:
                padding_mask_reshaped = padding_mask_reshaped[:, :-extra]
            padding_mask_reshaped = padding_mask_reshaped.view(
                padding_mask_reshaped.size(0), T, -1
            )
            padding_mask_reshaped = padding_mask_reshaped.all(-1)
            input["padding_count"] = padding_mask_reshaped.sum(-1).max().item()
            mask_indices, mask_channel_indices = self._compute_mask_indices(
                (B, T, self._C),
                padding_mask_reshaped,
            )
            input["mask_indices"] = mask_indices
            input["mask_channel_indices"] = mask_channel_indices
            out["sample_size"] = mask_indices.sum().item()

        out["net_input"] = input
        return out

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in self._conv_feature_layers:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size,
                )
            )
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets,
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the audio dataset: "
                f"{self.buckets}"
            )


class FileAudioDatasetPair(RawAudioDatasetPair):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        text_compression_level=TextCompressionLevel.none,
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )

        self.text_compressor = TextCompressor(level=text_compression_level)

        source_skipped = 0
        self.source_fnames = []
        source_sizes = []
        self.source_skipped_indices = set()
        #pdb.set_trace()
        manifest_base = manifest_path.split(r".")[0]
        self.source_path = "_".join([manifest_base, "source.tsv"])
        with open(self.source_path, "r") as f:
            self.source_root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_sample_size is not None and sz < min_sample_size:
                    source_skipped += 1
                    self.source_skipped_indices.add(i)
                    continue
                self.source_fnames.append(self.text_compressor.compress(items[0]))
                source_sizes.append(sz)
        logger.info(f"loaded {len(self.source_fnames)}, skipped {source_skipped} samples for source dataset")

        self.source_sizes = np.array(source_sizes, dtype=np.int64)
        self.source_length = len(source_sizes)
        self.source_range = range(self.source_length)

        target_skipped = 0
        self.target_fnames = []
        target_sizes = []
        self.target_skipped_indices = set()

        self.target_path = "_".join([manifest_base, "target.tsv"])
        with open(self.target_path, "r") as f:
            self.target_root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_sample_size is not None and sz < min_sample_size:
                    target_skipped += 1
                    self.target_skipped_indices.add(i)
                    continue
                self.target_fnames.append(self.text_compressor.compress(items[0]))
                target_sizes.append(sz)
        logger.info(f"loaded {len(self.target_fnames)}, skipped {target_skipped} samples for target dataset")

        self.target_sizes = np.array(target_sizes, dtype=np.int64)
        self.target_length = len(target_sizes)
        self.target_range = range(self.target_length)
        #pdb.set_trace()
        self.sizes = np.concatenate([self.source_sizes, self.target_sizes])
        domain_index = np.concatenate([np.zeros(self.source_length, dtype=int), np.ones(self.target_length, dtype=int)])
        fnames = []
        fnames.extend(self.source_fnames)
        fnames.extend(self.target_fnames)
        self.domain_index = domain_index
        self.fnames = np.array(fnames).tolist()
        #self.shuffle_index = [*range(self.source_length+self.target_length)]
        #random.shuffle(self.shuffle_index)
        #self.domain_index = domain_index[self.shuffle_index]
        #self.fnames = np.array(fnames)[self.shuffle_index].tolist()
        #self.sizes = self.source_sizes
        #self.fnames = self.source_fnames
        try:
            import pyarrow

            self.source_fnames = pyarrow.array(self.source_fnames)
            self.target_fnames = pyarrow.array(self.target_fnames)
            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)
        #self.base = 10000
        self.base = 4000
        #source_labels = []
        #source_clusters = {"{0}".format(i):[] for i in range(self.num_clusters)}        
        #source_km_path = manifest_base+"_source.km"
        #with open(source_km_path, "r") as f:
        #    for i, label in enumerate(f):
        #        label = label.split("\n")[0]
        #        source_labels.append(label)
        #        source_clusters["{0}".format(label)].append(i)
        #with open("/home/wupeng/fairseq/sample_stat/source.tsv", "w") as fw:
        #    for i in range(self.num_clusters):
        #        fw.write("{0}:{1}\n".format(i, len(source_clusters["{0}".format(i)])))

        #target_labels = []
        #target_clusters = {"{0}".format(i):[] for i in range(self.num_clusters)}        
        #target_km_path = manifest_base+"_target.km"
        #with open(target_km_path, "r") as f:
        #    for i, label in enumerate(f):
        #        label = label.split("\n")[0]
        #        target_labels.append(label)
        #        target_clusters["{0}".format(label)].append(i)
        #with open("/home/wupeng/fairseq/sample_stat/target.tsv", "w") as fw:
        #    for i in range(self.num_clusters):
        #        fw.write("{0}:{1}\n".format(i, len(target_clusters["{0}".format(i)])))
        #pdb.set_trace()

    def __getitem__(self, index):
        import soundfile as sf
        #pdb.set_trace()
        index = 300
        domain_index = self.domain_index[index]
        fn = self.fnames[index]  
        #print(fn)
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        fn = self.text_compressor.decompress(fn)
        root_dir = self.source_root_dir if domain_index == 0 else self.target_root_dir
        path_or_fp = os.path.join(root_dir, fn)
        _path, slice_ptr = parse_path(path_or_fp)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)

        wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")

        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
       
        target_domain_index = 0 if domain_index == 1 else 1
        if target_domain_index == 0:
            domain_range = self.source_range
            tgt_index = [index % self.base]
        else:
            domain_range = self.target_range
            proffix = np.random.choice(domain_range, 1, replace=True)[0] // self.base
            tgt_index = [proffix * self.base + index]
        #print(self.source_range)
        #print(self.target_range)
        #print(index)
        #print(tgt_index)
        #pdb.set_trace()
        tgt_fn = self.source_fnames[tgt_index[0]] if target_domain_index == 0 else self.target_fnames[tgt_index[0]]
        #print(tgt_fn)
        #pdb.set_trace()
        #print(tgt_index[0])
        tgt_fn = tgt_fn if isinstance(self.target_fnames, list) else tgt_fn.as_py()
        tgt_fn = self.text_compressor.decompress(tgt_fn)
        tgt_root_dir = self.source_root_dir if target_domain_index == 0 else self.target_root_dir
        tgt_path_or_fp = os.path.join(tgt_root_dir, tgt_fn)
        tgt_path, tgt_slice_ptr = parse_path(tgt_path_or_fp)
        if len(tgt_slice_ptr) == 2:
            tgt_byte_data = read_from_stored_zip(tgt_path, tgt_slice_ptr[0], tgt_slice_ptr[1])
            assert is_sf_audio_data(tgt_byte_data)
            tgt_path_or_fp = io.BytesIO(tgt_byte_data)
        #print(tgt_path_or_fp)
        tgt_wav, tgt_curr_sample_rate = sf.read(tgt_path_or_fp, dtype="float32")

        tgt_feats = torch.from_numpy(tgt_wav).float()
        tgt_feats = self.postprocess(tgt_feats, tgt_curr_sample_rate)
        return {"id": index, "source": feats, "target": tgt_feats}

        #fn = self.fnames[index]  
        ##print(index)
        #fn = fn if isinstance(self.fnames, list) else fn.as_py()
        #fn = self.text_compressor.decompress(fn)
        #root_dir = self.source_root_dir 
        #path_or_fp = os.path.join(root_dir, fn)
        #_path, slice_ptr = parse_path(path_or_fp)
        #if len(slice_ptr) == 2:
        #    byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
        #    assert is_sf_audio_data(byte_data)
        #    path_or_fp = io.BytesIO(byte_data)
        ##print(path_or_fp)
        #wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")

        #feats = torch.from_numpy(wav).float()
        #feats = self.postprocess(feats, curr_sample_rate)

        #tgt_index = np.random.choice(self.target_range, 1, replace=True)
        #tgt_fn = self.target_fnames[tgt_index[0]]
        #tgt_fn = tgt_fn if isinstance(self.target_fnames, list) else tgt_fn.as_py()
        #tgt_fn = self.text_compressor.decompress(tgt_fn)
        #tgt_root_dir = self.target_root_dir
        #tgt_path_or_fp = os.path.join(tgt_root_dir, tgt_fn)
        #tgt_path, tgt_slice_ptr = parse_path(tgt_path_or_fp)
        #if len(tgt_slice_ptr) == 2:
        #    tgt_byte_data = read_from_stored_zip(tgt_path, tgt_slice_ptr[0], tgt_slice_ptr[1])
        #    assert is_sf_audio_data(tgt_byte_data)
        #    tgt_path_or_fp = io.BytesIO(tgt_byte_data)
        ##print(tgt_path_or_fp)
        #tgt_wav, tgt_curr_sample_rate = sf.read(tgt_path_or_fp, dtype="float32")

        #tgt_feats = torch.from_numpy(tgt_wav).float()
        #tgt_feats = self.postprocess(tgt_feats, tgt_curr_sample_rate)

        #return {"id": index, "source": feats, "target": tgt_feats}



class BinarizedAudioDatasetPair(RawAudioDatasetPair):
    def __init__(
        self,
        data_dir,
        split,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )

        from fairseq.data import data_utils, Dictionary

        self.fnames_dict = Dictionary.load(os.path.join(data_dir, "dict.txt"))

        root_path = os.path.join(data_dir, f"{split}.root")
        if os.path.exists(root_path):
            with open(root_path, "r") as f:
                self.root_dir = next(f).strip()
        else:
            self.root_dir = None

        fnames_path = os.path.join(data_dir, split)
        self.fnames = data_utils.load_indexed_dataset(fnames_path, self.fnames_dict)
        lengths_path = os.path.join(data_dir, f"{split}.lengths")

        with open(lengths_path, "r") as f:
            for line in f:
                sz = int(line.rstrip())
                assert (
                    sz >= min_sample_size
                ), f"Min sample size is not supported for binarized dataset, but found a sample with size {sz}"
                self.sizes.append(sz)

        self.sizes = np.array(self.sizes, dtype=np.int64)

        self.set_bucket_info(num_buckets)
        logger.info(f"loaded {len(self.fnames)} samples")

    def __getitem__(self, index):
        import soundfile as sf

        fname = self.fnames_dict.string(self.fnames[index], separator="")
        if self.root_dir:
            fname = os.path.join(self.root_dir, fname)

        wav, curr_sample_rate = sf.read(fname)
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        return {"id": index, "source": feats}
