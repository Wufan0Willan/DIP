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
from typing import Any, List, Optional, Union
import copy
import torchaudio

logger = logging.getLogger(__name__)


class RawAudioDatasetSepNoise(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        compute_mask_indices=False,
        mixing_max_len: int = -1,
        #mixing_prob: float = 1.0,
        mixing_prob: float = 0.0,
        mixing_num: int = 2,
        mixing_noise: bool = False,
        mixing_noise_prob: float = 0.0,
        mixing_noise_num: int = 1,
        noise_path: Optional[str] = None,
        random_crop: bool = False,
        single_target: bool = False,
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

        self.single_target = single_target
        self.random_crop = random_crop
        self.mixing_max_len = mixing_max_len
        self.mixing_prob = mixing_prob
        #self.mixing_prob = 0.0
        self.mixing_num = mixing_num
        self.mixing_noise = mixing_noise
        self.mixing_noise_prob = mixing_noise_prob
        self.mixing_noise_num = mixing_noise_num
        self.noise_path = noise_path
        print("mixing_max_len:{0}".format(self.mixing_max_len))
        print("mixing_prob:{0}".format(self.mixing_prob))
        #self.audio_aug = "speed_pertub"
        self.audio_aug = None
        if self.audio_aug:
            print("audio augment:{0}".format(self.audio_aug))
        else:
            print("No audio augment applied")
            #print("include clean")

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
            print("normaled")
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

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

        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        #########################################
        #sources = []
        #for s in samples:
        #    mix,s1,s2 = s["source"].reshape(3,-1)
        #    sources.extend([mix,s1,s2])
        sizes = [len(s) for s in sources]
        #print(sizes)
        #if self.audio_aug == "speed_pertub":
        #    audio_sources = torch.stack(sources, axis=0)
        #    print(audio_sources.size())
        #    effects = [
        #        ["speed", f"{(np.random.rand()/10+0.95):1.2f}"],
        #        ["rate", f"{16000}"],
        #    ]
        #    audio, sr = torchaudio.sox_effects.apply_effects_tensor(audio_sources, 16000, effects)
        #    sources = [audio[i] for i in range(len(audio))]   
        #    
        ##########################################
        #sizes = [len(s) for s in sources]
        #print("***************size*******************")
        #print(sizes)
        #print("***************end*******************")
        #print("***************sources*******************")
        #print(sources) 
        #print("***************mix*******************")
        #print(mix)
        #print("***************s1*******************")
        #print(s1)
        #print("***************s2*******************")
        #print(s2)
        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)
        collated_sources, padding_mask, audio_starts = self.collater_audio(
            sources, target_size
        )
        #print(collated_sources)
        #if self.audio_aug == "speed_pertub":
        #    audio_sources = collated_sources
        #    print(audio_sources.size())
        #    effects = [
        #        ["speed", f"{(np.random.rand()/10+0.95):1.2f}"],
        #        ["rate", f"{16000}"],
        #    ]
        #    collated_sources, sr = torchaudio.sox_effects.apply_effects_tensor(audio_sources, 16000, effects)
        #new_sizes = collated_sources.size()
        #print("***************size*******************")
        #print(new_sizes)
        collated_mix_audios = self.mixing_collated_audios(collated_sources, target_size)
        #print("***************collated sources*******************")
        #print(collated_sources)
        input = {"source": collated_mix_audios}
        #input = {"source": mix_list}

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
        #print("***************out*******************")
        #print(out)
        out["net_input"] = input
        return out

    def mixing_collated_audios(self, source_list, audio_size):
        # mixing utterance or noise within the current batch
        source_audio = copy.deepcopy(source_list)
        source = copy.deepcopy(source_list)
        mix_list = []
        for i in range(self.mixing_num):
            mix_list.append(source_list[0].new_zeros(len(source_list), audio_size)) 
        B = source.shape[0]
        T = source.shape[1]
        #mixing_max_len = T // 2 if self.mixing_max_len < 0 else T // self.mixing_max_len
        mixing_max_len = T if self.mixing_max_len < 0 else T // self.mixing_max_len
        mixing_max_len = T if mixing_max_len > T else mixing_max_len
        for i in range(B):
            if self.mixing_noise and np.random.random() < self.mixing_noise_prob:
                # mixing with noise
                choices = np.random.choice(self.noise_list, self.mixing_noise_num)
                for c in choices:
                    path, key, start, end = c["loc"].split("\t")
                    if path not in self.noise_container:
                        self.noise_container[path] = h5py.File(path, "r")["wav"]
                    noise = self.noise_container[path][int(start): int(end)]
                    noise = noise.astype(np.float32) / np.iinfo(np.int16).max

                    ref_pow = np.mean(source[i].numpy() ** 2)
                    noise_pow = np.mean(noise ** 2)
                    if noise_pow == 0:
                        scale = 0
                    else:
                        snr = np.random.uniform(-5, 20)
                        scale = (ref_pow / (noise_pow * 10 ** (snr / 10))) ** 0.5
                    noise = scale * noise
                    noise = torch.from_numpy(noise).type_as(source)

                    c_len = np.random.randint(0, mixing_max_len + 1)
                    c_len = min(c_len, noise.shape[0])

                    c_end = np.random.randint(c_len, noise.shape[0] + 1)
                    c_start = c_end - c_len
                    s_end = np.random.randint(c_len, T + 1)
                    s_start = s_end - c_len

                    source[i, s_start:s_end] += noise[c_start:c_end]

            else:
                # mixing with utterance
                index_list = [j for j in range(B) if j!=i]
                #index_list = [j for j in range(B)]
                choices = np.random.choice(index_list, self.mixing_num, replace=False)
                clean_center = False
                for k,c in enumerate(choices):
                    c_len = T
                    c_end = np.random.randint(c_len, T + 1)
                    c_start = c_end - c_len
                    s_end = np.random.randint(c_len, T + 1)
                    s_start = s_end - c_len
                    ref_pow = np.mean(source_audio[i].numpy() ** 2)
                    noise_pow = np.mean(source_audio[c].numpy() ** 2)
                    rnd = np.random.random()
                    if noise_pow == 0:
                        scale = 0
                    elif (rnd < self.mixing_prob and clean_center==False):
                        scale = 0
                        clean_center = True
                    else:
                        snr = np.random.uniform(-5, 5)
                        scale = (ref_pow / (noise_pow * 10 ** (snr / 10))) ** 0.5
                    mix_list[k][i,  c_start:c_end] = source[i, s_start:s_end] + source_audio[c, c_start:c_end].clone() * scale
                    #mix_list[k][i,  c_start:c_end] = source[i, s_start:s_end] + source_audio[c, c_start:c_end].clone()
            if self.normalize:
                print("norm")
                with torch.no_grad():
                    source[i] = F.layer_norm(source[i], source[i].shape)
        #print(source)
        #print(source_audio)
        #print(mix_list)
        #print("***********************")
        return mix_list

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            if self.audio_aug == "speed_pertub":
                effects = [
                    ["speed", f"{(np.random.rand()/20+0.95):1.2f}"],
                    ["rate", f"{16000}"],
                ]
                audio, sr = torchaudio.sox_effects.apply_effects_tensor(audio.unsqueeze(0), 16000, effects)
                audio = audio.squeeze(0)
            diff = len(audio) - audio_size
            #print(diff)
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )

        return collated_audios, padding_mask, audio_starts

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


class FileAudioDatasetSepNoise(RawAudioDatasetSepNoise):
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

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(self.text_compressor.compress(items[0]))
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)

    def __getitem__(self, index):
        import soundfile as sf

        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        fn = self.text_compressor.decompress(fn)
        path_or_fp = os.path.join(self.root_dir, fn)
        _path, slice_ptr = parse_path(path_or_fp)
        #print("***************path*******************")
        #print(_path)
        #print(slice_ptr)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)
        #print(path_or_fp)
        wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")

        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        #print("***************feats*******************")
        #print(feats)
        #print(feats.shape)
        return {"id": index, "source": feats}


class BinarizedAudioDataset(RawAudioDatasetSepNoise):
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
