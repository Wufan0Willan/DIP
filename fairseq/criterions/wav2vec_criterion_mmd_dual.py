# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round
from fairseq.utils import is_xla_tensor
import pdb

@dataclass
class Wav2VecCriterionConfig(FairseqDataclass):
    infonce: bool = field(
        default=False,
        metadata={
            "help": "if set, uses cross entropy instead of binary cross entropy (i.e. InfoNCE loss)"
        },
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )


@register_criterion("wav2vec_mmd_dual", dataclass=Wav2VecCriterionConfig)
class Wav2vecCriterion(FairseqCriterion):
    def __init__(self, task, infonce=False, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.infonce = infonce
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #pdb.set_trace()
        #for name, param in model.named_parameters():
        #    if name.startswith("feature_extractor"):
        #        param.requires_grad = False
        src_output, tgt_output, da_output = model(**sample["net_input"])
        src_logits = model.get_logits(src_output).float()
        src_target = model.get_targets(sample, src_output)
        tgt_logits = model.get_logits(tgt_output).float()
        tgt_target = model.get_targets(sample, tgt_output)
        self.xla = is_xla_tensor(src_logits)
        pdb.set_trace()
        # XXX: handle weights on xla.
        weights = None
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []

        reduction = "none" if ((not reduce) or self.xla) else "sum"
        if self.infonce:
            src_loss = F.cross_entropy(src_logits, src_target, reduction=reduction)
        else:
            src_loss = F.binary_cross_entropy_with_logits(
                src_logits, src_target.float(), weights, reduction=reduction
            )

        if self.infonce:
            tgt_loss = F.cross_entropy(tgt_logits, tgt_target, reduction=reduction)
        else:
            tgt_loss = F.binary_cross_entropy_with_logits(
                tgt_logits, tgt_target.float(), weights, reduction=reduction
            )

        loss = src_loss + tgt_loss

        if self.xla:
            # tpu-comment: since dynamic shapes lead to recompilations on xla,
            # we don't shrink tensors using mask_indices.
            # Instead, we use mask indices to adjust loss.
            mi = (
                sample["net_input"]["mask_indices"]
                .transpose(0, 1)  # logits are transposed in `model.get_logits`
                .reshape(logits.size(0))
            )
            loss = (loss * mi).sum() if reduce else (loss * mi)

        if "sample_size" in sample:
            sample_size = sample["sample_size"]
        elif "mask_indices" in sample["net_input"]:
            sample_size = sample["net_input"]["mask_indices"].sum()
        else:
            src_sample_size = src_target.numel() if self.infonce else src_target.long().sum().item()
            tgt_sample_size = tgt_target.numel() if self.infonce else src_target.long().sum().item()

        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            src_extra_losses = model.get_extra_losses(src_output)
            tgt_extra_losses = model.get_extra_losses(tgt_output)
            extra_losses = [[src_extra_losses,tgt_extra_losses]]
            extra_losses.append(da_output["mmd_mean_loss"])
            extra_losses.append(da_output["mmd_var_loss"])
            #pdb.set_trace()
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    if isinstance(p, list):
                        p = coef * (p[0].float() * src_sample_size + p[1].float() * tgt_sample_size)
                    else:
                        p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        logging_output = {
            "loss": loss.item() if (reduce and not self.xla) else loss.detach(),
            "mmd_loss": da_output["mmd_loss"].item() if (reduce and not self.xla) else da_output["mmd_loss"].detach(),
            "coral_loss": da_output["coral_loss"].item() if (reduce and not self.xla) else da_output["coral_loss"].detach(),
            "ntokens": src_sample_size + tgt_sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": src_sample_size + tgt_sample_size,
        }
        pdb.set_trace()
        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["src_logits"] = src_logits.cpu().numpy()
                    logging_output["tgt_logits"] = tgt_logits.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    # If the targets have been mixed with the predictions of
                    # teacher models, find the original targets
                    if hasattr(model, "get_original_targets"):
                        original_target = model.get_original_targets(sample, net_output)
                    else:
                        original_target = target
                    logging_output["target"] = original_target.cpu().numpy()
            elif lk in net_output:
                value = net_output[lk]
                if not is_xla_tensor(value):
                    value = float(value)
                logging_output[lk] = value

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item() if not self.xla else l.detach()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    if is_xla_tensor(logits):
                        max, min = max * mi, min * mi
                        both = max & min
                        corr = max.long().sum() - both.long().sum()
                        count = mi.sum()
                    else:
                        both = max & min
                        corr = max.long().sum().item() - both.long().sum().item()
                        count = float(max.numel())

                logging_output["correct"] = corr
                logging_output["count"] = count

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(
                    meters["_correct"].sum / meters["_total"].sum, 5
                )
                if meters["_total"].sum > 0
                else float("nan"),
            )

        builtin_keys = {
            "loss",
            "ntokens",
            "nsentences",
            "sample_size",
            "correct",
            "count",
        }

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs)
                if k.startswith("loss"):
                    metrics.log_scalar(
                        k, val / (sample_size or 1) / math.log(2), sample_size, round=3
                    )
                else:
                    metrics.log_scalar(k, val / len(logging_outputs), round=3)

    # FIXME: revert when gather based xla reduction is implemented
    # @staticmethod
    # def logging_outputs_can_be_summed() -> bool:
    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        # XXX: Gather based reduction not implemented for xla yet.
        # So we fall to sum based reduction for xla.
        return self.xla
