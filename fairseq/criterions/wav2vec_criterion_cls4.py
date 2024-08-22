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


@register_criterion("wav2vec_mmd_cls4", dataclass=Wav2VecCriterionConfig)
class Wav2vecCriterion(FairseqCriterion):
    def __init__(self, task, infonce=False, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.infonce = infonce
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys
        self.mmd_loss = MMD_loss(kernel_type='linear')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_output, tgt_output, src_xvec, tgt_xvec  = model(**sample["net_input"])

        mmd_loss_f = 0.0

        src_logits, src_target, src_cpc_loss, src_weighted_xvec = self.forward_logits(model, src_output, sample, src_xvec)
        tgt_logits, tgt_target, tgt_cpc_loss, tgt_weighted_xvec = self.forward_logits(model, tgt_output, sample, tgt_xvec)
        mmd_loss_u = self.mmd_loss(src_weighted_xvec, tgt_weighted_xvec)/2        
        #print(mmd_loss_u)
        #mmd_loss_u = 0.0

        src_loss, src_sample_size, src_logging_output = self.forward_cpc(model, src_output, sample, src_logits, src_target, src_cpc_loss, mmd_loss_u, mmd_loss_f, domain="src")
        tgt_loss, tgt_sample_size, tgt_logging_output = self.forward_cpc(model, tgt_output, sample, tgt_logits, tgt_target, tgt_cpc_loss, mmd_loss_u, mmd_loss_f, domain="tgt")
        loss = src_loss + tgt_loss
        sample_size = src_sample_size + tgt_sample_size
        logging_output = {}
        logging_output["correct"] = src_logging_output["src_correct"] + tgt_logging_output["tgt_correct"]
        logging_output["count"] = src_logging_output["src_count"] + tgt_logging_output["tgt_count"]
        logging_output["loss"] = src_logging_output["src_loss"] + tgt_logging_output["tgt_loss"]
        logging_output["ntokens"] = src_logging_output["src_ntokens"] + tgt_logging_output["tgt_ntokens"]
        logging_output["nsentences"] = src_logging_output["src_nsentences"] + tgt_logging_output["tgt_nsentences"]
        logging_output["sample_size"] = src_logging_output["src_sample_size"] + tgt_logging_output["tgt_sample_size"]
        logging_output.update(src_logging_output)
        logging_output.update(tgt_logging_output)
        #print(logging_output)
        #pdb.set_trace()
        return loss, sample_size, logging_output

    def forward_logits(self, model, net_output, sample, xvec, reduce=True):
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output)

        clogits = model.get_clogits(net_output).float()
        ctarget = model.get_ctargets(sample, net_output)

        nsz, bsz, tsz = net_output["x"].size()
        self.xla = is_xla_tensor(logits)
        # XXX: handle weights on xla.
        weights = None
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []

        reduction = "none" if ((not reduce) or self.xla) else "sum"
        #pdb.set_trace()
        if self.infonce:
            loss = F.cross_entropy(logits, target, reduction=reduction)
            utt_nce_loss = F.cross_entropy(clogits, ctarget, reduction="none").view(-1, bsz)
            utt_nce_loss = torch.mean(utt_nce_loss, axis=0)
            pc = torch.exp(-utt_nce_loss)
            #print(pc)
            weighted_xvec = torch.mul(pc.unsqueeze(-1), xvec)
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, target.float(), weights, reduction=reduction
            )
        return [logits, target, loss, weighted_xvec]

    def forward_cpc(self, model, net_output, sample, logits, target, loss, mmd_loss_u, mmd_loss_f, domain="src", reduce=True):
        #pdb.set_trace()
        losses = []

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
            sample_size = target.numel() if self.infonce else target.long().sum().item()
        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            extra_losses.append(mmd_loss_u)
            extra_losses.append(mmd_loss_f)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                #pdb.set_trace()
                if coef != 0 and p is not None:
                    #print(coef)
                    #print(p.float())
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        logging_output = {
            "{0}_loss".format(domain): loss.item() if (reduce and not self.xla) else loss.detach(),
            "{0}_ntokens".format(domain): sample_size,
            "{0}_nsentences".format(domain): sample["id"].numel(),
            "{0}_sample_size".format(domain): sample_size,
        }
        #print(logging_output)
        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits"] = logits.cpu().numpy()
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
                if domain == "src":
                    logging_output[f"src_loss_{i}"] = l.item() if not self.xla else l.detach()
                else:
                    logging_output[f"tgt_loss_{i}"] = l.item() if not self.xla else l.detach()

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

                logging_output["{0}_correct".format(domain)] = corr
                logging_output["{0}_count".format(domain)] = count

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

class MMD_loss(torch.nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            n = int(source.size()[0])
            m = int(target.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = kernels[:n, :n]
            YY = kernels[n:, n:]
            XY = kernels[:n, n:]
            YX = kernels[n:, :n]
            XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)
            XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1)
            YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1)
            YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)

            loss = (XX + XY).sum() + (YX + YY).sum()
            return loss
