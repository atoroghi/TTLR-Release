# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING: 
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

from transformers import AutoModelForCausalLM

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], model_args=None, data_args=None, pretrain_model=None, **kwargs   
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.model_args = model_args
        self.data_args = data_args
        self.pretrain_model = pretrain_model

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.special_token_ids = set()
        for tok_id in [
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.unk_token_id,
        ]:
            if tok_id is not None:
                self.special_token_ids.add(tok_id)
        self.special_token_ids = list(self.special_token_ids)


    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "a", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res)+"\n")


    def log_to_file(self, sentence_ce, sentence_kl, mask, coeff, loss):
        """
        Log the sentence cross-entropy and KL divergence to a file.
        """
        with open(f"{self.args.output_dir}/logfile.txt", 'a', encoding="utf-8") as f:
            for ce, kl, m, coef in zip(sentence_ce.clone().detach(), sentence_kl.clone().detach(), mask, coeff):
                if m:
                    print(f"This sample is selected. Threshold: {self.finetuning_args.threshold}, Cross-entropy: {ce}, KL divergence: {kl}, Weight coefficient: {coef}, Final loss: {loss}", file=f)
                else:
                    print(f"This sample is discarded. Threshold: {self.finetuning_args.threshold}, Cross-entropy: {ce}, KL divergence: {kl}", file=f)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        assert self.tokenizer.padding_side == 'right', "Training should be done with right padding."

        # print("IN COMPUTE LOSS FUNCTION:")
        # batch_size = inputs["input_ids"].size(0)
        # print("Batch size:", batch_size)

        # for i in range(batch_size):
        #     print(
        #         f"Decoded input {i}:",
        #         self.tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=False)
        #     )


        # Use Tent entropy-generation loss for training. The function autoregressively
        # generates `max_new_tokens` and returns an entropy-based loss.
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("`input_ids` must be provided in `inputs` for Tent loss.")


        loss, _generated_ids = self.tent_generation_entropy_loss(model, input_ids)
        total_loss = loss

        if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
            if return_outputs:
                return (total_loss / self.args.gradient_accumulation_steps, None)
            else:
                return total_loss / self.args.gradient_accumulation_steps

        return (total_loss, None) if return_outputs else total_loss
        

    def softmax_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Entropy of softmax distribution from logits.
        Args:
            logits: (batch_size, vocab_size)
        Returns:
            entropy: (batch_size,)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1)




    @torch.enable_grad()
    def tent_generation_entropy_loss(
        self,
        model,
        input_ids: torch.Tensor,
        max_new_tokens: int = 80,
    ):
        """
        Autoregressively generate tokens and compute TENT entropy loss
        with DETACHED transformer backbone.
        """

        batch_size = input_ids.size(0)
        device = input_ids.device

        generated_ids = input_ids
        total_entropy = 0.0
        token_count = 0.0

        eos_id = self.tokenizer.eos_token_id
        alive_mask = torch.ones(batch_size, device=device)

        for _ in range(max_new_tokens):

            # --------------------------------------------------
            # 1) Transformer forward WITHOUT autograd
            # --------------------------------------------------
            with torch.no_grad():
                outputs = model.model(
                    input_ids=generated_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[-1]  # (B, T, H)

            # --------------------------------------------------
            # 2) Detach and re-enable gradients
            # --------------------------------------------------
            hidden = hidden.detach()
            hidden.requires_grad_(True)

            # --------------------------------------------------
            # 3) Forward through lm_head (+ norms)
            # --------------------------------------------------
            logits = model.lm_head(hidden[:, -1, :])  # (B, vocab)

            # --------------------------------------------------
            # 4) Greedy decoding (NO grad needed)
            # --------------------------------------------------
            with torch.no_grad():
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                if eos_id is not None:
                    alive_mask = alive_mask * (next_token.squeeze(-1) != eos_id).float()

            # --------------------------------------------------
            # 5) Mask special tokens FOR LOSS ONLY
            # --------------------------------------------------
            if len(self.special_token_ids) > 0:
                logits[:, self.special_token_ids] = -float("inf")

            # --------------------------------------------------
            # 6) Entropy loss (THIS is what backpropagates)
            # --------------------------------------------------
            entropy_t = self.softmax_entropy(logits)  # (B,)

            total_entropy += entropy_t * alive_mask
            token_count += alive_mask.sum()

            if alive_mask.sum() == 0:
                break

        loss = total_entropy.sum() / token_count.clamp(min=1.0)
        return loss, generated_ids


    # @torch.enable_grad()
    # def tent_generation_entropy_loss(
    #     self,
    #     model,
    #     input_ids: torch.Tensor,
    #     max_new_tokens: int = 80,
    # ):
    #     """
    #     Autoregressively generate tokens and compute TENT entropy loss.
    #     """

    #     batch_size = input_ids.size(0)
    #     device = input_ids.device

    #     generated_ids = input_ids
    #     total_entropy = 0.0
    #     token_count = 0.0  

    #     eos_id = self.tokenizer.eos_token_id  
    #     alive_mask = torch.ones(batch_size, device=device)  #(1 = still active)

    #     for _ in range(max_new_tokens):
    #         outputs = model(input_ids=generated_ids)
    #         logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)

    #         # greedy decoding
    #         next_token = logits.argmax(dim=-1, keepdim=True)  # (batch_size, 1)
    #         generated_ids = torch.cat([generated_ids, next_token], dim=1)

    #         # update alive mask (samples that just produced EOS become inactive)
    #         if eos_id is not None: 
    #             alive_mask = alive_mask * (next_token.squeeze(-1) != eos_id).float() 

    #         # mask special tokens for LOSS ONLY
    #         if len(self.special_token_ids) > 0:
    #             logits[:, self.special_token_ids] = -float("inf")

    #         # entropy for current token
    #         entropy_t = self.softmax_entropy(logits)  # (batch_size,)

    #         # accumulate entropy only for alive samples
    #         total_entropy += entropy_t * alive_mask  
    #         token_count += alive_mask.sum() 

    #         # optional early stop if all samples finished
    #         if alive_mask.sum() == 0:  
    #             break

    #     # normalize by number of generated tokens
    #     loss = total_entropy.sum() / token_count 

    #     return loss, generated_ids
