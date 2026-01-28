from typing import TYPE_CHECKING, List, Optional


from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import cal_effective_tokens, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

import torch
import torch.nn as nn
from .eval_accuracy import eval_accuracy

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ...hparams import FinetuningArguments, ModelArguments

logger = logging.get_logger(__name__)

class InferenceModel(nn.Module):
    def __init__(self, 
                 data_args: "DataArguments",
                 model_args: "ModelArguments", 
                 training_args: "Seq2SeqTrainingArguments", 
                 finetuning_args: "FinetuningArguments", 
                 generating_args: "GeneratingArguments",
                tokenizer_module,
                template,
                model
                ):
        super().__init__()
        self.data_args = data_args
        self.training_args = training_args
        self.finetuning_args = finetuning_args
        self.model_args = model_args
        self.generating_args = generating_args
        self.template = template

        self.tokenizer_module = tokenizer_module
        self.tokenizer = self.tokenizer_module["tokenizer"]
        
        self.model = model
        
        self.base_output_dir = self.training_args.output_dir
        
    
    def reset_trainer(self, train_dataset, **kwargs):
        data_collator = SFTDataCollatorWith4DAttentionMask(
            template=self.template,
            # pad_to_multiple_of=8 if self.training_args.do_train else None,  # for shift short attention
            pad_to_multiple_of= None,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            **self.tokenizer_module,
        )

        self.trainer = CustomSeq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            finetuning_args=self.finetuning_args,
            model_args=self.model_args,
            data_args=self.data_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            **self.tokenizer_module,
            **kwargs
        )
    
    def unwrap_model(self):
        self.model = self.trainer.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False)
    

    def forward(self, predict_batch):  
        """
        Predict.
        """


        self.tokenizer.padding_side = "right"  # use right-padding in training
        self.training_args.generation_max_length = self.training_args.generation_max_length or self.data_args.cutoff_len
        self.training_args.generation_num_beams = self.data_args.eval_num_beams or self.training_args.generation_num_beams
        self.training_args.remove_unused_columns = False  # important for multimodal dataset
        self.reset_trainer(train_dataset=None)

        self.trainer.save_model()

        self.unwrap_model()



        # predict
        gen_kwargs = self.generating_args.to_dict()
        gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()
        # decoder-only models must use left-padding for batched generation.
        if self.training_args.predict_with_generate:
            self.tokenizer.padding_side = "left"  # use left-padding in generation
        self.training_args.output_dir = self.base_output_dir + f'/predict-temperature_{self.generating_args.temperature}-max_new_tokens_{self.generating_args.max_new_tokens}'
        
        self.reset_trainer(train_dataset=None)
        predict_results = self.trainer.predict(predict_batch, metric_key_prefix="predict", **gen_kwargs)
        self.trainer.save_predictions(predict_batch, predict_results)
    

    
def run_inferenceonly(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ttl", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    

    eval_dataset = dataset_module['eval_dataset']
    print("eval_dataset")
    print(eval_dataset)
    print("eval_dataset[0]")
    print( eval_dataset[0])
    

    inferenceonly_model = InferenceModel(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        tokenizer_module=tokenizer_module,
        template=template, 
        model=model
    )
    

    inferenceonly_model.forward(predict_batch=eval_dataset)

    path = training_args.output_dir + "/generated_predictions.jsonl"
    print("Evaluating accuracy based on predictions saved at:", path)
    paths = [path]
    eval_accuracy(paths) 
