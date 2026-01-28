from typing import TYPE_CHECKING, List, Optional

import json


from ..sft.workflow import run_sft
from ...extras import logging

from .eval_accuracy import eval_accuracy

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)



def load_augmented_dataset_from_file(file_path: str):
    r"""
    Load an augmented dataset from a JSON file to use the 'output' field
    for supervised fine-tuning.
    
    Args:
        file_path: Path to the JSON file containing the augmented dataset
        
    Returns:
        A datasets.Dataset object with samples ready for SFT using the 'output' field
    """
    try:
        with open(file_path, 'r') as f:
            augmented_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Augmented dataset file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {file_path}")
    
    # Import Dataset here to avoid circular imports
    from datasets import Dataset
    
    # Create a dataset from the loaded data
    augmented_dataset = Dataset.from_dict({
        key: [sample.get(key) for sample in augmented_data]
        for key in augmented_data[0].keys()
    })
    
    # The 'output' field in the augmented dataset will be used as the target for SFT
    logger.info(f"Loaded augmented dataset with {len(augmented_dataset)} samples from {file_path}")
    
    return augmented_dataset



def run_ttlr(model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):

    # Load the augmented dataset from the path
    # dataset_path = 'augmented_dataset_0_1000.json'
    # augmented_data = load_augmented_dataset_from_file(dataset_path)
    # logger.info(f"Augmented data sample: {augmented_data[0]}")
        
    run_sft(model_args, data_args, training_args, finetuning_args, 
            generating_args, callbacks)
    
    path = training_args.output_dir + "/generated_predictions.jsonl"
    print("Evaluating accuracy based on predictions saved at:", path)
    paths = [path]
    eval_accuracy(paths)
       








############ Previos code// till end//############
# class TTLRModel(nn.Module):

#     def __init__(self, 
#                  data_args: "DataArguments",
#                  model_args: "ModelArguments", 
#                  training_args: "Seq2SeqTrainingArguments", 
#                  finetuning_args: "FinetuningArguments", 
#                  generating_args: "GeneratingArguments",
#                 tokenizer_module,
#                 template,
#                 model
#                 ):
#         super().__init__()
#         self.data_args = data_args
#         self.training_args = training_args
#         self.finetuning_args = finetuning_args
#         self.model_args = model_args
#         self.generating_args = generating_args
#         self.template = template

#         self.tokenizer_module = tokenizer_module
#         self.tokenizer = self.tokenizer_module["tokenizer"]
        
#         self.model = model
        
#         self.base_output_dir = self.training_args.output_dir
        
    
#     def reset_trainer(self, train_dataset, **kwargs):
#         data_collator = SFTDataCollatorWith4DAttentionMask(
#             template=self.template,
#             # pad_to_multiple_of=8 if self.training_args.do_train else None,  # for shift short attention
#             pad_to_multiple_of= None,  # for shift short attention
#             label_pad_token_id=IGNORE_INDEX if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
#             block_diag_attn=self.model_args.block_diag_attn,
#             attn_implementation=getattr(self.model.config, "_attn_implementation", None),
#             compute_dtype=self.model_args.compute_dtype,
#             **self.tokenizer_module,
#         )

#         self.trainer = build_sft_trainer(
#             model=self.model,
#             tokenizer=self.tokenizer,
#             training_args=self.training_args,
#             train_dataset=train_dataset,
#             eval_dataset=kwargs.get("eval_dataset", None),
#             data_collator=data_collator,
#         )

    
#     def forward(self, train_batch, predict_batch):
#         if self.finetuning_args.setting == "offline_ttlr":
#             self.forward_cot_sft_offline(train_batch=train_batch)
#             self.generate_and_save_predictions(predict_batch=predict_batch)

        
#         elif self.finetuning_args.setting == "online_ttlr":
#             pass  # to be implemented

#         else:
#             raise ValueError(
#                 f'NO such setting: {self.finetuning_args.setting}'
#             )

#     def forward_cot_sft_offline(
#         self,
#         train_batch
#     ):
#         """
#         Perform supervised fine-tuning on the best zero-shot CoT response.
#         """
#         logger.info("Starting zero-shot CoT augmentation and supervised fine-tuning...")
        
#         # Set tokenizer padding side for generation
#         self.tokenizer.padding_side = "left"
#         logger.info(f"Train batch sample: {train_batch[0]}")
        
#         # Load the augmented dataset from the path
#         dataset_path = 'augmented_dataset_0_1000.json'
#         augmented_data = load_augmented_dataset_from_file(dataset_path)
#         logger.info(f"Augmented data sample: {augmented_data[0]}")
        
#         # The augmented_data is a python dictionary with an "instruction" field and an "output" field.
#         # We need to encode the data using the template for proper SFT format
#         def encode_sft_sample(examples):
#             input_ids_list = []
#             labels_list = []

#             for instruction, output in zip(examples["instruction"], examples["output"]):
#                 messages = [
#                     {"role": "user", "content": instruction},
#                     {"role": "assistant", "content": output},
#                 ]

#                 input_ids, labels = self.template.encode_oneturn(
#                     tokenizer=self.tokenizer,
#                     messages=messages,
#                     system=None,
#                     tools=None,
#                 )

#                 input_ids_list.append(input_ids)
#                 labels_list.append(labels)

#             return {"input_ids": input_ids_list, "labels": labels_list}

                
#         # Preprocess the dataset for SFT
#         logger.info("Preprocessing augmented dataset for SFT...")
#         augmented_dataset = augmented_data.map(
#             encode_sft_sample,
#             batched=True,
#             remove_columns=augmented_data.column_names,
#             batch_size=1,
#         )
        
#         logger.info(f"Preprocessed augmented dataset sample: {augmented_dataset[0]}")
#         logger.info(f"Total augmented samples: {len(augmented_dataset)}")
        
#         # Reset the trainer with the augmented dataset
#         logger.info("Resetting trainer with augmented dataset...")
#         self.reset_trainer(augmented_dataset)
        
#         # Train the model
#         logger.info("Starting supervised fine-tuning...")
#         train_result = self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        
#         # Log training metrics
#         if self.finetuning_args.include_effective_tokens_per_second:
#             effective_token_num = sum(len(sample["input_ids"]) for sample in augmented_dataset)
#             train_result.metrics["effective_tokens_per_sec"] = cal_effective_tokens(
#                 effective_token_num, train_result.metrics["epoch"], train_result.metrics["train_runtime"]
#             )
        
#         self.trainer.log_metrics("train", train_result.metrics)
#         self.trainer.save_metrics("train", train_result.metrics)
#         self.trainer.save_state()
        
#         # Save the model
#         logger.info(f"Saving model to {self.training_args.output_dir}...")
#         self.trainer.save_model()
        
#         # Plot loss if requested
#         if self.trainer.is_world_process_zero() and self.finetuning_args.plot_loss:
#             plot_loss(self.training_args.output_dir, keys=["loss", "eval_loss"])
        
#         logger.info("Supervised fine-tuning completed successfully!")



#     def generate_and_save_predictions(
#         self,
#         predict_batch
#     ):
#         """
#         Generate model predictions for all test samples and save to a JSON file.

#         Each entry in the JSON file will have:
#         {
#             "prompt": str,
#             "label": str,
#             "prediction": str
#         }
#         """

#         self.model.eval()
#         self.tokenizer.padding_side = "left"

#         results = []

#         device = self.model.device

#         with torch.no_grad():
#             for sample in tqdm(predict_batch, desc="Generating predictions"):
#                 prompt = sample["instruction"]
#                 label = sample["label"]

#                 # Build messages for inference (NO assistant output)
#                 messages = [
#                     {"role": "user", "content": prompt},
#                 ]

#                 # Encode prompt using the same template
#                 input_ids, _ = self.template.encode_oneturn(
#                     tokenizer=self.tokenizer,
#                     messages=messages,
#                     system=None,
#                     tools=None,
#                 )

#                 input_ids = torch.tensor([input_ids], device=device)

#                 # Generate
#                 generated_ids = self.model.generate(
#                     input_ids=input_ids,
#                     max_new_tokens=self.training_args.max_new_tokens,
#                     do_sample=(self.training_args.temperature > 0),
#                     temperature=self.training_args.temperature,
#                     top_p=self.training_args.top_p,
#                     pad_token_id=self.tokenizer.eos_token_id,
#                 )

#                 # Decode only newly generated tokens
#                 generated_text = self.tokenizer.decode(
#                     generated_ids[0][input_ids.shape[-1]:],
#                     skip_special_tokens=True,
#                 ).strip()

#                 results.append({
#                     "prompt": prompt,
#                     "label": label,
#                     "prediction": generated_text,
#                 })

#         # Save to JSON
#         with open(self.training_args.output_json_path, "w", encoding="utf-8") as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)

#         print(f"Saved {len(results)} predictions to {self.training_args.output_json_path}")



    
    
    
# def run_ttlr(
#     model_args: "ModelArguments",
#     data_args: "DataArguments",
#     training_args: "Seq2SeqTrainingArguments",
#     finetuning_args: "FinetuningArguments",
#     generating_args: "GeneratingArguments",
#     callbacks: Optional[List["TrainerCallback"]] = None,
# ):  
#     print("in run_ttlr")
#     tokenizer_module = load_tokenizer(model_args)
#     tokenizer = tokenizer_module["tokenizer"]
#     template = get_template_and_fix_tokenizer(tokenizer, data_args)
#     dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ttlr", **tokenizer_module)
#     print("loading model")
#     model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
#     print("model loaded")
#     train_dataset = dataset_module['train_dataset']
#     print("train_dataset done")
#     eval_dataset = dataset_module['eval_dataset']
#     print("eval_dataset done")
#     print(train_dataset, eval_dataset)
#     print(train_dataset[0], '\n', eval_dataset[0])

#     ttlr_model = TTLRModel(
#         data_args=data_args,
#         model_args=model_args,
#         training_args=training_args,
#         finetuning_args=finetuning_args,
#         generating_args=generating_args,
#         tokenizer_module=tokenizer_module,
#         template=template, 
#         model=model
#     )
    
#     if finetuning_args.setting == "offline_ttlr":
#         ttlr_model.forward(train_batch=train_dataset, predict_batch=eval_dataset)
#     elif finetuning_args.setting == "online_ttlr":
#         streaming_batch_size = finetuning_args.streaming_batch_size
#         num_of_batch = len(train_dataset) // streaming_batch_size
#         if len(train_dataset) % streaming_batch_size != 0:
#             num_of_batch += 1
#         for k in range(num_of_batch):
#             logger.info_rank0(f"Processing batch {k+1}/{num_of_batch} with streaming batch size {streaming_batch_size}")
#             if (k+1)*streaming_batch_size > len(train_dataset):
#                 end_index = len(train_dataset)
#             else:
#                 end_index = (k+1)*streaming_batch_size
#             sub_trainset = train_dataset.select(range(k*streaming_batch_size, end_index))
#             sub_evalset = eval_dataset.select(range(k*streaming_batch_size, end_index))
#             ttlr_model.forward(train_batch=sub_trainset, predict_batch=sub_evalset)
#     else:
#         raise ValueError(
#             f'NO such setting: {finetuning_args.setting}'
#         )