<h1 align="center">
     <br>Self-Supervised Chain-of-Thought Learning for Test-Time Reasoning Enhancement in LLMs

<h4 align="center"></a>

Thank you for visiting this repository!
This repository contains the implementation of our ICML-26 submission, **"Self-Supervised Chain-of-Thought Learning for Test-Time Reasoning Enhancement in LLMs"**.

This repository builds upon and reuses substantial parts of the implementation from the excellent [repo](https://github.com/Fhujinwu/TLM/tree/main) of the TTL paper, and the [LlamaFactory](https://github.com/hiyouga/LlamaFactory) library. We are very thankful to both of these groups!


In order to use the code, please follow these steps:
## 1- Install requirements
~~~
pip install -r requirements.txt
~~~

## 2- Downloading Datasets
You can download the datasets used in this work from this [anonymous link](https://drive.google.com/drive/folders/1ffCuF9iYE6AR012Ocj9E2TSUWFsCrN4y?usp=sharing) and place them at `data/AdaptEval`.

If you are interested in generating the CoT augmented datasets yourself, you can use the script at `src/llamafactory/train/ttlr/create_augmented_dataset.py`, and then run `src/llamafactory/train/ttlr/data_pruner.py`.

To perform sample selection, you can use the script at `src/llamafactorytrain/ttlr/data_selector.py`.

## 3- Training

All datasets must be defined in the `dataset_info.json` file included in this repository. You need to specify the desired dataset in your configuration file to use it.

For example, to train using the GSM8k dataset, you can start training with the following command::
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/offline_ttlr.yaml
```
The configuration files (e.g., `offline_ttlr.yaml`) can be accessed from `examples/train_lora`.

## 4- Evaluation

All evaluation-related scripts are located in the `scripts/eval` folder:
- For GSM8K, MetaMathQA, LogiQA, and CoLoTa, copy the path to your model inference results into `eval_accuracy.py` and run the script.
- For 

## 💬 Citation
Thanks for the open-source code of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

If you find our work interesting and meaningful, welcome to give a 🌟 to our repo and cite our paper.

```text
@inproceedings{hutest,
  title={Test-Time Learning for Large Language Models},
  author={Hu, Jinwu and Zhang, Zitian and Chen, Guohao and Wen, Xutao and Shuai, Chao and Luo, Wei and Xiao, Bin and Li, Yuanqing and Tan, Mingkui},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=Fhujinwu/TLM&type=Date)
