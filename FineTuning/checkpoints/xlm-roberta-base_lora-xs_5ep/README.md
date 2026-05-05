---
license: mit
library_name: peft
tags:
- generated_from_trainer
metrics:
- accuracy
base_model: xlm-roberta-base
model-index:
- name: output_2026-05-03 21:12:32.590610
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output_2026-05-03 21:12:32.590610

This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9081
- Accuracy: 0.6083

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 32
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5.0

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|
| 1.1428        | 1.0   | 7088  | 1.2704          | 0.4579   |
| 1.1106        | 2.0   | 14176 | 1.1349          | 0.5191   |
| 1.0605        | 3.0   | 21264 | 1.0206          | 0.5552   |
| 0.9861        | 4.0   | 28352 | 0.9493          | 0.5902   |
| 0.9583        | 5.0   | 35440 | 0.9081          | 0.6083   |


### Framework versions

- PEFT 0.10.0
- Transformers 4.40.1
- Pytorch 2.2.1+cu121
- Datasets 2.16.1
- Tokenizers 0.19.1