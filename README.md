The project includes the benchmark data and code collected from our paper "Multi-Stage LLM Fine-Tuning with a Continual Learning Setting" (Findings of NAACL 2025).

## Introduction

This work addresses the challenge of multi-stage fine-tuning of large language models (LLMs) in a continual learning setting, particularly when domains undergo rapid changes. Standard fine-tuning methods often degrade model performance due to two primary obstacles: conflicts between new and old knowledge, which can lead to hallucinations, and the scarcity of fine-tuning data compared to the vast pre-training corpus. To mitigate these issues, we propose a conflict-aware fine-tuning approach that detects and selectively forgets outdated conflicting knowledge, ensuring better integration of new information. Additionally, we introduce a model-based self-distillation data augmentation technique to enhance training data quality through background enrichment, logic-driven augmentation, and paraphrasing. A reasoning-based data selection strategy further improves learning efficiency. Experimental results in both domain-independent and cross-domain scenarios demonstrate the effectiveness of our method in mitigating knowledge degradation and preserving non-conflicting original knowledge. Notably, in multi-stage fine-tuning with LLaMA3-8B, our approach improves accuracy by 46.9%, whereas conventional continual fine-tuning methods suffer from significant performance drops.

## Methods

The figure illustrates the overview of our approach, which consists of three main modules.
Particularly, our method first employs a preference based learning bias to resolve potential knowledge conflicts between the training data and the knowledge store in the current model.
Then, it uses  self distillation strategies to augment training data, with 
a dynamic sample selection mechanism to filter noise and improve learning. 
![19951728136653_ pic](https://github.com/user-attachments/assets/b03a14e5-e16c-424b-b591-47c864115489)

## Data Availability

Due to data timeliness and privacy regulations, we provide the questions from the dataset along with scripts for constructing the training and testing datasets. To generate the complete data, simply supplement the API interface and run the following command:

```bash
bash Data_progress.sh
```

## Model Evluation

To validate the model metrics, you can run the following command:
```bash
bash eval.sh```
