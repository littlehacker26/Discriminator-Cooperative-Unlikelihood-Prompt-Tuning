
# DisCup
This repository contains code for the paper [DisCup: Discriminator Cooperative Unlikelihood Prompt Tuning for Controllable Text Generation](https://arxiv.org/abs/2210.09551) which is appeared at EMNLP2022. If you have any questions, please feel free to create an issue or contact the email of the first author: zhanghanqing@bit.edu.cn

## Description of Main files 
- `discrimination.py`: discriminator training(i.e., detoxic classifier and sentiment classifier), and pure disriminator-based(FUDGE) generation
- `prompt_tuning.py`: the implemetation of vanilla prompt-tauning; it contains the vanilla prompt training and prompt-based generation
- `distill_tuning.py`: the implemetation of DisCup; it contains the discriminator cooperative unlikelihood prompt training and prompt-based generation
- `/script`: it contains the bash commands for model trainng and controllable text generation
- `evaluate.py`: evaluate the generated texts (i.e., dist1/dist-2/dist-3, Perplexity, and domain keyword coverage) with *.txt* format.


## Dependence

- **Install the following Conda environment**
- - our code is bulit on `python3.6`
- - pip install -r `requirements.txt`
- **Download the datasets**: [click here](https://drive.google.com/file/d/1jeBGqImwkGJhEELDMUbIP4n0nAbR58Ox/view?usp=sharing)
- **Downlad the check_points**: [click here](https://drive.google.com/file/d/1k4qSpYhuS1SYWL0SVmQ6CuSH_PdAYjdc/view?usp=sharing)
- **Prepare the GPT2 models**:
- - [GPT2-small](https://huggingface.co/gpt2)
- - [GPT2-large](https://huggingface.co/gpt2-large)

After downloading the trained check-points, you also can directly jump to `Controllable Text Generation`, conducting text generation experiments.

## Discriminator Training
It contains the training process of attribute-discriminator, and it is the premise of the DisCup.


**Sentiment classifer training**
- cd ./script
- bash train_sentiment_disc.bash

**Detoxic classifer training**
- cd ./script
- bash train_detoxic_disc.bash

**Parameter Configuration** 
- `--data_path`: training corpus data for classifer training
- `--model_name_or_path`： the path for the pretrained language model, we use GPT2-samll here
- `--out_dir`: the output directory  to save the check-point
- `--template`: configure the prompt length of the control-prompt


## Control-prompt Tuning

It contains the training process of control-prompts for vanilla-prompt tuning and DisCup.

**Sentiment control task**
- cd ./script

- `bash train_sentiment_distill.bash` or `bash train_sentiment_prompt.bash`

**Detoxic task**
- cd ./script

- `bash train_detoxic_distill.bash` or `bash train_detoxic_prompt.bash`

**Parameter Configuration**

- `--data_path`:  the training corpus for prompt-tuning, attribute-specific corpus should  be set for vanilla prompt-tuning  
- `--model_name_or_path`： the path for the pretrained langauge model, we use GPT2-Large here
- `--out_dir`: the output directory to save the control-prompts
- `--disc_embedding_checkpoint`: the path of trained discriminators, it only needs to be specified in DisCup
- `--template`: configure the prompt length
- `--ranking_scope`: configure the size of re-ranked candidate tokens, it only needs to be specified in DisCup
- `--corpus_type`: the attribute of control-prompts, it contain [positive" "negative"] in sentiment control generation, only 'positive' is optional in toxicity avoidance task
- `--temperature`: configure the distribution shapeness of re-ranked candidate tokens, it only needs to be specified in DisCup


## Controllable Text Generation
It contains the generation processes for vanilla-prompt and DisCup.


**Sentiment control task**
- cd ./script

- `bash generate_sentiment_distill.bash` or `bash generate_sentiment_prompt.bash`

**Detoxic task**
- cd ./script

- `bash generate_detoxic_distill.bash` or `bash generate_detoxic_prompt.bash`


**Parameter Configuration**

- `--data_path`:  the prompts data for text generation
- `--model_name_or_path`： the path for the pretrained langauge model, we use GPT2-Large here
- `--file_name`: the output directory to save the generation results, it is saved with '.csv' format
- `--embedding_checkpoint`: the path of the saved control-prompts
- `--template`: configure the prompt length, which is consistent to the actual prompt length of embedding_checkpoint
- `--prompt_type`: specify the prompt type, it contain ["neutral" "positive" "negative"] in sentiment generation, only 'negative' is  optional in detoxic
- `--target_type`: consistent to the type embedding_checkpoint it contain [positive" "negative"] in sentiment control generation, only 'positive' is optional in toxicity avoidance task


## Evaluation:
- For sentiment control \
  you can refer to `sentiment_classifier.ipynb`, and evaluate the correctness and PPL of your generated csv file. Then, you can convert the csv file to txt, using the function save_csv_to_text, so as to run the `evaluate.py` to obtain the results of dist-1/2/3 and and domain keyword coverage.

- For toxicity avoidance \
  you could test the toxicity prob with the [Google API](https://perspectiveapi.com/), the measurement of dist-1/2/3 and  PPL is same as sentiment control task.

## Citation
```
@inproceedings{zhang-song-2022-discup,
    title = "{D}is{C}up: Discriminator Cooperative Unlikelihood Prompt-tuning for Controllable Text Generation",
    author = "Zhang, Hanqing  and
      Song, Dawei",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.223",
    pages = "3392--3406",
    abstract = "Prompt learning with immensely large Casual Language Models (CLMs) has been shown promising for attribute-controllable text generation (CTG). However, vanilla prompt tuning tends to imitate training corpus characteristics beyond the control attributes, resulting in a poor generalization ability. Moreover, it is less able to capture the relationship between different attributes, further limiting the control performance. In this paper, we propose a new CTG approach, namely DisCup, which incorporates the attribute knowledge of discriminator to optimize the control-prompts, steering a frozen CLM to produce attribute-specific texts. Specifically, the frozen CLM model, capable of producing multitudinous texts, is first used to generate the next-token candidates based on the context, so as to ensure the diversity of tokens to be predicted. Then, we leverage an attribute-discriminator to select desired/undesired tokens from those candidates, providing the inter-attribute knowledge. Finally, we bridge the above two traits by an unlikelihood objective for prompt-tuning. Extensive experimental results show that DisCup can achieve a new state-of-the-art control performance while maintaining an efficient and high-quality text generation, only relying on around 10 virtual tokens.",
}
```

The part of the code was built on top of [All NLP Tasks Are Generation Tasks: A General Pretraining Framework](https://github.com/THUDM/P-tuning).
