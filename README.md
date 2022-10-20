
# DisCup
This repository contains code for the paper [DisCup: Discriminator Cooperative Unlikelihood Prompt Tuning for Controllable Text Generation](https://arxiv.org/abs/2210.09551) which is appeared at EMNLP2022. If you have any questions, please feel free to create an issue or contact the email of the first author: zhanghanqing@bit.edu.cn


## Main File Description
- `discrimination.py`: discriminator training(i.e., detoxic classifier and sentiment classifier), and disriminator-based method generation
- `prompt_tuning.py`: the implemetation of vanilla prompt-tuning; it contains the vanilla prompt training and prompt-based generation
- `distill_tuning.py`: the implemetation of DisCup; it contains the discriminator cooperative unlikelihood prompt training and prompt-based generation
- `/script`: it  contains the bashes command for model trainng and controllable text generation
- `evaluate.py`: evaluate the generation texts (i.e., dist1/dist-2/dist-3, Perplexity, and domain keyword coverage) with *.txt* format.


## Dependence:

- **Install the following Conda environment**: *./requirements.txt*
- - our code is bulit on `python3.6`
- - pip install -r `requirements.txt`
- **Download the datasets**: [click here](https://drive.google.com/file/d/1jeBGqImwkGJhEELDMUbIP4n0nAbR58Ox/view?usp=sharing)
- **Downlad the check_point**: [click here](https://drive.google.com/file/d/1k4qSpYhuS1SYWL0SVmQ6CuSH_PdAYjdc/view?usp=sharing)
- **Prepare the GPT2 model**:
- - [GPT2-small](https://huggingface.co/gpt2)
- - [GPT2-large](https://huggingface.co/gpt2-large)

## Discriminator Training
**Sentiment classifer training**
- cd ./script
- bash train_sentiment_disc.bash

**Detoxic classifer training**
- cd ./script \
- bash train_detoxic_disc.bash

**Parameter Configuration** 
- `--data_path`: training corpus data for classifer training
- `--model_name_or_path`： the path for the pretrained language model, we use GPT2-samll here
- `--out_dir`: the output direction to save the check-point
- `--template`: configure the prompt length of the control-prompt


## Controllable Text Generation:

**Sentiment control task**
- cd ./script
- bash generate_sentiment_distill.bash

**Detoxic task**
- cd ./script
- bash generate_detoxic_distill.bash

**Parameter Configuration**

- `--data_path`:  the prompts data for text generation
- `--model_name_or_path`： the path for the pretrained langauge model, we use GPT2-Large here
- `--file_name`: the output direction to save the generation result, it is with '.csv' format
- `--embedding_checkpoint`: the path of saved control-prompt
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
@inproceedings{DisCup-2022,
    title = "DisCup: Discriminator Cooperative Unlikelihood Prompt Tuning for Controllable Text Generation",
    author = "Hanqing Zhang and Dawei song",
    booktitle = "The 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
    month = "Dec",
    address = "Abu Dhabi",
    url = {https://arxiv.org/abs/2210.09551},
}
```

The part of the code was built on top of [All NLP Tasks Are Generation Tasks: A General Pretraining Framework](https://github.com/THUDM/P-tuning).
