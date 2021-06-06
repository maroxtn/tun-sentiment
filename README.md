# <b> Tunisian Arabizi Sentiment Analysis Zindi Competition (Ranked #1st): </b>

This repo contains my approach in solving the Tunisian dialect sentiment analysis problem. My solution took place in [Tunisian Sentiment Analysis competition](https://zindi.africa/competitions/ai4d-icompass-social-media-sentiment-analysis-for-tunisian-arabizi/) in which I ranked first before getting disqualified (for allegedly breaking a rule). Even after the competition ended, my solution scored the best by a significant margin on both public and private dataset.

The competition posed several challenges:

1- Dataset is relatively small

2- Tunisian dialect has a high degree of variance

3- No available pretrained models for Tunisian dialect nor any Maghrebi dialect

4- The text is in latin characters (Arabizi) not Arabic

----

## My solution

Due to the small size of the dataset it would deem impractical to train a model from scratch; it would be hard for a model to learn a language with as many specific cases as Tunisian with a dataset only containing 70k sentences. Appropriately, using a pretrained model seems like the natural solution. 

If we analyze the Tunisian dialect, we would find that it is a melting pot of many language: Arabic by a big margin (root of words, grammar, etc ...), Amazigh words (which is shared among all Maghrebi dialects), French (also shared with some Maghrebi dialects), and to a certain extent some English and Italian.

Because of this diversity, using an ensemble of multiple languages trained on the same dataset would surely yield higher accuracy. But how is it possible to take advantage of Arabic pretrained model and the dataset is in latin letters, one might ask. 

For that reason, I trained an independent transformer model for transliterating from Latin letters to Tunisian using a dataset that I personally harvested and annotated. This dataset contains around 17k commonly used Tunisian words in both Arabic letters and Arabizi. More on the dataset later.

After trying several pretrained models from the huggingface hub, and through lots trial and error, I determined that the combination of `bert-base-uncased`, alongside with `moha/arabert_c19` (multi-dialect Arabic model trained on 1.5M COVID19 tweets, paper: [https://arxiv.org/pdf/2105.03143.pdf](https://arxiv.org/pdf/2105.03143.pdf)) and `camembert` gave the best results. However camembert didn't improve accuracy that much, so I am not using it in this project for less computation time.

I believe that the ensemble of both `bert-base-uncased` and `moha/arabert_c19` is the most optimal because the latter covers all of the words that has Arabic / Amazigh origins (it was trained on a multi-dialectal Arabic, meaning it has inherent understanding for other Maghrebi languages), and bert model covers English, French and even Italian roots (bert-base was trained on wikipedia English which still contains a lot of forein words text). So in that sense, this ensemble would have a more thorough understanding with a reasonable computation time (Training only takes 3h30 on P100, inference for 30k sentences only 2 minutes on the same hardware).

To tackle the small dataset problem, I used cross validation of 10 folds for arabert and 5 folds bert-base, then averaged the output of every fold. This helped me stabilize the score against any form of overfitting.


## Arabic to Arabizi dataset

To create this dataset, I scraped off 25k facebook comments, took the 30k most common words but I only annotated 17k words because of time constraints. This dataset covered 60% of the words that existed in the sentiment analysis dataset.

Note that many of these words are just misspellings of the same word repeating over and over. For example the word Mashallah is spelled machalah, mashalah, machala, macha2allah ....


## Transliteration model

The model I used for the transliteration task is the classical transformer (from Attention is all you need, paper: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)), with the following hyper-parameters: 

```
d_model = 128
dec_layer = 1  #Decoder layers
enc_layer = 1

dropout = 0.15 
epochs = 100
attention_heads = 4
```
With a more thorough hyper-parameter search, model performance would surely improve. Special letters and punctuation were removed before inference then restored afterwards.

## Dataloading and Sampling

One of the keys of the success of my models is dataloading speed. Since the competition training time is limited to 9 hours, and since bert models are infamous for slow computation, optimizing training speed is crucial. For that reason I created a custom <i>data_collator*</i> and a custom sampler.

A <i> data_collator</i> is a function that is a callback function used by pytorch dataloaders, it gets called after batch creation before turning it into a pytorch tensor. It is useful in this context because it helps us pad input sequences the maxlen of the current batch instead of the maxlen of the entire dataset.

For example, let's assume we have a dataset of 10k sentences, and the longest has 300 tokens. Using hugggingface bert preprocessing function, all the sentences in the dataset would be padded up to 300 tokens. This loses us precious computation time. Using a data_collator would allow us to pad sentences to the maximum length of only the batch, not the entire dataset, thus gaining us time. The custom data collator can be found in both `dataloader/bert_dataloader` and `dataloader/transliteration_dataloader.py`.

Custom samplers were also used to reduce the amount of padding in a batch. This custom sampler makes batches similar in length, so when they are padded to its max length, padding is minimal. Code for my custom sampler can be found in `dataloader/custom_sampler.py`.

Using two of these tricks took me from 3 hours of training a model to only 15 minutes, which is quite impressive.


# Use guide

Contains four folders: models (contains the classes of each model used and their inference code), utils (utility functions), dataloader (dataloading code), and Data (contains both competition dataset and my personal dataset)

For the sake of convenience, I added a jupyter notebook which contains the original code of my solution since it might be easier reading it in an interactive notebook, however mind that it is messy and not documented.

To reproduce my solution:

1- Clone repo and install requirements

```
$git clone https://github.com/maroxtn/tun-sentiment.git
$cd tun-sentiment
$pip install -r requirements.txt
```

2- Train transliteration model
```
$python train_transliteration.py --epochs 100 --batch_size 32
$python train_transliteration.py --help


optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number for epochs. Default: 100
  --batch_size BATCH_SIZE
                        Batch size. Default: 32
  --d_model D_MODEL     Transformer model dimension. Default 128
  --save_folder SAVE_FOLDER
                        Folder to save model in. Default: `checkpoint`
  --val_frac VAL_FRAC   Fraction of that data that would constitute as the validation set. Default: 0.1
  --dataset_dir DATASET_DIR
                        Transliteration dataset directory. Default: `data/external/transliteration`

```

3 - Transliterate train and test data

```
$python transliteration_inference.py
$python transliteration_inference.py --help


Transliterate sentences in data/external/['train', 'test'].csv to data/interim/['train', 'test'].csv. No arguments
needed.

optional arguments:
  -h, --help  show this help message and exit
 
```

4 - Train sentiment analysis model, both Bert and Arabert

```
$python train_sentiment.py en --epochs 4 --batch_size 32   #bert-base
$python train_sentiment.py ar --epochs 4 --batch_size 32 --folds 10   #arabert
$python train_sentiment.py --help

usage: train_sentiment.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--evaluate_every EVALUATE_EVERY]
                          [--save_folder SAVE_FOLDER] [--folds FOLDS] [--dropout DROPOUT] [--lr LR]
                          [--log_every LOG_EVERY] [--dataset_dir DATASET_DIR]
                          {ar,en}

Train the sentiment analysis models, if there is any models already trained, this command would replace them.

positional arguments:
  {ar,en}               Language prefix of the huggingface pretrained model. Check model_names.yaml.

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number for epochs. Default: 4
  --batch_size BATCH_SIZE
                        Batch size. Default: 32
  --evaluate_every EVALUATE_EVERY
                        Evaluation every n steps
  --save_folder SAVE_FOLDER
                        Folder to save model in. `checkpoint` is default value
  --folds FOLDS         Number of Kfolds used in training.
  --dropout DROPOUT     Dropout value.
  --lr LR               Learning rate.
  --log_every LOG_EVERY
                        Log every n step.
  --dataset_dir DATASET_DIR
                        Transliteration dataset directory. Default: `data/external/sentiment_analysis`


```

5 - Perform inference on the dataset

```
$python sentiment_inference.py --ar_folds 10 --en_folds 5
$python sentiment_inference.py --help

usage: sentiment_inference.py [-h] [--ar_folds AR_FOLDS] [--en_folds EN_FOLDS]

Perform inference on test dataframe and exports results to data/final/test.csv

optional arguments:
  -h, --help           show this help message and exit
  --ar_folds AR_FOLDS  Number of folds used while training ar bert model.
  --en_folds EN_FOLDS  Number of folds used while training bert-base.

```

Output will be stored as `data\final\Test.csv`.

If you wish to have code in front of you rather than CLI, then `sentimentAnalysis.py` does exactly the same thing as the commands above.

---

## Final thoughts on what might improve accuracy

After everything I tried, there were few things that I did that significantly improved accuracy. First of them is a bigger transliteration dataset, the bigger the dataset for transliteration, the better the accuracy.

Knowing that, I believe that a bigger and a higher quality dataset would surely improve the score. Also a transliteration model that is trained on whole sentences rather than standalone words would be more precise since it would deal with ambiguous words better. Also we might replace common French words that we use in Tunisian with their Arabic translation, that way, we can better hone the knowledge of pretrained Arabic models.

I also believe that once we have a big enough Tunisian dialect dataset, we can train a BERT model then fine-tune it on such tasks and it would out-perform all other methods.