# Adversarial Pretraining of Cross-Lingual Language Models

## Introduction
Implementation of multi-task shared-private model in order to create context dependent cross-lingual representations of sentences.  Model jointly performs BERT's pretraining task for a set of languages and their corresponding corpora.  This is in addition to the probability of swapping an input token with the mask token or a random token.

## Installation
CPU and single GPU training only require packages in ```requirements.txt```.  Distributed training requires installation of NVIDIA Apex.

NVIDIA Apex <https://github.com/NVIDIA/apex/> (note install with ```--cuda_ext``` flag to enable needed cuda support)

## Usage

### Pretraining
Pre-training is executed by running the command ```python train_multilingual_bert.py [--help] [ARGS]```.  Training can be parallelized through the use of pytorch's distributed launch, e.g. ```python -m torch.distributed.launch --nproc_per_node=N train_multilingual_bert.py [ARGS]```.

Pretraining consists of first training the adversary for a set number of cycles before then training a batch for each language model in a round-robin fashion.  Each language model is trained using BERT's pretraining task and is optimized using a weighted sum of BERT's task specific losses, adversary prediction loss, and the squared Frobenius distance between shared and private features.

Pretraining is performed by using the class **AdversarialPretrainer** implemented in the *trainers* folder.

#### Data
Each sample should be contained within its own document, with each sentence occupying its own line.  Since pretraining involves determining if a sequence of sentences follows another sequence, preferably each sample should contain at least two sentences.  Samples that do not are ignored when loading datsets.

```txt
This is a sample text.
It is untokenized.
Only complete sentences are on each line.
```

Samples belonging to a single language should be kept in the same directory with the name of the directory corresponding to the language.  In the below example, ```en``` will be interpreted as a dataset with 3 samples while ```vi``` will be interpreted as a different dataset with 2 samples.

```txt
data
|-- en
    |-- sample_1.txt
    |-- sample_2.txt
    |-- en_subdir
        |-- sample_3.txt
|-- vi
    |-- sample_1.txt
    |-- sample_2.txt
```

The file can be used by the dataset class **LanguageDataset** to train for a specific language.

The adversary uses a separate dataset during training implemented by the class **DiscriminatorDataset**.  Samples are stored in exactly the same manner, with a single sample in each document, with a single sentence per line.

Example dataset can be found in the folder *example_data*.

### Translator Tuning and XNLI Evaluation
Translator training is executed by running ```python train_translator.py [--help] [ARGS]```.

Fine-tuning consists of training a transformer model on top of a pretrained model which performs universal translations to a target language.  In the case of the training script this is English.  The final model is then evaluated on the XNLI data set in order to produce a set of BLEU scores.

#### Translator Data
Translator data should be entered as a .tsv file, with the name of each language as the header for each column.

```tsv
en	zh
This is a sentence.	这是一句话
I love cats!	我爱猫!
```

The classes **ParallelDataset** and **ParallelTrainDataset** make use of this data format.

## Data Sources
Current data is generated from the OpenSubtitles Corpus, downloaded from OPUS: the open parallel corpus, and various Wikipedia dumps through the use of the tool WikiExtractor <https://github.com/attardi/wikiextractor>.

## Coding References
Implementation of transformer architecture and wordpiece tokenizers are taken from [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

Implementation of training classes take inspiration from [BERT-pytorch](https://github.com/codertimo/BERT-pytorch).
