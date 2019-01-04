# Language Adversarial Training for Cross-lingual Contextualized Vectors

## Introduction
Implementation of multi-task shared-private model in order to create context dependent cross-lingual representations of sentences.  Model jointly performs BERT's pretraining task for a set of languages and their corresponding corpora.  BERT's masked token prediction task is modified to have a probability of swapping an input token with a context equivalent token in another language.  This is in addition to thhe probability of swapping an input token with the mask token or a random token.

## Usage

### Data format
All pytorch dataset classes are implemented in the folder *dataset*.

#### Pretraining Data
Each sample should be a sequence of sentences separated by a tab.  Each line in the data file should contain a different sample.  Since pretraining involves determining if a sequence of sentences follows another sequence, each sample must contain at least two sentences.

```txt
This is a sample text.	It is untokenized.	Only sentences are separated by tabs.
This is another sample.	Each sample should contain at least two sentences.
```

The file can be used by the dataset class **LanguageDataset** to train for a specific language.

The adversary uses a separate dataset during training implemented by the class **DiscriminatorDataset**.  The data file is much the same, with each line consisting of a tab separated sequence of sentences.  However, each sample should have the name of each language before each sequence.

```txt
en	This is a sample english text.	The language name is separated from the sentences by a tab as well.
zh	这是一句话	这也是一句话
```

#### Translator Data
Translator data should be entered as a .tsv file, with the name of each language as the header for each column.

```tsv
en	zh
This is a sentence.	这是一句话
I love cats!	我爱猫!
```

The classes **ParallelDataset** and **ParallelTrainDataset** make use of this data format.

### Pretraining
A sample script for pretraining can be viewed in the file *train_multilingual_bert.py*.  Data used in the script are contained in the folder *example_data*.

Pretraining consists of first training the adversary for a set number of cycles before then training a batch for each language model in a round-robin fashion.  Each language model is trained using BERT's pretraining task and is optimized using a weighted sum of BERT's task specific losses, adversary prediction loss, and the squared Frobenius distance between shared and private features.

Pretraining is performed by using the class **AdversarialPretrainer** implemented in the *trainers* folder.

### Translator Tuning and XNLI Evaluation
A sample script for fine-tuning and evaluation can be viewed in the file *train_translator_test_xnli.py*.

Fine-tuning consists of training a transformer model on top of a pretrained model which performs universal translations to a target language.  In the case of the training script this is English.  The final model is then evaluated on the XNLI data set in order to produce a set of BLEU scores.

## Data
Current data is generated from the OpenSubtitles Corpus, downloaded from OPUS: the open parallel corpus.  Data is generated using the **SequenceGenerator** and **OpenSubtitlesReader** classes implemented in the *corpus* folder.  Further data is being generated from Wikipedia in order to capture sentences in written documents rather than the more conversational nature of OpenSubtitles.

## Coding References
Implementation of transformer architecture and wordpiece tokenizers are taken from [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT).

Implementation of training classes take inspiration from [BERT-pytorch](https://github.com/codertimo/BERT-pytorch).
