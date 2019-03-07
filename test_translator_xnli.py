import os
import os.path as path
import argparse
import torch
from torch.utils.data import DataLoader

from crosslingual_bert import ParallelDataset, ParallelTrainDataset, BertTokenizer
from crosslingual_bert import TranslatorTrainer
from crosslingual_bert import MultilingualTranslator, BertConfig
from crosslingual_bert import EvaluateXNLI


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model arguments
    parser.add_argument("--vocab_file", type=str, default="./example_data/bert-base-multilingual-cased-vocab.txt")
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--intermediate_size", type=int, default=768)
    parser.add_argument("--hidden_act", type=str, default="gelu")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--languages", type=str, nargs='+', required=True)
    parser.add_argument("--target_language", type=str, required=True)
    parser.add_argument("--axlm_folder", type=str, required=True)

    # training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_step", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=192) # XNLI max sequence length with wordpiece tokenization is 167
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=1000)

    # checkpoint parameters
    parser.add_argument("--checkpoint_folder", type=str, default="./checkpoints/")
    parser.add_argument("--restore_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_frequency", type=int, default=10)

    # device parameters
    parser.add_argument("--on_memory", action="store_true")
    parser.add_argument("--enable_cuda", action="store_true")
    parser.add_argument("--batch_workers", type=int, default=0)

    args = parser.parse_args()

    # load datasets
    ar_en_raw = ParallelTrainDataset('./example_data/sample.ar-en.tsv', tokenizer, seq_len, 'ar', 'en')
    bg_en_raw = ParallelTrainDataset('./example_data/sample.bg-en.tsv', tokenizer, seq_len, 'bg', 'en')
    de_en_raw = ParallelTrainDataset('./example_data/sample.de-en.tsv', tokenizer, seq_len, 'de', 'en')

    ar_en_data = DataLoader(ar_en_raw, batch_size=16, shuffle=True, num_workers=4)
    bg_en_data = DataLoader(ar_en_raw, batch_size=16, shuffle=True, num_workers=4)
    de_en_data = DataLoader(ar_en_raw, batch_size=16, shuffle=True, num_workers=4)

    train_data = {
        'ar': ar_en_data,
        'bg': bg_en_data,
        'de': de_en_data
    }

    # load multilingual model
    multi_model = torch.load('./example_data/sample.model')
    multi_model = multi_model.multilingual_model
    translator_model = MultilingualTranslator(multi_model, 'en', config)

    # train model
    trainer = TranslatorTrainer(translator_model, languages, target_language,
            train_data, train_data)

    best_epoch = 0
    best_loss = 1e9
    for epoch in range(100):
        epoch += 1
        trainer.train(epoch)
        test_loss = trainer.test(epoch)
        if test_loss < best_loss:
            best_epoch = epoch
            best_loss = test_loss
            trainer.save(epoch, 'translator', 'best.model')
        if epoch % 10 == 0:
            trainer.save(epoch, 'translator')

