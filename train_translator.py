import os
import os.path as path
import argparse
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

from crosslingual_bert.dataset import ParallelDataset, ParallelTrainDataset, BertTokenizer
from crosslingual_bert.trainer import TranslatorTrainer, TranslatorTrainerConfig
from crosslingual_bert.model import MultilingualTranslator, MultilingualConfig, MultilingualBert


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model arguments
    parser.add_argument("--vocab_file", type=str, default="./data/bert-base-multilingual-cased-vocab.txt")
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--intermediate_size", type=int, default=1536)
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
    parser.add_argument("--checkpoint_folder", type=str, default="./translator_checkpoints/")
    parser.add_argument("--restore_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_frequency", type=int, default=10)

    # device parameters
    parser.add_argument("--on_memory", action="store_true")
    parser.add_argument("--enable_cuda", action="store_true")
    parser.add_argument("--batch_workers", type=int, default=0)

    args = parser.parse_args()
    if args.target_language in args.languages:
        raise ValueError("target language %s is in input languages [%s]" %
                (args.target_language, ', '.join(args.languages)))

    # load pretrained AXLM
    tokenizer = BertTokenizer(args.vocab_file)
    axlm_config = MultilingualConfig.from_json_file(path.join(args.axlm_folder, "config.json"))
    axlm_model = MultilingualBert(axlm_config)

    axlm_state = OrderedDict()
    pretrained_state = torch.load(path.join(args.axlm_folder, "best.model.state"))
    for key in axlm_model.state_dict():
        axlm_state[key] = pretrained_state["model"]["multilingual_model.%s" % key]

    axlm_model.load_state_dict(axlm_state)
    translator_config = MultilingualConfig(
        languages=args.languages,
        target_language=args.target_language,
        vocab_size_or_config_json_file=len(tokenizer.vocab),
        hidden_size=2*axlm_config.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_act=args.hidden_act,
        hidden_dropout_prob=args.hidden_dropout_prob,
        initializer_range=args.initializer_range
    )
    trainer_config = TranslatorTrainerConfig(
        model_config=translator_config,
        lr=args.learning_rate,
        with_cuda=args.enable_cuda,
        train_freq=args.train_step
    )

    # load datasets
    tokenizer = BertTokenizer(args.vocab_file)
    train_files = [(language, "./example_data/sample.%s-%s.tsv" % (language, args.target_language))
            for language in args.languages]
    train_raw = [(language, ParallelTrainDataset(file_path, tokenizer, args.sequence_length, language, args.target_language))
            for language, file_path in train_files]
    train_data = {language: DataLoader(dataset, batch_size=args.batch_size, num_workers=args.batch_workers, pin_memory=args.enable_cuda)
            for language, dataset in train_raw}

    # initialize trainer and model
    loss_log = path.join(args.checkpoint_folder, "loss.tsv")
    try:
        if not args.restore_checkpoint:
            raise FileNotFoundError

        # get best loss from loss record
        with open(loss_log, 'r') as f:
            lines = f.readlines()

        best_epoch = 0
        best_loss = 1e9
        for line in lines[1:]: # skip column names
            epoch, _, test_loss = line.strip().split('\t')
            epoch = int(epoch)
            test_loss = float(test_loss)
            if test_loss < best_loss:
                best_epoch = epoch
                best_loss = test_loss

        # try restoring from checkpoint
        trainer, start = TranslatorTrainer.load_checkpoint(args.checkpoint_folder,
                MultilingualTranslator, train_data, train_data)
    except FileNotFoundError:
        translator_model = MultilingualTranslator(axlm_model, translator_config)
        trainer = TranslatorTrainer(translator_model, trainer_config, train_data, train_data)
        start = 0
        best_epoch = 0
        best_loss = 1e9

    # try and create checkpoint folder
    try:
        os.mkdir(args.checkpoint_folder)
    except FileExistsError:
        pass

    # start training, periodically save model
    checkpoint_file = path.join(args.checkpoint_folder, "checkpoint.state")
    best_model_file = path.join(args.checkpoint_folder, "best.model.state")
    save_epoch_file = path.join(args.checkpoint_folder, "epoch.%d.state")
    with open(loss_log, 'w+' if start == 0 else 'a') as f:
        if start == 0:
            f.write("epoch\ttrain\ttest\n")

        for epoch in range(start, args.epochs):
            epoch += 1
            train_loss = trainer.train(epoch)
            test_loss = trainer.test(epoch)
            trainer.save(checkpoint_file)
            f.write("%d\t%.6f\t%.6f\n" % (epoch, train_loss, test_loss))

            if test_loss < best_loss:
                best_epoch = epoch
                best_loss = test_loss
                trainer.save(best_model_file)
            if epoch % 10 == 0:
                trainer.save(save_epoch_file % epoch)

