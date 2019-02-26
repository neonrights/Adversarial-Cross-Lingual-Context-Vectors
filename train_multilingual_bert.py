import os
import os.path as path
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, DistributedSampler

from pytorch_pretrained_bert import BertTokenizer
from crosslingual_bert.dataset import LanguageDataset, DiscriminatorDataset
from crosslingual_bert.model import MultilingualBert, MultilingualConfig
from crosslingual_bert.trainer import AdversarialPretrainer, DistributedAdversarialPretrainer, AdversarialPretrainerConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument("--vocab_file", type=str, default="./example_data/bert-base-multilingual-cased-vocab.txt")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--hidden_act", type=str, default="gelu")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--attention_dropout_prob", type=float, default=0.1)
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument("--type_vocab_size", type=int, default=16)
    parser.add_argument("--initializer_range", type=float, default=0.02)

    # training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_step", type=int, default=1)
    parser.add_argument("--adversary_batch_size", type=int, default=64)
    parser.add_argument("--sequence_length", type=int, default=192) # XNLI max sequence length with wordpiece tokenization is 167
    parser.add_argument("--adversary_repeat", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adversary_loss_weight", type=float, default=1e-4)
    parser.add_argument("--frobenius_loss_weight", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--warm_start", action="store_true")

    # checkpoint parameters
    parser.add_argument("--checkpoint_folder", type=str, default="./checkpoints/")
    parser.add_argument("--restore_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_frequency", type=int, default=10)

    # hardware parameters
    parser.add_argument("--enable_cuda", action="store_true")
    parser.add_argument("--batch_workers", type=int, default=0)
    parser.add_argument("--adversary_workers", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=None)

    # debugging
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    args.world_size = torch.cuda.device_count() if args.local_rank else 1

    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank)
        print("Started process %d" % args.local_rank)

    # initialize model and trainer configurations
    ltoi = {'ar': 0, 'bg': 1, 'de': 2, 'en': 3}
    if args.warm_start:
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    else:
        tokenizer = BertTokenizer(args.vocab_file)

    model_config = MultilingualConfig(
        languages=ltoi,
        vocab_size_or_config_json_file=len(tokenizer.vocab),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_act=args.hidden_act,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=args.type_vocab_size,
        initializer_range=args.initializer_range,
    )

    trainer_config = AdversarialPretrainerConfig(
        model_config=model_config,
        language_ids=ltoi,
        adv_repeat=args.adversary_repeat,
        lr=args.learning_rate,
        beta=args.adversary_loss_weight,
        gamma=args.frobenius_loss_weight,
        with_cuda=args.enable_cuda,
        train_freq=args.train_step,
        gpu_id=args.local_rank
    )

    # load datasets
    if args.debug:
        train_files = [('ar', "./example_data/ar/"), ('bg', "./example_data/bg/"),
                ('de', "./example_data/de/"), ('en', "./example_data/en/")]
        adversary_file = "./example_data/"
        test_files = [('ar', "./example_data/ar"), ('bg', "./example_data/bg/"),
                ('de', "./example_data/de/"), ('en', "./example_data/en/")]
    else:
        train_files = [('ar', "/mnt/SSD120/wikidata/arwiki/"), ('bg', "/mnt/SSD120/wikidata/bgwiki/"),
                ('de', "/mnt/SSD120/wikidata/dewiki/"), ('en', "/mnt/SSD120/wikidata/enwiki/")]
        adversary_file = "./data/train/"
        test_files = [('ar', "./data/test/ar"), ('bg', "./data/test/bg/"),
                ('de', "./data/test/de/"), ('en', "./data/test/en/")]

    train_raw = [(language, LanguageDataset(language, file_path, tokenizer, args.sequence_length, verbose=not args.local_rank))
            for language, file_path in train_files]
    adversary_raw = DiscriminatorDataset(adversary_file, tokenizer, ltoi, args.sequence_length, verbose=not args.local_rank)

    test_raw = [(language, LanguageDataset(language, file_path, tokenizer, args.sequence_length, verbose=not args.local_rank))
            for language, file_path in test_files]

    if args.local_rank is not None:
        train_raw = [(language, dataset, DistributedSampler(dataset))
                    for language, dataset in train_raw]
        adversary_sampler = DistributedSampler(adversary_raw)

        test_raw = [(language, dataset, DistributedSampler(dataset))
                    for language, dataset in test_raw]
    else:
        train_raw = [(language, dataset, None) for language, dataset in train_raw]
        adversary_sampler = None

        test_raw = [(language, dataset, None) for language, dataset in test_raw]

    train_data = {language: DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                    num_workers=args.batch_workers, drop_last=True, pin_memory=args.enable_cuda)
            for language, dataset, sampler in train_raw}
    train_data["adversary"] = DataLoader(adversary_raw, batch_size=args.adversary_batch_size, sampler=adversary_sampler,
            num_workers=args.adversary_workers, pin_memory=args.enable_cuda)

    test_data = {language: DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                    num_workers=args.batch_workers, drop_last=True, pin_memory=args.enable_cuda)
            for language, dataset, sampler in test_raw}

    print({key: len(value) for key, value in train_data.items()})
    print({key: len(value) for key, value in test_data.items()})

    # initialize model and trainer
    trainer_class = DistributedAdversarialPretrainer if args.local_rank is not None else AdversarialPretrainer
    loss_log = path.join(args.checkpoint_folder, "loss.tsv")
    print("initializing model")
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
        trainer, start = trainer_class.load_checkpoint(args.checkpoint_folder,
                MultilingualBert, train_data, test_data, position=args.local_rank)
    except FileNotFoundError:
        if args.local_rank is not None:
            torch.manual_seed(80085)

        if args.warm_start:
            model = MultilingualBert.from_pretrained_bert(ltoi, "bert-base-cased")
            model_config = model.config
            trainer_config = AdversarialPretrainerConfig(
                model_config=model_config,
                language_ids=ltoi,
                adv_repeat=args.adversary_repeat,
                lr=args.learning_rate,
                beta=args.adversary_loss_weight,
                gamma=args.frobenius_loss_weight,
                with_cuda=args.enable_cuda,
                train_freq=args.train_step,
                gpu_id=args.local_rank
            )
        else:
            model = MultilingualBert(model_config)

        trainer = trainer_class(model, trainer_config, train_data, test_data, position=args.local_rank, seed=420)
        start = 0
        best_epoch = 0
        best_loss = 1e9

    try:
        os.mkdir(args.checkpoint_folder)
    except FileExistsError:
        pass

    # train model, checkpoint every 10th epoch
    print("starting training")
    checkpoint_file = path.join(args.checkpoint_folder, "checkpoint.state")
    best_model_file = path.join(args.checkpoint_folder, "best.model.state")
    save_epoch_file = path.join(args.checkpoint_folder, "epoch.%d.state")
    if not args.local_rank:
        with open(loss_log, 'w+' if start == 0 else 'a') as f:
            if start == 0:
                f.write('epoch\ttrain\ttest\n')

            for epoch in range(start, args.epochs):
                epoch += 1
                train_loss = trainer.train(epoch)
                test_loss = trainer.test(epoch)

                f.write("%d\t%.6f\t%.6f\n" % (epoch, train_loss, test_loss))
                trainer.save(epoch, checkpoint_file)

                if epoch % args.checkpoint_frequency == 0:
                    trainer.save(epoch, save_epoch_file % epoch)

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_epoch = epoch
                    trainer.save(epoch, best_model_file)

                print("test loss %.6f" % test_loss)

        print("best loss %f at epoch %d" % (best_loss, best_epoch))
    else:
        for epoch in range(start, args.epochs):
            epoch += 1
            train_loss = trainer.train(epoch)
            test_loss = trainer.test(epoch)


