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
	assert args.target_language is not in args.languages, "target language is in input languages"

    # load pretrained AXLM
    tokenizer = BertTokenizer(args.vocab_file)
    pretrained_state = torch.load(path.join(args.axlm_folder, "best.model.state"))
    bert_model = pretrained_state["model"].multilingual_model
    bert_config = bert_model.config
    translator_config = None # generate translator config

	# load datasets
	tokenizer = BertTokenizer(args.vocab_file)
	train_files = [(language, "./example_data/sample.%s-%s.tsv" % (language, args.target_language))
			for language in args.languages]
	train_raw = [(language, ParallelTrainDataset(file_path, tokenizer, args.sequence_length, language, args.target_language))
			for language, file_path in train_files]
	train_data = {language: DataLoader(train_raw, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_workers, pin_memory=args.enable_cuda)
			for language, dataset in train_raw}

	# load multilingual model
	multi_model = torch.load('./example_data/sample.model')
	multi_model = multi_model.multilingual_model
	translator_model = MultilingualTranslator(multi_model, 'en', config)

	# train model
	trainer = TranslatorTrainer(translator_model, translator_config, train_data, train_data)

	best_epoch = 0
	best_loss = 1e9
	for epoch in range(args.epochs):
		epoch += 1
		trainer.train(epoch)
		test_loss = trainer.test(epoch)
		if test_loss < best_loss:
			best_epoch = epoch
			best_loss = test_loss
			trainer.save(epoch, 'translator', 'best.model')
		if epoch % 10 == 0:
			trainer.save(epoch, 'translator')

