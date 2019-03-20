import os
import argparse
import torch
import pandas as pd

from os import path
from collections import OrderedDict, Counter
from nltk.translate.bleu_score import sentence_bleu

from crosslingual_bert.dataset import BertTokenizer
from crosslingual_bert.model import MultilingualBert, MultilingualTranslator, MultilingualConfig
import tqdm


def pad_ids(token_ids, max_length):
    if len(token_ids) < max_length:
        padding = [0 for _ in range(max_length - len(token_ids))]
        token_tensor = torch.tensor(token_ids + padding)
    else:
        token_tensor = torch.tensor(token_ids[:max_length])
    token_tensor = token_tensor.unsqueeze(0)
    token_mask = token_tensor == 0
    return token_tensor.cuda(), token_mask.cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--vocab_file", type=str, default="./example_data/bert-base-multilingual-cased-vocab.txt")
    parser.add_argument("--languages", type=str, nargs='+', required=True)
    parser.add_argument("--target_language", type=str, required=True)
    parser.add_argument("--axlm_folder", type=str, required=True)
    parser.add_argument("--translator_folder", type=str, required=True)
    parser.add_argument("--evaluation_tsv", type=str, default="./data/raw data/XNLI-15way/xnli.15way.orig.tsv")
    parser.add_argument("--max_length", type=int, default=192)

    args = parser.parse_args()

    tokenizer = BertTokenizer(args.vocab_file)
    samples = pd.read_csv(args.evaluation_tsv, sep='\t', usecols=args.languages + [args.target_language])

    axlm_config = MultilingualConfig.from_json_file(path.join(args.axlm_folder, "config.json"))
    axlm_model = MultilingualBert(axlm_config)

    axlm_state = OrderedDict()
    pretrained_state = torch.load(path.join(args.axlm_folder, "best.model.state"))
    for key in axlm_model.state_dict():
        axlm_state[key] = pretrained_state["model"]["multilingual_model.%s" % key]

    axlm_model.load_state_dict(axlm_state)
    axlm_model = axlm_model.eval().cuda()

    translator_config = MultilingualConfig.from_json_file(path.join(args.translator_folder, "config.json"))
    translator_model = MultilingualTranslator(translator_config)

    translator_state = OrderedDict()
    pretrained_state = torch.load(path.join(args.translator_folder, "best.model.state"))
    for key in translator_model.state_dict():
        translator_state[key] = pretrained_state["model"]["translator_model.%s" % key]

    translator_model.load_state_dict(translator_state)
    translator = translator_model.eval().cuda()

    sos_id = tokenizer.vocab['[CLS]']
    eos_id = tokenizer.vocab['[SEP]']
    bleu_scores = Counter()
    for i, row in tqdm.tqdm(samples.iterrows(), total=100):
        gold_ids = tokenizer.tokenize_and_convert_to_ids(row[args.target_language])
        for language in args.languages:
            input_ids = tokenizer.tokenize_and_convert_to_ids(row[language])
            input_ids, input_mask = pad_ids([sos_id] + input_ids + [eos_id], args.max_length)
            output_id = None
            hypothesis = []
            while output_id != eos_id and len(hypothesis) < args.max_length:
                output_ids, output_mask = pad_ids([sos_id] + hypothesis, args.max_length)
                with torch.no_grad():
                    input_vectors, _ = axlm_model(language, input_ids, attention_mask=input_mask)
                    target_vectors, _ = axlm_model(args.target_language, output_ids, attention_mask=output_mask)
                    scores = translator_model(input_vectors[-1], target_vectors[-1], input_mask, output_mask)
                output_id = scores.argmax(-1).item()
                if output_id != eos_id:
                    hypothesis.append(output_id)

            bleu_scores[language] += sentence_bleu([gold_ids], hypothesis)
        if i > 100:
            break

    for key in scores:
        bleu_scores[key] /= 100

    print(bleu_scores)
