import os
import json
from itertools import chain

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .optimization import BERTAdam
from .utils import *
from .language_model import NextSentencePrediction, MaskedLanguageModel


import tqdm



class SingleBERTWrapper(nn.Module):
    """
    wrapper that adds pretraining prediction for a single language model
    """
    def __init__(self, language_model, adversary_model, mask_model, next_model, hidden):
        super().__init__()
        self.language_model = language_model
        self.adversary_model = adversary_model
        self.mask_model = mask_model
        self.next_model = next_model
        self.hidden = hidden
    
    def forward(self, *args, **kwargs):
        hidden_vectors, pooled_vectors = self.language_model(*args, **kwargs)
        
        # logits for prediction tasks
        token_logits = self.mask_model(hidden_vectors[-1])
        next_logits = self.next_model(pooled_vectors)

        # public-private vector similarity loss
        public_vectors, private_vectors = torch.split(hidden_vectors[-1], self.hidden // 2, -1)
        diff = torch.bmm(private_vectors, torch.transpose(public_vectors, 1, 2))
        diff_loss = torch.sum(diff ** 2)
        diff_loss /= pooled_vectors.size(0)

        # adversarial prediction
        public_pooled, _ = torch.split(pooled_vectors, self.hidden // 2, -1)
        language_logits = self.adversary_model(public_pooled)

        return token_logits, next_logits, language_logits, diff_loss


class AdversarialBERTWrapper:
    """
    adds pretraining tasks to entire multilingual model
    """
    def __init__(self, multilingual_model, adversary_model, hidden, vocab_size):
        self.multilingual_model = multilingual_model
        self.components = multilingual_model.components.copy()
        self.components['adversary'] = adversary_model
        self.components['mask'] = MaskedLanguageModel(hidden, vocab_size)
        self.components['next'] = NextSentencePrediction(hidden)
        # add necessary prediction task
        self.pretraining_models = [SingleBERTWrapper(self.multilingual_model[language],
                self.components['adversary'], self.components['mask'], self.components['next'],
                hidden) for language in self.multilingual_model.ltoi]

    def __getitem__(self, index):
        if type(index) is str:
            return self.pretraining_models[self.multilingual_model.ltoi[index]]
        else:
            return self.pretraining_models[index]
    
    def __len__(self):
        return len(self.pretraining_models)

    def adversary_forward(self, *args, **kwargs):
        _, pooled_vectors = self.multilingual_model.components['public'](*args, **kwargs)
        return self.components['adversary'](pooled_vectors)

    def get_components(self):
        return self.components

    def parameters(self):
        return chain(self.multilingual_model.parameters(), self.components['adversary'].parameters(),
                self.components['mask'].parameters(), self.components['next'].parameters())


class AdversarialPretrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, multilingual_model, adversary_model, vocab_size: int, hidden_size, languages,
                 train_data, test_data, adv_repeat=5, lr=1e-4, beta=1e-2, gamma=1e-4,
                 with_cuda=True, log_freq=10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: training with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # initialize public, private, and adversarial discriminator
        self.ltoi = languages
        self.model = AdversarialBERTWrapper(multilingual_model, adversary_model, hidden_size, vocab_size)

        freeze(self.model.pretraining_models + [self.model.components['adversary']])

        self.model.components['adversary'].to(self.device)
        for model in self.model.pretraining_models:
            model.to(self.device)

        # get data
        self.train_data = train_data
        self.test_data = test_data
        
        # initialize loss function and optimizers
        self.D_repeat = adv_repeat
        self.criterion = nn.NLLLoss()   
        self.mask_criterion = nn.NLLLoss(ignore_index=0)     
        self.D_optim = Adam(adversary_model.parameters(), lr)
        self.lm_optim = BERTAdam(self.model.parameters(), lr)

        # loss function hyperparameters
        self.beta = beta
        self.gamma = gamma

        self.log_freq = log_freq

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        if train:
            thaw(self.model.components['adversary'])

        for repeat in range(self.D_repeat):
            # for batch in batches
            D_iter = tqdm.tqdm(enumerate(data["adversary"]),
                    desc="D_train:{}:{}/{}".format(epoch+1, repeat+1, self.D_repeat),
                    total=len(data["adversary"]))

            total_loss = 0
            total_correct = 0
            total_elements = 0
            for i, batch in D_iter:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits = self.model.adversary_forward(batch["input_ids"], attention_mask=batch['mask'])
                loss = self.criterion(logits, batch['language_label'])
                
                total_loss += loss.item()
                total_correct += logits.argmax(-1).eq(batch['language_label']).sum().item()
                total_elements += batch['language_label'].nelement()

                if train:
                    self.D_optim.zero_grad()
                    loss.backward()
                    self.D_optim.step()

            print("EP{0}_D_{1}: loss={2:.6f} acc={3:.6f}".format(
                    epoch+1, repeat+1, total_loss / len(D_iter), total_correct / total_elements))

        if train:
            freeze(self.model.components['adversary'])
            thaw(self.model.multilingual_model.components['public'])

        micro_loss = 0
        for language, language_label in self.ltoi.items():
            if language == "adversary":
                continue

            language_iter = tqdm.tqdm(enumerate(data[language]),
                    desc="{}_{}:{}".format(language, str_code, epoch+1,
                    total=len(data[language])))

            language_model = self.model[language]
            if train:
                thaw(language_model.language_model.private)

            total_mask_loss = 0
            total_next_loss = 0
            total_adv_loss = 0
            total_diff_loss = 0
            total_mask_correct = 0
            total_mask_elements = 0
            total_next_correct = 0
            total_next_elements = 0
            for i, batch in language_iter:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                mask_logits, next_logits, language_logits, diff_loss = language_model(batch['input_ids'], batch['segment_label'], batch['mask'])

                mask_loss = self.mask_criterion(mask_logits.transpose(1,2), batch['token_labels'])
                next_loss = self.criterion(next_logits, batch['is_next'])
                language_labels = language_label + torch.zeros(language_logits.size(0), dtype=torch.long)
                adv_loss = -self.criterion(language_logits, language_labels.to(self.device)) # TODO correct loss

                train_loss = mask_loss + next_loss + self.beta * adv_loss + self.gamma * diff_loss
                if train:
                    self.lm_optim.zero_grad()
                    train_loss.backward()
                    self.lm_optim.step()

                micro_loss += train_loss.item()
                total_mask_loss += mask_loss.item()
                total_next_loss += next_loss.item()
                total_adv_loss += adv_loss.item()
                total_diff_loss += diff_loss.item()
                
                mask_correct = mask_logits.argmax(-1).eq(batch['token_labels'])
                mask_elements = (batch['token_labels'] > 0)
                total_mask_correct += (mask_correct & mask_elements).sum().item()
                total_mask_elements += mask_elements.sum().item()

                total_next_correct += next_logits.argmax(dim=-1).eq(batch['is_next']).sum().item()
                total_next_elements += batch['is_next'].nelement()

                if i % self.log_freq == 0:
                    post_fix = {
                        "epoch": epoch+1,
                        "language": language,
                        "iter": i+1,
                        "mask_loss": total_mask_loss / (i + 1),
                        "next_loss": total_next_loss / (i + 1),
                        "adversary_loss": total_adv_loss / (i + 1),
                        "difference_loss": total_diff_loss / (i + 1),
                        "mask_accuracy": total_mask_correct / total_mask_elements,
                        "next_accuracy": total_next_correct / total_next_elements
                    }
                    with open(language + '.log', 'a') as f:
                        f.write(json.dumps(post_fix) + '\n')


            avg_mask_loss = total_mask_loss / len(data[language])
            avg_next_loss = total_next_loss / len(data[language])
            avg_adv_loss = total_adv_loss / len(data[language])
            avg_diff_loss = total_diff_loss / len(data[language])
            avg_mask_acc = total_mask_correct / total_mask_elements
            avg_next_acc = total_next_correct / total_next_elements
            print("EP{0}_{1}_{2}:\n\
                    mask={3:.4f}\n\
                    next={4:.4f}\n\
                    adv={5:.4f}\n\
                    diff={6:.4f}\n\
                    mask_acc={7:0.4f}\n\
                    next_acc={8:0.4f}".format(
                epoch+1, language, str_code, avg_mask_loss, avg_next_loss, avg_adv_loss, avg_diff_loss, avg_mask_acc, avg_next_acc))

            if train:
                freeze(language_model.language_model.private)

        if train:
            freeze(self.model.multilingual_model.components['public'])

        return micro_loss / (len(data) - 1)


    def save(self, epoch, directory_path="output/"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param directory_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        directory_path = os.path.join(directory_path, str(epoch+1))
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        for save_name, save_model in self.model.get_components().items():
            torch.save(save_model.cpu(), os.path.join(directory_path, "{}.model".format(save_name)))
            save_model.to(self.device)

        print("EP:%d Model Saved in:" % epoch, directory_path)
        return directory_path
