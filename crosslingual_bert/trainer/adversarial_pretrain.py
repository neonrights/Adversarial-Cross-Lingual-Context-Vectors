import os
import json
from itertools import chain

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .optimization import BERTAdam
from .utils import *

import tqdm


class NextSentencePrediction(nn.Module):
    """2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class MaskedLanguageModel(nn.Module):
    """predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class SimpleAdversary(nn.Module):
    """simpled adversarial language predictor
    """
    def __init__(self, hidden_size, language_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, language_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        return self.softmax(self.linear(inputs))


class AdversarialBertWrapper(nn.Module):
    """
    adds pretraining tasks to entire multilingual model
    """
    def __init__(self, multilingual_model, config):
        super().__init__()
        self.multilingual_model = multilingual_model
        self.adversary_model = SimpleAdversary(config.hidden_size, len(config.languages))
        self.mask_model = MaskedLanguageModel(config.hidden_size*2, config.vocab_size)
        self.next_model = NextSentencePrediction(config.hidden_size*2)

    def forward(self, component, input_ids, token_type_ids=None, attention_mask=None):
        if component == 'adversary':
            # return logits for adversarial language prediction
            _, pooled_vectors = self.multilingual_model.shared(input_ids, token_type_ids, attention_mask)
            return self.adversary_model(pooled_vectors)
        else:
            hidden_vectors, pooled_vectors = self.multilingual_model(component, input_ids, token_type_ids, attention_mask)
        
            # logits for prediction tasks
            token_logits = self.mask_model(hidden_vectors[-1])
            next_logits = self.next_model(pooled_vectors)

            # public-private vector similarity loss
            hidden_dim = hidden_vectors[-1].size(-1)
            public_vectors, private_vectors = torch.split(hidden_vectors[-1], hidden_dim // 2, -1)
            diff = torch.bmm(private_vectors, torch.transpose(public_vectors, 1, 2))
            diff_loss = torch.sum(diff ** 2)
            diff_loss /= pooled_vectors.size(0)

            # adversarial prediction
            public_pooled, _ = torch.split(pooled_vectors, hidden_dim // 2, -1)
            language_logits = self.adversary_model(public_pooled)

            return token_logits, next_logits, language_logits, diff_loss

    def component_parameters(self, component):
        if component == 'adversary':
            return self.adversary_model.parameters()
        else:
            return chain(self.multilingual_model.language_parameters(component), self.mask_model.parameters(), self.next_model.parameters())


class AdversarialPretrainerConfig(object):
    def __init__(self, model_config, language_ids, adv_repeat=5, lr=1e-4, beta=1e-2, gamma=1e-4, with_cuda=True):
        self.model_config = model_config
        self.language_ids = language_ids
        self.adv_repeat = adv_repeat
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.with_cuda = with_cuda


class AdversarialPretrainer:
    """Adversarial pre-training on crosslingual BERT model for a set of languages
    """
    def __init__(self, multilingual_model, config: AdversarialPretrainerConfig, train_data, test_data=None):
        """
        :param multilingual_model: a multilingual sequence model which you want to train
        :param config: config of trainer containing parameters and total word vocab size
        :param train_data: a dictionary of dataloaders specifying train data
        :param test_data: a dictionary of dataloaders specifying test data, if none train_data is used instead
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and config.with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # initialize public, private, and adversarial discriminator
        self.ltoi = config.language_ids
        self.model = AdversarialBertWrapper(multilingual_model, config.model_config)
        self.model = self.model.to(self.device)

        # parallelize model
        if config.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for training" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # assign data
        self.train_data = train_data
        self.test_data = test_data if test_data else train_data
        
        # initialize loss function and optimizers
        self.D_repeat = config.adv_repeat
        self.criterion = nn.NLLLoss()
        self.mask_criterion = nn.NLLLoss(ignore_index=0)

        self.D_optim = Adam(self.model.component_parameters("adversary"), config.lr) # adversary optimizer
        self.lm_optims = {language: BERTAdam(self.model.component_parameters(language), config.lr)
            for language in config.language_ids} # optimizer for each language model

        # loss function hyperparameters
        self.beta = config.beta
        self.gamma = config.gamma

        self._config = config # for checkpointing

    def train(self, epoch):
        return self.iteration(epoch, self.train_data)

    def test(self, epoch):
        return self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"
        self.model = self.model.train() if train else self.model.eval()

        if train:
            for repeat in range(self.D_repeat):
                # for batch in batches
                D_iter = tqdm.tqdm(enumerate(data["adversary"]),
                        desc="D_train:{}:{}/{}".format(epoch, repeat+1, self.D_repeat),
                        total=len(data["adversary"]))

                total_loss = 0
                total_correct = 0
                total_elements = 0
                for i, batch in D_iter:
                    batch = {key: value.to(self.device) for key, value in batch.items()}
                    logits = self.model.forward("adversary", batch["input_ids"], attention_mask=batch['mask'])
                    loss = self.criterion(logits, batch['language_label']).to(self.device)
                    
                    total_loss += loss.detach().item()
                    total_correct += logits.argmax(-1).eq(batch['language_label']).sum().detach().item()
                    total_elements += batch['language_label'].detach().nelement()

                    if train:
                        self.D_optim.zero_grad()
                        loss.backward()
                        self.D_optim.step()

                print("loss={0:.6f} acc={1:.6f}".format(
                        total_loss / len(D_iter), total_correct / total_elements))

        micro_loss = 0
        language_iter = IterDict({language: data[language] for language in self.ltoi})
        language_iter = tqdm.tqdm(enumerate(language_iter),
                desc="{}:{}".format(str_code, epoch),
                total=len(language_iter))

        total_mask_loss = 0.
        total_next_loss = 0.
        total_adv_loss = 0.
        total_diff_loss = 0.
        total_loss = 0.

        total_mask_correct = 0
        total_mask_elements = 0
        total_next_correct = 0
        total_samples = 0
        for i, batches in language_iter:
            for language, language_label in self.ltoi.items():
                try:
                    batch = batches[language]
                except KeyError:
                    continue

                batch = {key: value.to(self.device) for key, value in batch.items()}
                mask_logits, next_logits, language_logits, diff_loss =\
                        self.model(language, batch['input_ids'], token_type_ids=batch['segment_label'], attention_mask=batch['mask'])

                mask_loss = self.mask_criterion(mask_logits.transpose(1,2), batch['token_labels']).to(self.device)
                next_loss = self.criterion(next_logits, batch['is_next']).to(self.device)
                language_labels = language_label + torch.zeros(language_logits.size(0), dtype=torch.long)
                adv_loss = -self.criterion(language_logits, language_labels.to(self.device)) # TODO correct loss

                train_loss = mask_loss + next_loss + self.beta * adv_loss + self.gamma * diff_loss
                if train:
                    self.lm_optims[language].zero_grad()
                    train_loss.backward()
                    self.lm_optims[language].step()

                total_loss += train_loss.detach().item()
                
                total_mask_loss += mask_loss.detach().item()
                total_next_loss += next_loss.detach().item()
                total_adv_loss += adv_loss.detach().item()
                total_diff_loss += diff_loss.detach().item()

                mask_correct = mask_logits.argmax(-1).eq(batch['token_labels']).sum().detach().item()
                mask_elements = (batch['token_labels'] > 0).sum().detach().item()
                total_mask_correct += (mask_correct & mask_elements)
                total_mask_elements += mask_elements

                total_next_correct += next_logits.argmax(dim=-1).eq(batch['is_next']).sum().detach().item()
                total_samples += batch['is_next'].detach().nelement()

        avg_loss = total_loss / total_samples
        avg_mask_loss = total_mask_loss / total_mask_elements
        avg_next_loss = total_next_loss / total_samples
        avg_adv_loss = self.beta * total_adv_loss / total_samples
        avg_diff_loss = self.gamma * total_diff_loss  / total_samples
        mask_acc = total_mask_correct / total_mask_elements
        next_acc = total_next_correct / total_samples

        print("EP{0}_{1}_{2}:\nmask={3:.6f}\tnext={4:.6f}\tadv={5:.6f}\ndiff={6:.6f}\tmask_acc={7:.6f}\tnext_acc={8:.6f}".format(
                epoch, language, str_code, avg_mask_loss, avg_next_loss, avg_adv_loss, avg_diff_loss, mask_acc, next_acc))

        return avg_loss

    def get_multilingual_model(self):
        return self.model.multilingual_model

    def save(self, epoch, file_path=None):
        """saving the current training state in file_path

        :param epoch: current epoch number
        :param directory_path: model output path which gonna be file_path+"ep%d" % epoch
        """
        if file_path is None:
            file_path = "epoch.%d.state" % epoch

        directory_path, file_name = os.path.split(file_path)

        if file_name == '':
            file_path = os.path.join(directory_path, "epoch.%d.state" % epoch)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # store optimizer state and model
        current_state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizers': {key: value.state_dict() for key, value in self.lm_optims.items()},
            'adv_optim': self.D_optim.state_dict(),
            'config': self._config # potentially convert to dict
        }
        torch.save(current_state, file_path)

        print("Epoch %d Model and Trainer Saved in:" % epoch, file_path)


    @classmethod
    def load_checkpoint(cls, save_path, arch, train_data, test_data=None):
        """loading a saved training and model state
        """
        save_state = torch.load(save_path)
        print("Restoring from epoch %d at" % save_state['epoch'], save_path)

        # initialize new trainer
        model = arch(save_state['config'].model_config)
        trainer = cls(model, save_state['config'], train_data, test_data)
        trainer.model.load_state_dict(save_state['model'])

        # replace model, move to appropriate device
        if save_state['config'].with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for training" % torch.cuda.device_count())
            trainer.model = nn.DataParallel(trainer.model, device_ids=cuda_devices)
        else:
            trainer.model = trainer.model.to(trainer.device)
        
        # restore optimer states
        trainer.D_optim.load_state_dict(save_state['adv_optim'])
        for key in trainer.lm_optims:
            trainer.lm_optims[key].load_state_dict(save_state['optimizers'][key])

        return trainer, save_state['epoch']
