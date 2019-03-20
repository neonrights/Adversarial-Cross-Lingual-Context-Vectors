import os
import six
import json
import copy
from os import path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .optimization import Adafactor
from .utils import *

import tqdm


class TranslatorTrainerConfig(object):
    def __init__(self,
                model_config,
                lr=None,
                with_cuda=True,
                train_freq=None,
                share_file=None,
                gpu_id=0):
        
        if model_config is not None:
            self.__dict__.update(model_config.__dict__) # add model configuration
        self.lr = lr
        self.with_cuda = with_cuda
        self.train_freq = train_freq
        self.gpu_id = gpu_id

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = AdversarialPretrainerConfig(model_config=None, language_ids=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TranslatorTrainerWrapper(nn.Module):
    def __init__(self, axlm_model, translator_model, config):
        super().__init__()
        self.axlm_model = axlm_model
        self.translator_model = translator_model
        self.target_language = config.target_language
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, language, input_ids, target_ids, input_mask, target_mask, token_labels):
        with torch.no_grad():
            input_vectors, _ = self.axlm_model(language, input_ids, attention_mask=input_mask)
            target_vectors, _ = self.axlm_model(self.target_language, target_ids, attention_mask=target_mask)
        token_scores = self.translator_model(input_vectors[-1], target_vectors[-1], input_mask, target_mask)
        token_loss = self.criterion(token_scores, token_labels)
        correct = token_scores.argmax(-1).eq(token_labels).sum().item()
        count = token_labels.nelement()
        return token_loss, correct, count


class TranslatorTrainer:
    def __init__(self, axlm_model, translator_model, config, train_data, test_data=None, with_cuda=True):
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.model = TranslatorTrainerWrapper(axlm_model, translator_model, config)
        self.languages = config.languages
        self.target_language = config.target_language
        
        # Distributed GPU training if CUDA can detect more than 1 GPU
        self.model.to(self.device)
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_data
        self.test_data = train_data if test_data is None else test_data

        # Setting the Adam optimizer with hyper-param
        self.optimizer = Adafactor(self.model.parameters(), config.lr)
        self.config = config
        self.train_freq = config.train_freq if config.train_freq is not None else 1

    def train(self, epoch):
        return self.iteration(epoch, self.train_data)

    def test(self, epoch):
        return self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every epoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"
        if train:
            self.model.train()
            language_iter = SmoothedRandomSampler({language: data[language] for language in self.languages})
        else:
            self.model.eval()
            language_iter = SequentialSampler({language: data[language] for language in self.languages})

        # Setting the tqdm progress bar
        language_iter = tqdm.tqdm(enumerate(language_iter),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(language_iter))

        total_loss = 0.
        total_correct = 0
        total_element = 0
        for i, (language, batch) in language_iter:
                # send data to cpu or gpu depending on settings
                batch = {key: value.to(self.device) for key, value in batch.items()}

                # calculate loss for specific language
                loss, correct, count = self.model(language, **batch)

                # 3. backward and optimization only in train
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # next sentence prediction accuracy
                total_loss += loss.item()
                total_correct += correct
                total_element += count

        avg_loss = total_loss / total_element
        print("EP%d_%s, avg_loss=%.6f, total_acc=%.6f" %
                (epoch, str_code, avg_loss, total_correct / total_element))
        return avg_loss

    def get_translator_model(self):
        return self.model.translator_model

    def save(self, epoch, file_path=None):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        if file_path is None:
            file_path = "epoch.%d.state" % epoch

        directory_path, file_name = path.split(file_path)

        if file_name == '':
            file_path = path.join(directory_path, "epoch.%d.state" % epoch)

        if not path.exists(directory_path):
            os.makedirs(directory_path)

        with open(path.join(directory_path, "config.json"), 'w+') as f:
            f.write(self.config.to_json_string())

        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.cpu().state_dict()
            self.model.module.to(self.device)
        else:
            model_state = self.model.cpu().state_dict()
            self.model.to(self.device)

        # store optimizer state and model
        current_state = {
            'name': 'translator_trainer',
            'epoch': epoch,
            'model': model_state,
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(current_state, file_path)

        print("Epoch %d Model and Trainer Saved in:" % epoch, file_path)


    @classmethod
    def load_checkpoint(cls, checkpoint_folder, arch, train_data, test_data=None):
        save_state = torch.load(path.join(checkpoint_folder, "checkpoint.state"))
        print("Restoring from epoch %d from %s" % (save_state['epoch'], checkpoint_folder))

        config = TranslatorTrainerConfig.from_json_file(path.join(checkpoint_folder, "config.json"))

        # initialize new trainer
        model = arch(config)
        trainer = cls(model, config, train_data, test_data)
        if isinstance(trainer.model, nn.DataParallel):
            trainer.model.module.load_state_dict(save_state['model'])
        else:
            trainer.model.load_state_dict(save_state['model'])
        trainer.model.to(trainer.device)

        # restore optimer states
        trainer.optimizer.load_state_dict(save_state['optimizer'])
        
        return trainer, save_state['epoch']
