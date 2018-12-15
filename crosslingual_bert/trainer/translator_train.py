import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import chain

from .optimization import BERTAdam
from .utils import *

import tqdm
import pdb


class TranslatorTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, translator_model, languages, target_language, train_data, test_data, with_cuda=True, lr=1e-4, log_freq=10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.model = translator_model
        self.languages = languages
        self.target_language = target_language

        for translator in self.model.language_translators:
            translator.to(self.device)
        # Distributed GPU training if CUDA can detect more than 1 GPU
        """if with_cuda and torch.cuda.device_count() > 1:
                print("Using %d GPUS for BERT" % torch.cuda.device_count())
                self.model = nn.DataParallel(self.model, device_ids=cuda_devices)"""

        # Setting the train and test data loader
        self.train_data = train_data
        self.test_data = test_data

        # Setting the Adam optimizer with hyper-param
        self.optim = BERTAdam(self.model.parameters(), lr=lr)

        # Using Negative Log Likelihood Loss function
        self.criterion = nn.NLLLoss()

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

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

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(iter_dict(data)),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=min(len(v) for v in data.values()))

        avg_loss = 0.
        total_correct = 0
        total_element = 0
        for i, batches in data_iter:
            for language in self.languages:
                translator = self.model[language]
                batch = batches[language]

                # send data to cpu or gpu depending on settings
                batch = {key: value.to(self.device) for key, value in batch.items()}

                # calculate loss for specific language
                predictions = translator(batch['input_ids'], batch['target_ids'], batch['input_mask'], batch['target_mask'])
                loss = self.criterion(predictions, batch['labels'])

                # 3. backward and optimization only in train
                if train:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                # next sentence prediction accuracy
                correct = predictions.argmax(dim=-1).eq(batch["labels"]).sum().item()
                avg_loss += loss.item()
                total_correct += correct
                total_element += batch["labels"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        raise NotImplementedError
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
