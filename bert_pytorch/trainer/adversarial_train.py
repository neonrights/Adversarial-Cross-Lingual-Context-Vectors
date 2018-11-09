import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import BERTLM, BERT
from .optim_schedule import ScheduledOptim

import tqdm


class AdversarialTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, public_model, private_models, discriminator, vocab_size: int,
                 train_data, discriminator_data, discriminator_repeat, test_data, with_cuda=True):
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
        self.public_model = public_model
        self.private_models = private_models
        self.D_model = discriminator

        self.languages = set(self.private_models.keys())

        # initialize optimizer
        # initialize loss functions
        self.train_data = train_data
        self.test_data = test_data
        self.D_data = discriminator_data
        
        self.D_repeat = discriminator_repeat
        self.D_loss_function = None
        self.adv_loss_function = None
        self.mask_loss_function = None
        self.next_loss_function = None

        self.D_optim = None
        self.lm_optim = None

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

        # TODO set up tqdm for pretty training bars
        if train:
            self.public_model.freeze()
            self.D_model.thaw()
            for repeat in range(self.D_repeat):
            	# for batch in batches
                D_iter = tqdm.tqdm(enumerate(self.D_data),
                        desc="D_train:{}:{}/{}".format(epoch, repeat, self.D_data),
                        total=len(self.D_data),
                        bar_format="{l_bar}{r_bar}")

            	for i, batch in D_iter:
                    context_vectors = self.public_model(batch[0].to(self.device))
                    logits = self.D_model(context_vectors)
                    loss = self.D_loss_function(logits, batch[1].to(self.device))
                    
                    self.D_optim.zero_grad()
                    loss.backward()
                    self.D_optim.step()

            self.D_model.freeze()
            self.public_model.thaw()

        for language in self.languages:
            language_iter = tqdm.tqdm(enumerate(data[language]),
                    desc="{}_{}:{}".format(language, str_code, epoch),
                    total=len(data[language]),
                    bat_format="{l_bar}{r_bar}")

            language_model = self.private_models[language]
            if train:
                language_model.thaw()

            total_loss = 0
            for i, batch in language_iter:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                context_vectors = torch.cat([self.public_model(batch["input_ids"]),
                        language_model(batch["input_ids"])], dim=-1)

                # calculate discriminator loss
                D_logits = self.D_model(context_vectors)
                D_loss = self.adv_loss_function(D_logits, language)

                # calculate token prediction loss
                mask_loss = self.mask_loss_function(context_vectors, batch["token_labels"])

                # calculate next_sent prediction loss
                next_sent_loss = self.next_loss_function(context_vectors, batch["is_next"])

                if train:
                    train_loss = None # perform weighted sum of values
                    self.lm_optim.zero_grad()
                    train_loss.backward()
                    self.lm_optim.step()

                total_loss += D_loss + mask_loss + next_sent_loss # unweighted sum for model comparison

            if train:
                language_model.freeze()

        if train:
            self.public_model.freeze()

        # calculate model metrics

        # if train:
        #	perform weighted sum of losses
        #	update public-private model
        # else:
        #	perform weighted sum of losses for comparison to training
        #	perform sum of losses for comparison to other models

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
