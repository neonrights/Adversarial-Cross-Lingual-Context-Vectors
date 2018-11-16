import os
import path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .optim_schedule import ScheduledOptim
from .adversarial_model import AdversarialPretrainingWrapper

import tqdm


def freeze(models):
    try:
        for model in models:
            for weight in model.parameters():
                weight.requires_grad = False
    except TypeError:
        for weight in model.parameters():
            weight.requires_grad = False

def thaw(models):
    try:
        for model in models:
            for weight in model.parameters():
                weight.requires_grad = True
    except TypeError:
        for weight in model.parameters():
            weight.requires_grad = True


class AdversarialPretrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, multilingual_model, discriminator, vocab_size: int, hidden_size, languages,
                 train_data, discriminator_data, discriminator_repeat, test_data, alpha, beta, gamma, with_cuda=True):
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
        self.model = AdversarialPretrainingWrapper(multilingual_model, discriminator, vocab_size, hidden_size)

        freeze(model.language_models + model.adversary_model)

        self.train_data = train_data
        self.test_data = test_data
        self.D_data = discriminator_data
        
        self.D_repeat = discriminator_repeat

        # initialize optimizer
        # initialize loss functions
        self.criterion = nn.NLLLoss()
        
        self.D_optim = None
        self.lm_optim = None

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

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
            thaw(self.model.adversary)
            for repeat in range(self.D_repeat):
                # for batch in batches
                D_iter = tqdm.tqdm(enumerate(self.D_data),
                        desc="D_train:{}:{}/{}".format(epoch, repeat, self.D_data),
                        total=len(self.D_data),
                        bar_format="{l_bar}{r_bar}")

                for i, batch in D_iter:
                    context_vectors = self.shared_model(batch[0].to(self.device))
                    logits = self.model.adversary_forward(batch[0].to(self.device))
                    loss = self.nll_loss(logits, batch[1].to(self.device))

                    self.D_optim.zero_grad()
                    loss.backward()
                    self.D_optim.step()

            freeze(self.model.adversary)
            thaw(self.public_model)

        for language, language_label in self.ltoi.items():
            language_iter = tqdm.tqdm(enumerate(data[language]),
                    desc="{}_{}:{}".format(language, str_code, epoch),
                    total=len(data[language]),
                    bat_format="{l_bar}{r_bar}")

            language_model = self.language_models[language]
            if train:
                thaw(language_model.private)

            total_mask_loss = 0
            total_next_loss = 0
            total_adv_loss = 0
            total_diff_loss = 0
            total_correct = 0
            total_elements = 0
            language_labels = language_label + torch.zeros(data.batch_size)
            for i, batch in language_iter:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                token_logits, next_logits, language_logits, diff_loss = language_model(batch['input_ids'], batch['segment_label'])

                mask_loss = self.criterion(token_logits, batch['token_labels'])
                next_loss = self.criterion(next_logits, batch['is_next'])
                adv_loss = -self.criterion(language_logits, language_labels) # TODO correct loss

                if train:
                    train_loss = mask_loss + next_sent_loss + self.beta * D_loss + self.gamma * self.diff_loss # unweighted sum for model comparison
                    self.lm_optim.zero_grad()
                    train_loss.backward()
                    self.lm_optim.step()

                total_mask_loss += mask_loss.item()
                total_next_loss += next_loss.item()
                total_adv_loss += adv_loss,item()
                total_diff_loss += diff_loss.item()
                total_correct += next_logits.argmax(dim=-1).eq(batch['is_next']).sum().item()
                total_elements += batch['is_next'].nelement()

                if i & self.log_freq == 0:
                    post_fix = {
                        "epoch": epoch,
                        "language": language,
                        "iter": i,
                        "mask_loss": total_mask_loss / (i + 1),
                        "next_loss": total_next_loss / (i + 1),
                        "adversary_loss": total_adv_loss / (i + 1),
                        "difference_loss": total_diff_loss / (i + 1),
                        "accuracy": total_correct / total_elements
                    }
                    data_iter.write(str(post_fix))

            avg_mask_loss = total_mask_loss / len(language_iter)
            avg_next_loss = total_next_loss / len(language_iter)
            avg_adv_loss = total_adv_loss / len(language_iter)
            avg_diff_loss = total_diff_loss / len(language_iter)
            avg_acc = total_correct / total
            print("EP{}_{}_{}, mask={} next={} adv={} diff={}".format(
                    epoch, language, str_code, avg_mask_loss, avg_next_loss, avg_adv_loss, avg_diff_loss, avg_acc))

            if train:
                freeze(language_model.private)

        if train:
            freeze(self.model.public_model)


    def save(self, epoch, directory_path="output/"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param directory_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        directory_path = path.join(directory_path, str(epoch))
        if not os.isdir(directory_path):
            os.mkdir(directory_path)

        raise NotImplementedError

        print("EP:%d Model Saved on:" % epoch, directory_path)
        return directory_path
