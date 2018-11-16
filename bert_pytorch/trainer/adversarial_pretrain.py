import os
import json

import torch
import torch.nn as nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader

from .adversarial_model import AdversarialPretrainingWrapper

import tqdm


def freeze(models):
    try:
        for model in models:
            for weight in model.parameters():
                weight.requires_grad = False
    except TypeError:
        for weight in models.parameters():
            weight.requires_grad = False

def thaw(models):
    try:
        for model in models:
            for weight in model.parameters():
                weight.requires_grad = True
    except TypeError:
        for weight in models.parameters():
            weight.requires_grad = True


class AdversarialPretrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, multilingual_model, adversarial_model, vocab_size: int, hidden_size, languages,
                 train_data, discriminator_data, test_data, discriminator_repeat=5, beta=1e-2, gamma=1e-3,
                 with_cuda=True, log_freq=100, log_file="training.log"):
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
        self.model = AdversarialPretrainingWrapper(multilingual_model, adversarial_model, hidden_size, vocab_size)

        freeze(self.model.pretraining_models + [self.model.adversary_model])

        self.model.adversary_model.to(self.device)
        for model in self.model.pretraining_models:
            model.to(self.device)

        # get data
        self.train_data = train_data
        self.test_data = test_data
        self.D_data = discriminator_data
        
        # initialize loss function and optimizers
        self.D_repeat = discriminator_repeat
        self.criterion = nn.NLLLoss()        
        self.D_optim = Adagrad(adversarial_model.parameters())
        self.lm_optim = Adagrad(param for model in self.model.pretraining_models for param in model.parameters())

        # loss function hyperparameters
        self.beta = beta
        self.gamma = gamma

        self.log_freq = log_freq
        self.log_file = log_file

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
            thaw(self.model.adversary_model)
            for repeat in range(self.D_repeat):
                # for batch in batches
                D_iter = tqdm.tqdm(enumerate(self.D_data),
                        desc="D_train:{}:{}/{}".format(epoch+1, repeat+1, self.D_repeat),
                        total=len(self.D_data),
                        bar_format="{l_bar}{r_bar}")

                total_loss = 0
                for i, batch in D_iter:
                    logits = self.model.adversary_forward(batch[0].to(self.device),
                            torch.zeros_like(batch[0], dtype=torch.long).to(self.device))
                    loss = self.criterion(logits, batch[1].to(self.device))

                    self.D_optim.zero_grad()
                    loss.backward()
                    self.D_optim.step()

                    total_loss += loss.item()

                print("EP{0}_D_{1}: loss={2:.6f}".format(epoch+1, repeat+1, total_loss / len(D_iter)))

            freeze(self.model.adversary_model)
            thaw(self.model.multilingual_model.public_model)

        for language, language_label in self.ltoi.items():
            language_iter = tqdm.tqdm(enumerate(data[language]),
                    desc="{}_{}:{}".format(language, str_code, epoch+1,
                    total=len(data[language]),
                    bar_format="{l_bar}{r_bar}"))

            language_model = self.model[language]
            if train:
                thaw(language_model.language_model.private)

            total_mask_loss = 0
            total_next_loss = 0
            total_adv_loss = 0
            total_diff_loss = 0
            total_correct = 0
            total_elements = 0
            for i, batch in language_iter:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                token_logits, next_logits, language_logits, diff_loss = language_model(batch['input_ids'], batch['segment_label'])

                mask_loss = self.criterion(token_logits.transpose(1,2), batch['token_labels'])
                next_loss = self.criterion(next_logits, batch['is_next'])
                language_labels = language_label + torch.zeros(language_logits.size(0), dtype=torch.long)
                adv_loss = -self.criterion(language_logits, language_labels.to(self.device)) # TODO correct loss

                if train:
                    train_loss = mask_loss + next_loss + self.beta * adv_loss + self.gamma * diff_loss # unweighted sum for model comparison
                    self.lm_optim.zero_grad()
                    train_loss.backward()
                    self.lm_optim.step()

                total_mask_loss += mask_loss.item()
                total_next_loss += next_loss.item()
                total_adv_loss += adv_loss.item()
                total_diff_loss += diff_loss.item()
                total_correct += next_logits.argmax(dim=-1).eq(batch['is_next']).sum().item()
                total_elements += batch['is_next'].nelement()

                if i % self.log_freq == 0:
                    post_fix = {
                        "epoch": epoch+1,
                        "language": language,
                        "iter": i+1,
                        "mask_loss": total_mask_loss / (i + 1),
                        "next_loss": total_next_loss / (i + 1),
                        "adversary_loss": total_adv_loss / (i + 1),
                        "difference_loss": total_diff_loss / (i + 1),
                        "accuracy": total_correct / total_elements
                    }
                    with open(self.log_file, 'a') as f:
                        f.write(json.dumps(post_fix) + '\n')


            avg_mask_loss = total_mask_loss / len(data[language])
            avg_next_loss = total_next_loss / len(data[language])
            avg_adv_loss = total_adv_loss / len(data[language])
            avg_diff_loss = total_diff_loss / len(data[language])
            avg_acc = total_correct / total_elements
            print("EP{0}_{1}_{2}, mask={3:.6f} next={4:.6f} adv={5:.6f} diff={6:.6f}".format(
                    epoch+1, language, str_code, avg_mask_loss, avg_next_loss, avg_adv_loss, avg_diff_loss, avg_acc))

            if train:
                freeze(language_model.language_model.private)

        if train:
            freeze(self.model.multilingual_model.public_model)


    def save(self, epoch, directory_path="output/"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param directory_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        directory_path = os.path.join(directory_path, str(epoch))
        if not os.isdir(directory_path):
            os.mkdir(directory_path)

        raise NotImplementedError

        print("EP:%d Model Saved on:" % epoch, directory_path)
        return directory_path
