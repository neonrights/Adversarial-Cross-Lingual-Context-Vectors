import os
import json
from itertools import chain

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .optimization import BERTAdam

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
    def __init__(self, multilingual_model, language_size, config):
        super().__init__()
        self.multilingual_model = multilingual_model
        self.adversary_model = SimpleAdversary(config.hidden_size, language_size)
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


class AdversarialPretrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, multilingual_model, config, languages, train_data, test_data,
                adv_repeat=5, lr=1e-4, beta=1e-2, gamma=1e-4, with_cuda=True, log_freq=10):
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
        self.model = AdversarialBertWrapper(multilingual_model, len(languages), config)
        self.model.to(self.device)

        # get data
        self.train_data = train_data
        self.test_data = test_data
        
        # initialize loss function and optimizers
        self.D_repeat = adv_repeat
        self.criterion = nn.NLLLoss()   
        self.mask_criterion = nn.NLLLoss(ignore_index=0)     
        self.D_optim = Adam(self.model.component_parameters("adversary"), lr)
        self.lm_optims = {language: BERTAdam(self.model.component_parameters(language), lr) for language in languages}

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
                logits = self.model.forward("adversary", batch["input_ids"], attention_mask=batch['mask'])
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

        micro_loss = 0
        for language, language_label in self.ltoi.items():
            language_iter = tqdm.tqdm(enumerate(data[language]),
                    desc="{}_{}:{}".format(language, str_code, epoch+1,
                    total=len(data[language])))

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
                mask_logits, next_logits, language_logits, diff_loss = self.model(language, batch['input_ids'], batch['segment_label'], batch['mask'])

                mask_loss = self.mask_criterion(mask_logits.transpose(1,2), batch['token_labels'])
                next_loss = self.criterion(next_logits, batch['is_next'])
                language_labels = language_label + torch.zeros(language_logits.size(0), dtype=torch.long)
                adv_loss = -self.criterion(language_logits, language_labels.to(self.device)) # TODO correct loss

                train_loss = mask_loss + next_loss + self.beta * adv_loss + self.gamma * diff_loss
                if train:
                    self.lm_optims[language].zero_grad()
                    train_loss.backward()
                    self.lm_optims[language].step()

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

        torch.save(self.model.cpu(), os.path.join(directory_path, "epoch%d.model" % epoch))

        print("EP:%d Model Saved in:" % epoch, directory_path)
        return directory_path
