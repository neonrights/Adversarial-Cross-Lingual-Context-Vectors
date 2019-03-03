import os
import copy
import json
import six
import torch
import torch.nn as nn

from itertools import chain
from torch.optim import *
from torch.utils.data import DataLoader
from apex.parallel import DistributedDataParallel

from .optimization import *
from .utils import *

import tqdm


class AdversarialPretrainerConfig(object):
    def __init__(self,
                model_config,
                language_ids,
                adv_repeat=5,
                lr=1e-4,
                beta=1e-2,
                gamma=1e-4,
                with_cuda=True,
                train_freq=None,
                share_file=None,
                gpu_id=0):
        
        if model_config is not None:
            self.__dict__.update(model_config.__dict__) # add model configuration
        self.language_ids = language_ids
        self.adv_repeat = adv_repeat
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.with_cuda = with_cuda
        self.train_freq = train_freq
        self.share_file = share_file
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
        torch.nn.init.xavier_uniform_(self.linear.weight)

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
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.softmax(self.linear(x))


class SimpleAdversary(nn.Module):
    """simpled adversarial language predictor
    """
    def __init__(self, hidden_size, language_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, language_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        torch.nn.init.xavier_uniform_(self.linear.weight)

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

        # loss calculation
        self.criterion = nn.NLLLoss()
        self.mask_criterion = nn.NLLLoss(ignore_index=0)

    def forward(self,
                component,
                language_labels,
                input_ids,
                mask,
                segment_label=None,
                token_labels=None,
                is_next=None):
        if component == 'adversary':
            # return logits for adversarial language prediction
            embeddings = self.multilingual_model.embeddings(input_ids)
            _, pooled_vectors = self.multilingual_model.shared(embeddings, attention_mask=mask)
            language_logits = self.adversary_model(pooled_vectors)
            language_loss = self.criterion(language_logits, language_labels)
            return language_loss, language_logits
        else:
            hidden_vectors, pooled_vectors = self.multilingual_model(component, input_ids,
                    token_type_ids=segment_label, attention_mask=mask)
        
            # mask prediction loss
            token_logits = self.mask_model(hidden_vectors[-1])
            mask_loss = self.mask_criterion(token_logits.transpose(2,1), token_labels)
            
            # next sentence prediction loss
            next_logits = self.next_model(pooled_vectors)
            next_loss = self.criterion(next_logits, is_next)

            # adversarial prediction loss
            hidden_dim = hidden_vectors[-1].size(-1)
            public_pooled, _ = torch.split(pooled_vectors, hidden_dim // 2, -1)
            language_logits = self.adversary_model(public_pooled)
            adv_loss = -self.criterion(language_logits, language_labels)

            # public-private vector similarity loss
            public_vectors, private_vectors = torch.split(hidden_vectors[-1], hidden_dim // 2, -1)
            diff = torch.bmm(private_vectors, torch.transpose(public_vectors, 1, 2))
            diff_loss = torch.sum(diff ** 2) / pooled_vectors.size(0)

            # calculate accuracy statistics
            mask_correct = token_logits.argmax(-1).eq(token_labels)
            mask_elements = (token_labels > 0)
            mask_correct = (mask_correct & mask_elements).sum()
            mask_elements = mask_elements.sum()

            next_correct = next_logits.argmax(dim=-1).eq(is_next)
            next_acc = next_correct.sum().double() / is_next.nelement()

            return {
                "mask_loss": mask_loss,
                "next_loss": next_loss,
                "adv_loss": adv_loss,
                "diff_loss": diff_loss,
                "mask_correct": mask_correct.detach(),
                "mask_elements": mask_elements.detach(),
                "next_acc": next_acc.detach()
            }

    def component_parameters(self, component=None):
        if component == 'adversary':
            return self.adversary_model.parameters()
        elif component is None:
            return chain(self.multilingual_model.parameters(), self.mask_model.parameters(), self.next_model.parameters())
        else:
            return chain(self.multilingual_model.language_parameters(component), self.mask_model.parameters(), self.next_model.parameters())


class AdversarialPretrainer:
    """Adversarial pre-training on crosslingual BERT model for a set of languages
    """
    def __init__(self, multilingual_model, config: AdversarialPretrainerConfig, train_data, test_data=None, position=None, seed=None):
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
        self.model = AdversarialBertWrapper(multilingual_model, config)

        # move to GPU
        parallelize = cuda_condition and torch.cuda.device_count() > 1
        self.model.to(self.device)
        if parallelize:
            print("Using %d GPUS for training" % torch.cuda.device_count())
            gpu_ids = list(range(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model).to(self.device)

        # assign data
        self.train_data = train_data
        self.test_data = test_data if test_data else train_data
        
        # initialize loss function and optimizers
        self.D_repeat = config.adv_repeat

        # initialize optimizers
        if parallelize:
            self.D_optim = Adafactor(self.model.module.component_parameters("adversary"), config.lr)
            self.lm_optims = Adafactor(self.model.module.component_parameters(), config.lr)
        else:
            self.D_optim = Adafactor(self.model.component_parameters("adversary"), config.lr) # adversary optimizer
            self.lm_optims = Adafactor(self.model.component_parameters(), config.lr)

        # hyperparameters for loss
        self.beta = config.beta
        self.gamma = config.gamma

        # how many iterations to accumulate gradients for
        self.train_freq = config.train_freq if config.train_freq is not None else 1

        self._config = config # for checkpointing
        self.position = position
        self.seed = seed

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
        if train:
            self.model.train()
        else:
            self.model.eval()

        if train:
            for repeat in range(self.D_repeat):
                # for batch in batches
                D_iter = tqdm.tqdm(enumerate(data["adversary"]),
                        desc="D_train:{}:{}/{}".format(epoch, repeat+1, self.D_repeat),
                        total=len(data["adversary"]),
                        position=self.position)

                total_loss = 0
                total_correct = 0
                total_elements = 0
                for i, batch in D_iter:
                    batch = {key: value.to(self.device) for key, value in batch.items()}
                    loss, logits = self.model.forward("adversary", **batch)
                    loss = loss.mean()

                    total_loss += loss.item()
                    total_correct += logits.argmax(-1).eq(batch['language_labels']).sum().item()
                    total_elements += batch['language_labels'].detach().nelement()

                    if train:
                        self.D_optim.zero_grad()
                        loss.backward()
                        self.D_optim.step()
                
                print("loss={0:.6f} acc={1:.6f}".format(
                        total_loss / len(D_iter), total_correct / total_elements))

        micro_loss = 0
        if train:
            seed = self.seed * epoch if self.seed is not None else None
            language_iter = SmoothedRandomSampler({language: data[language] for language in self.ltoi}, seed=seed)
        else:
            language_iter = SequentialSampler({language: data[language] for language in self.ltoi})

        language_iter = tqdm.tqdm(enumerate(language_iter),
                desc="{}:{}".format(str_code, epoch),
                total=len(language_iter),
                position=self.position)

        total_mask_loss = 0.
        total_next_loss = 0.
        total_adv_loss = 0.
        total_diff_loss = 0.
        total_loss = 0.

        total_mask_correct = 0.
        total_mask_elements = 0.
        total_next_acc = 0.
        for i, (language, batch) in language_iter:
            # accumulate gradients if necessary
            if train and i % self.train_freq == 0:
                self.lm_optims.zero_grad()

            # compute losses for each subbatch
            batch['language_labels'] = self.ltoi[language] \
                     + torch.zeros(batch['input_ids'].size(0), dtype=torch.long)
            batch = {key: value.to(self.device) for key, value in batch.items()}

            # compute losses, sum/mean if parallelized
            output = self.model(language, **batch)
            output = {key: value.mean() if value.dtype is torch.float else value.sum()
                    for key, value in output.items()}
            loss = output["mask_loss"] + output["next_loss"] + self.beta * output["adv_loss"] + self.gamma * output["diff_loss"]

            if train:
                # average loss over accumulation and GPUs
                loss.backward()

            # record loss and accuracy statistics
            total_loss += loss.item()
            
            # individual loss statistics
            total_mask_loss += output["mask_loss"].item()
            total_next_loss += output["next_loss"].item()
            total_adv_loss += output["adv_loss"].item()
            total_diff_loss += output["diff_loss"].item()
            
            # batch accuracy statistics
            total_mask_correct += output["mask_correct"].item()
            total_mask_elements += output["mask_elements"].item()
            total_next_acc += output["next_acc"].item()

            if train and (i + 1) % self.train_freq == 0:
                self.lm_optims.step()

        # calculate avg loss and accuracy
        total_batches = len(language_iter)
        avg_loss = total_loss / total_batches
        avg_mask_loss = total_mask_loss / total_batches
        avg_next_loss = total_next_loss / total_batches
        avg_adv_loss = self.beta * total_adv_loss / total_batches
        avg_diff_loss = self.gamma * total_diff_loss  / total_batches
        avg_mask_acc = total_mask_correct / total_mask_elements
        avg_next_acc = total_next_acc / total_batches

        print("EP{0}_{1}:\nmask={2:.6f}\tnext={3:.6f}\tadv={4:.6f}\ndiff={5:.6f}\tmask_acc={6:.6f}\tnext_acc={7:.6f}".format(
                epoch, str_code, avg_mask_loss, avg_next_loss, avg_adv_loss, avg_diff_loss, avg_mask_acc, avg_next_acc))

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

        with open(os.path.join(directory_path, "config.json"), 'w+') as f:
            f.write(self._config.to_json_string())

        if isinstance(self.model, (nn.DataParallel, DistributedDataParallel)):
            model_state = self.model.module.cpu().state_dict()
            self.model.module.to(self.device)
        else:
            model_state = self.model.cpu().state_dict()
            self.model.to(self.device)

        # store optimizer state and model
        current_state = {
            'epoch': epoch,
            'model': model_state,
            'optimizer': self.lm_optims.state_dict(),
            'adv_optim': self.D_optim.state_dict()
        }
        torch.save(current_state, file_path)

        print("Epoch %d Model and Trainer Saved in:" % epoch, file_path)


    @classmethod
    def load_checkpoint(cls, checkpoint_folder, arch, train_data, test_data=None, position=0):
        """loading a saved training and model state
        """
        save_state = torch.load(os.path.join(checkpoint_folder, "checkpoint.state"))
        print("Restoring from epoch %d from %s" % (save_state['epoch'], checkpoint_folder))

        config = AdversarialPretrainerConfig.from_json_file(os.path.join(checkpoint_folder, "config.json"))

        # initialize new trainer
        model = arch(config)
        trainer = cls(model, config, train_data, test_data)
        if isinstance(trainer.model, (nn.DataParallel, DistributedDataParallel)):
            trainer.model.module.load_state_dict(save_state['model'])
        else:
            trainer.model.load_state_dict(save_state['model'])
        trainer.model.to(trainer.device)

        # restore optimer states
        trainer.D_optim.load_state_dict(save_state['adv_optim'])
        trainer.lm_optims.load_state_dict(save_state['optimizer'])

        return trainer, save_state['epoch']


class DistributedAdversarialPretrainer(AdversarialPretrainer):
    """Adversarial pre-training on crosslingual BERT model for a set of languages
    """
    def __init__(self, multilingual_model, config: AdversarialPretrainerConfig, train_data, test_data=None, position=None, seed=None):
        """
        :param multilingual_model: a multilingual sequence model which you want to train
        :param config: config of trainer containing parameters and total word vocab size
        :param train_data: a dictionary of dataloaders specifying train data
        :param test_data: a dictionary of dataloaders specifying test data, if none train_data is used instead
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        self.device = torch.device(config.gpu_id)

        # initialize public, private, and adversarial discriminator
        self.ltoi = config.language_ids
        self.model = AdversarialBertWrapper(multilingual_model, config)

        # move to GPU
        self.model.to(self.device)
        self.model = DistributedDataParallel(self.model, delay_allreduce=True)

        # assign data
        self.train_data = train_data
        self.test_data = test_data if test_data else train_data
        
        # initialize loss function and optimizers
        self.D_repeat = config.adv_repeat

        # initialize optimizers
        self.D_optim = Adafactor(self.model.module.component_parameters("adversary"), config.lr)
        self.lm_optims = Adafactor(self.model.module.component_parameters(), config.lr)

        # hyperparameters for loss
        self.beta = config.beta
        self.gamma = config.gamma

        # how many iterations to accumulate gradients for
        self.train_freq = config.train_freq if config.train_freq is not None else 1

        self._config = config # for checkpointing
        self.position = position
        self.seed = seed

