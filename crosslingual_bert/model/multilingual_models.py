import copy
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert import BertConfig, BertEmbeddings, NoEmbeddingBertModel, BertModel
#from .translator import TranslatorModel


class MultilingualConfig(BertConfig):
    def __init__(self, languages, *args, **kwargs):
        self.languages = languages
        super().__init__(*args, **kwargs)

    @classmethod
    def from_dict(cls, json_object):
        config = MultilingualBertConfig(languages=None, vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config


class MultilingualBert(nn.Module):
    """Cross-lingual context vector model using transformer architecture
    """
    def __init__(self, config: MultilingualConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.shared = NoEmbeddingBertModel(config)
        self.private = nn.ModuleDict({language: NoEmbeddingBertModel(config)
                for language in config.languages})

    def forward(self, language, input_ids, token_type_ids=None, attention_mask=None):
        assert language in self.private
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # embeddings shared across all languages
        embeddings = self.embeddings(input_ids, token_type_ids)

        shared_vectors, shared_pooled = self.shared(embeddings, attention_mask)
        private_vectors, private_pooled = self.private[language](embeddings, attention_mask)
        hidden_vectors = [torch.cat((sv, pv), -1) for sv, pv in zip(shared_vectors, private_vectors)]
        pooled_output = torch.cat((shared_pooled, private_pooled), -1)
        return hidden_vectors, pooled_output

    def language_parameters(self, language):
        """Returns all parameters for a specific language model
        """
        assert language in self.private
        return chain(self.shared.parameters(),
            self.private[language].parameters(),
            self.embeddings.parameters())

    @classmethod
    def from_pretrained_bert(cls, languages, *args, noise_func=None, **kwargs):
        pretrained_bert = BertModel.from_pretrained(*args, **kwargs)
        pretrained_state = pretrained_bert.state_dict()

        # convert bert config into multilingual config
        config = copy.deepcopy(pretrained_bert.config)
        config.languages = languages
        model = cls(config)

        bert_state = model.state_dict().copy()
        for key in bert_state:
            if key.startswith('embeddings.'):
                bert_state[key] = pretrained_state[key]
                continue
            elif key.startswith('shared.'):
                new_key = '.'.join(key.split('.')[1:])
            elif key.startswith('private.'):
                new_key = '.'.join(key.split('.')[2:])
            else:
                raise ValueError("missing key corresponding to %s" % key)
            new_weight = pretrained_state[new_key]
            if add_noise is not None:
                new_weight += noise_func(pretrained_state[new_key])
            bert_state[key] = new_weight

        model.load_state_dict(bert_state)
        return model


"""
class MultilingualTranslator(nn.Module):
    Universal to target language translation model using transformer architecture
    def __init__(self, model: MultilingualBert, target_language: str, config: MultilingualConfig):
        assert target_language in model.private
        super().__init__()
        self.multilingual_model = model
        
        # double intermediate and hidden size to account for shared-private model
        config = copy.copy(config)
        config.intermediate_size *= 2
        config.hidden_size *= 2

        self.translator_model = TranslatorModel(config)
        self.target_language = target_language

    def forward(self, language, input_ids, target_ids, input_mask=None, target_mask=None):
        language_vectors, _ = self.multilingual_model(language, input_ids, attention_mask=input_mask)
        target_vectors, _ = self.multilingual_model(self.target_language, target_ids, attention_mask=target_mask)
        return self.translator_model(language_vectors[-1], target_vectors[-1], input_mask, target_mask)

    def language_parameters(self, language):
        return chain(self.translator_model.parameters(), self.multilingual_model.language_parameters(language))
"""
