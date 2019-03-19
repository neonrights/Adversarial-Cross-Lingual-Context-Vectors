import six
import copy
from itertools import chain
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerConfig, EncoderModel, DecoderModel, BertEmbeddings


class MultilingualConfig(TransformerConfig):
    def __init__(self, languages, *args, target_language=None, **kwargs):
        self.languages = languages
        self.target_language = target_language
        super().__init__(*args, **kwargs)

    @classmethod
    def from_dict(cls, json_object):
        config = MultilingualConfig(languages=None, vocab_size_or_config_json_file=-1)
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
        self.shared = EncoderModel(config)
        self.private = nn.ModuleDict({language: EncoderModel(config)
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


class MultilingualTranslator(nn.Module):
    """
    Universal to target language translation model using transformer architecture
    """
    def __init__(self, config: MultilingualConfig):
        super().__init__()
        self.config = config
        self.translator_model = DecoderModel(config)
        self.token_prediction = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_vectors, target_vectors, input_mask=None, target_mask=None):
        _, pooled_output = self.translator_model(input_vectors, target_vectors, input_mask, target_mask)
        token_scores = self.token_prediction(pooled_output)
        return token_scores

    def language_parameters(self, language):
        return chain(self.translator_model.parameters(), self.multilingual_model.language_parameters(language))
