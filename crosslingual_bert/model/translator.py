import copy
import torch
import torch.nn

from .bert_official import *


class PosWordEmbeddings(nn.Module):
	"""
	Combines positional encodings and word vector embeddings
	"""
	def __init__(self, config):
		super().__init__()
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

		# self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
		# any TensorFlow checkpoint file
		self.LayerNorm = BERTLayerNorm(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, input_ids):
		seq_length = input_ids.size(1)
		position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
		position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
		
		words_embeddings = self.word_embeddings(input_ids)
		position_embeddings = self.position_embeddings(position_ids)
		
		embeddings = words_embeddings + position_embeddings
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		return embeddings


class DecoderSelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (config.hidden_size, config.num_attention_heads))
		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, encoder_state, decoder_state, attention_mask):
		mixed_query_layer = self.query(decoder_state)
		mixed_key_layer = self.key(encoder_state)
		mixed_value_layer = self.value(encoder_state)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)

		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		attention_scores = attention_scores + attention_mask

		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		return context_layer


class DecoderAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.self = DecoderSelfAttention(config)
		self.output = BERTSelfOutput(config)

	def forward(self, encoder_tensor, hidden_tensor, attention_mask):
		self_output = self.self(encoder_tensor, hidden_tensor, attention_mask)
		attention_output = self.output(self_output, encoder_tensor)
		return attention_output


class DecoderLayer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.attention = DecoderAttention(config)
		self.intermediate = BERTIntermediate(config)
		self.output = BERTOutput(config)

	def forward(self, encoder_states, hidden_states, attention_mask):
		attention_output = self.attention(encoder_states, hidden_states, attention_mask)
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output, attention_output)
		return layer_output


class Decoder(nn.Module):
	def __init__(self, config):
		super().__init__()
		layer = DecoderLayer(config)
		self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])	

	def forward(self, encoder_states, hidden_states, attention_mask):
		all_encoder_layers = []
		for layer_module in self.layer:
			hidden_states = layer_module(encoder_states, hidden_states, attention_mask)
			all_encoder_layers.append(hidden_states)
		return all_encoder_layers


class MixtureOfExperts(nn.Module):
	def __init__(self, input_dim, output_dim, expert_count):
		self.linear = nn.Linear(input_dim, output_dim * expert_count)
		self.expert_gate = nn.Linear(input_dim, expert_count)
		self.output_dim = output_dim
		self.expert_count = expert_count

	def forward(self, input_vectors):
		logits = self.linear(input_vectors).reshape(-1, self.output_dim, self.input_dim)
		activations = nn.Softmax(dim=-1)(self.expert_gate(input_vectors)).unsqueeze(-1)
		return torch.bmm(logits, activations).squeeze()


class TranslatorModel(nn.Module):
	def __init__(self, config: BertConfig):
		super().__init__()
		self.decoder = Decoder(config)
		self.pooler = BERTPooler(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.linear = nn.Linear(config.hidden_size, config.vocab_size)

	def forward(self, encoder_vectors, decoder_vectors, encoder_mask=None, decoder_mask=None):
		if encoder_mask is None:
			encoder_mask = torch.ones(encoder_vectors.shape[:-1], dtype=torch.uint8)
		if decoder_mask is None:
			decoder_mask = torch.ones(decoder_vectors.shape[:-1], dtype=torch.uint8)
		
		# combine masks for each sequence into single attention mask matrix
		attention_mask = torch.bmm(decoder_mask.unsqueeze(2), encoder_mask.unsqueeze(1))
		attention_mask = attention_mask.unsqueeze(1).to(dtype=next(self.parameters()).dtype)
		attention_mask = (1.0 - attention_mask) * -10000.0

		all_decoder_layers = self.decoder(encoder_vectors, decoder_vectors, attention_mask)
		sequence_output = all_decoder_layers[-1]
		pooled_output = self.dropout(self.pooler(sequence_output))
		logits = nn.LogSoftmax(dim=-1)(self.linear(pooled_output))
		return logits

