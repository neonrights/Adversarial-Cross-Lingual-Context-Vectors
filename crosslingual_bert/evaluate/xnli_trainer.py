import torch

import tqdm


class XNLITrainer:
	def __init__(self, model, train_data, test_data, with_cuda=True):

		cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

		# wrap model to produce prediction tasks
		self.model = model
		# load train and test data
		self.train_data = train_data
		self.test_data = test_data

	def train(self, epoch):
		return self.iteration(epoch, self.train_data)

	def test(self, epoch):
		return self.iteration(epoch, self.test_data, train=False)

	def iteration(self, epoch, data_loader, train=True):
		# optimization and prediction
        str_code = "train" if train else "test"
		data_iter = tqdm.tqdm(enumerate(data_loader),
							  desc="EP_{}:{}".format(str_code),
							  total=len(data_loader))

		for i, batch in data_iter:
			batch = {key: value.to(self.device)}
			# forward iteration with mixture of experts