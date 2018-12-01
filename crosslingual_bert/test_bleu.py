import torch
from numpy.testing import assert_almost_equal

from evaluate import batch_BLEU

print("Running unittests for batch BLEU calculation")

reference = torch.tensor([[1, 2, 3, 4]])
candidate = torch.tensor([[1, 2, 4]])
score = batch_BLEU(candidate, reference, weights=[1])[0].item()

assert_almost_equal(score, 0.7165313105737893)
print("passed unigram test")

reference = torch.tensor([[1, 2, 3, 4, 0, 0]])
ref_mask = torch.tensor([[0, 0, 0, 0, 1, 1]])
candidate = torch.tensor([[1, 2, 4, 0, 0, 0]])
can_mask = torch.tensor([[0, 0, 0, 1, 1 ,1]])
score = batch_BLEU(candidate, reference, can_mask, ref_mask, weights=[1])[0].item()

assert_almost_equal(score, 0.7165313105737893)
print("passed masked unigram test")

reference = torch.tensor([[0, 1, 2, 3, 4, 0, 0, 0, 0, 0],
						  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
ref_mask = torch.tensor([[1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
						 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
candidate = torch.tensor([[0, 1, 2, 4, 0, 0, 0, 0, 0, 0],
						  [0, 1, 0, 3, 4, 5, 6, 7, 0, 9]])
can_mask = torch.tensor([[1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
						 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

scores = batch_BLEU(candidate, reference, can_mask, ref_mask)
assert_almost_equal(scores[0].item(), 0.)
assert_almost_equal(scores[1].item(), 0.48549177170732344)
print("passed quadgram batch test")

