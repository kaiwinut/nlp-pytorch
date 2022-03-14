import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

""" Tensors
"""

# x = torch.randn(2, 3, 4)

# print(x)
# print(x.view(2, 12))
# print(x.view(2, -1))

""" Computation Graphs
"""

# x = torch.tensor([1., 2., 3.], requires_grad=True)
# y = torch.tensor([4., 5., 6.], requires_grad=True)

# z = x + y

# print(z)
# print(z.grad_fn)

# s = z.sum()
# print(s)
# print(s.grad_fn)

# s.backward()
# print(x.grad)

""" Deep Learning with Pytorch
"""

# lin = nn.Linear(5, 3)
# data = torch.randn(2, 5)
# print(lin(data))
# print(F.relu(lin(data)))

# data = torch.randn(5)
# print(F.softmax(data, dim=0))
# print(F.softmax(data, dim=0).sum())

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

word_to_ix = {}
for sent, _ in data + test_data:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)

print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


class BoWClassifier(nn.Module):
	def __init__(self, num_labels, vocab_size):
		super(BoWClassifier, self).__init__() # Always do this in an nn.Module
		self.linear = nn.Linear(vocab_size, num_labels)

	def forward(self, bow_vec):
		return F.log_softmax(self.linear(bow_vec), dim=1)

def make_bow_vector(sentence, word_to_ix):
	vec = torch.zeros(len(word_to_ix))
	for word in sentence:
		vec[word_to_ix[word]] += 1
	return vec.view(1, -1)

def make_target(label, label_to_ix):
	return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

# for param in model.parameters():
# 	print(param)

# with torch.no_grad():
# 	sample = data[0]
# 	bow_vector = make_bow_vector(sample[0], word_to_ix)
# 	log_probs = model(bow_vector)
# 	print(log_probs)	

label_to_ix = {'SPANISH': 0, 'ENGLISH':1}

# Before training
print('> Before training')

with torch.no_grad():
	for instance, label in test_data:
		bow_vector = make_bow_vector(instance, word_to_ix)
		log_probs = model(bow_vector)
		print(log_probs)	

print(next(model.parameters())[:, word_to_ix['creo']])

# Training loop
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
	for instance, label in data:
		# Step 1. Clear the accumulated gradients
		model.zero_grad()

		# Step 2. Make the bow vector and target vector
		bow_vec = make_bow_vector(instance, word_to_ix)
		target = make_target(label, label_to_ix)

		# Step 3. Calculate probability
		log_probs = model(bow_vec)

		# Step 4. Compute loss, update parameters
		loss = loss_function(log_probs, target)
		loss.backward()
		optimizer.step()

# After training
print('> After training')

with torch.no_grad():
	for instance, label in test_data:
		bow_vector = make_bow_vector(instance, word_to_ix)
		log_probs = model(bow_vector)
		print(log_probs)	

print(next(model.parameters())[:, word_to_ix['creo']])
















