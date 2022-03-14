import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
	context = (
		[raw_text[i - j - 1] for j in range(CONTEXT_SIZE)] 
		+ [raw_text[i + j + 1] for j in range(CONTEXT_SIZE)]
	)
	target = raw_text[i]
	data.append((context, target))
# print(data[:5])

class CBOW(nn.Module):
	def __init__(self, vocab_size, embedding_dim, context_size):
		super(CBOW, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(embedding_dim * context_size * 2, 128)
		self.linear2 = nn.Linear(128, vocab_size)

	def forward(self, inputs):
		embeds = self.embeddings(inputs).view((1, -1))
		out = self.linear1(embeds)
		out = F.relu(out)
		out = self.linear2(out)
		log_probs = F.log_softmax(out, dim=1)
		return log_probs


def make_context_vector(context, word_to_ix):
	idxs = [word_to_ix[w] for w in context]
	return torch.tensor(idxs, dtype=torch.long)


model = CBOW(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []

for epoch in range(10):
	total_loss = 0
	for context, target in data:
		# Step 1
		context_vec = make_context_vector(context, word_to_ix)
		target_vec = make_context_vector([target], word_to_ix)

		# Step 2
		model.zero_grad()

		# Step 3
		log_probs = model(context_vec)

		# Step 4
		loss = loss_function(log_probs, target_vec)
		loss.backward()
		optimizer.step()

		# Step 5
		total_loss += loss.item()

	losses.append(total_loss)

print(losses)
# print(model.embeddings.weight[word_to_ix['study']])

with torch.no_grad():
	test_vec = make_context_vector(['are', 'We', 'to', 'study'], word_to_ix)
	log_probs = model(test_vec)

	prob, idx = log_probs[0].max(0)
	for w, ix in word_to_ix.items():
		if ix == idx:
			print(f'word: {w}, p: {prob}')
			break