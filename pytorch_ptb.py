import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext import data, datasets


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

TEXT = data.Field(lower=True, batch_first=True, eos_token='<eos>', pad_token=None)
train, val, test = datasets.PennTreebank.splits(TEXT)
TEXT.build_vocab(train)

# print(TEXT.vocab.itos[34])
# print(TEXT.vocab.stoi['them'])


""" Define model
"""
class MyLSTM(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_size, dropout):
		super(MyLSTM, self).__init__()

		# Define architecture
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.drop1 = nn.Dropout(dropout)
		self.lstm1 = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
		self.drop2 = nn.Dropout(dropout)
		self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
		self.drop3 = nn.Dropout(dropout)
		self.fc = nn.Linear(hidden_size, vocab_size)

		# Initialize weight
		nn.init.normal_(self.embeddings.weight, std=0.01)
		nn.init.normal_(self.lstm1.weight_ih_l0, std=1/math.sqrt(embedding_dim))
		nn.init.normal_(self.lstm1.weight_hh_l0, std=1/math.sqrt(hidden_size))
		nn.init.zeros_(self.lstm1.bias_ih_l0)
		nn.init.zeros_(self.lstm1.bias_hh_l0)
		nn.init.normal_(self.lstm2.weight_ih_l0, std=1/math.sqrt(embedding_dim))
		nn.init.normal_(self.lstm2.weight_hh_l0, std=1/math.sqrt(hidden_size))
		nn.init.zeros_(self.lstm2.bias_ih_l0)
		nn.init.zeros_(self.lstm2.bias_hh_l0)

		# Tie weights
		# Linear layers have weights of (output_dim, input_dim) shape
		self.fc.weight = self.embeddings.weight
		nn.init.zeros_(self.fc.bias)

	def forward(self, inputs, hidden1_prev, hidden2_prev):
		embeds = self.drop1(self.embeddings(inputs))
		out, hidden1_next = self.lstm1(embeds, hidden1_prev)
		out = self.drop2(out)
		out, hidden2_next = self.lstm2(out, hidden2_prev)
		out = self.drop3(out)
		out = self.fc(out)
		return out, hidden1_next, hidden2_next


def eval_perplexity(model, dataset):
	total_loss = 0.0
	hidden1, hidden2 = None, None

	with torch.no_grad():
		model.eval()

		for data in dataset:
			x, t = data.text, data.target
			x, t = x.to(DEVICE), t.to(DEVICE)
			output, hidden1, hidden2 = model(x, hidden1, hidden2)
			loss = criterion(output.view(-1, vocab_size), t.view(-1))
			total_loss += loss.item()

		ppl = math.exp(total_loss / len(dataset))
		return ppl


""" Set hyperparameters
"""
max_epoch = 40
batch_size = 20
vocab_size = len(TEXT.vocab)
embedding_dim = 650
hidden_size = 650
dropout = 0.5
bptt_len = 35
lr = 20.0
max_norm = 0.25


model = MyLSTM(vocab_size, embedding_dim, hidden_size, dropout).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
train_set, val_set, test_set = data.BPTTIterator.splits(
		(train, val, test), batch_size=batch_size, bptt_len=bptt_len, device=DEVICE
	)
# print(len(train_set))


""" Training loop
"""
PATH = os.path.join(os.path.dirname(__file__), '.checkpoint')
os.makedirs(PATH, exist_ok=True)

total_loss = 0.0
count = 0
ppl_list = []
best_ppl = float('inf')
start_time = time.time()

for epoch in range(max_epoch):
	model.train()

	total_loss = 0.0
	hidden1, hidden2 = None, None

	for i, data in enumerate(train_set):
		x, t = data.text, data.target
		x, t = x.to(DEVICE), t.to(DEVICE)
		optimizer.zero_grad()

		output, hidden1, hidden2 = model(x, hidden1, hidden2)

		loss = criterion(output.view(-1, vocab_size), t.view(-1))
		total_loss += loss.item()
		count += 1
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), max_norm)
		optimizer.step()

		hidden1 = tuple(h.detach() for h in hidden1)
		hidden2 = tuple(h.detach() for h in hidden2)

		if i % 200 == 0:
			ppl = math.exp(total_loss / count)
			ppl_list.append(ppl)
			total_loss, count = 0.0, 0
			print(f'[epoch] {epoch+1} / {max_epoch} | iters {i+1} / {len(train_set)} | time {time.time() - start_time:.1f}[s] | perplexity {ppl:.2f}')

	val_ppl = eval_perplexity(model, val_set)
	if val_ppl < best_ppl:
		best_ppl = val_ppl
	else:
		lr /= 4.0
		for group in optimizer.param_groups:
			group['lr'] = lr

	print(f'\n[validation] perplexity {val_ppl:.2f} | best ppl {best_ppl:.2f}\n')
	
	# Save parameters
	checkpoint = f'checkpoint_e{epoch+1}'
	model_path = os.path.join(PATH, checkpoint, '.pth')
	torch.save(model.state_dict(), model_path) 


""" Plot results
"""
plt.plot(np.arange(len(ppl_list)), ppl_list)
plt.title('training perplexity')
plt.xlabel('iterations (x100)')
plt.ylabel('perplexity')
plt.ylim(0.0, 200.0)
plt.show()


""" Evaluation
"""
ppl = eval_perplexity(model, test_set)
print(f'\n[eval] perplexity {val_ppl:.2f} | best val ppl {best_ppl:.2f}\n')
