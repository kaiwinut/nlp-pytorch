import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from collections import Counter
import argparse
import numpy as np
import pandas as pd


class Args:
	def __init__(self, seq_len=4, batch_size=256, max_epoch=10):
		self.seq_len = seq_len
		self.batch_size = batch_size
		self.max_epoch = max_epoch


class Model(nn.Module):
	def __init__(self, dataset):
		super(Model, self).__init__()

		embedding_dim = 128
		self.lstm_size = 128
		self.num_layers = 3

		vocab_size = len(dataset.vocab)

		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(
			input_size = embedding_dim, 
			hidden_size = self.lstm_size, 
			num_layers = self.num_layers, 
			dropout=0.2
			)
		self.fc = nn.Linear(self.lstm_size, vocab_size)

	def forward(self, inputs, hidden):
		embeds = self.embeddings(inputs)
		output, hidden = self.lstm(embeds, hidden)
		log_probs = self.fc(output)
		return log_probs, hidden

	def init_state(self, seq_len):
		return (torch.zeros(self.num_layers, seq_len, self.lstm_size),
				torch.zeros(self.num_layers, seq_len, self.lstm_size))


class JokesDataset(Dataset):
	def __init__(self, seq_len=4):
		self.seq_len = seq_len
		self.raw_text = self.load_text()
		self.vocab = self.get_vocab()
		self.word_to_ix = {word: i for i, word in enumerate(self.vocab)}
		self.ix_to_word = {i: word for i, word in enumerate(self.vocab)}
		self.corpus = [self.word_to_ix[w] for w in self.raw_text]

	def load_text(self):
		train_df = pd.read_csv('.data/jokes.csv')
		text = train_df['Joke'].str.cat(sep=' ')
		return text.split(' ')

	def get_vocab(self):
		word_count = Counter(self.raw_text)
		return sorted(word_count, key=word_count.get, reverse=True)

	def __getitem__(self, index):
		return (
			torch.tensor(self.corpus[index:index+self.seq_len]),
			torch.tensor(self.corpus[index+1:index+self.seq_len+1])
		)

	def __len__(self):
		return len(self.corpus) - self.seq_len


def train(dataset, model, args):
	model.train()

	seq_len = args.seq_len
	batch_size = args.batch_size
	max_epoch = args.max_epoch

	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	for epoch in range(max_epoch):
		state_h, state_c = model.init_state(seq_len)

		for batch, (x, y) in enumerate(dataloader):
			model.zero_grad()

			output, (state_h, state_c) = model(x, (state_h, state_c))
			loss = criterion(output.transpose(1, 2), y)

			state_h = state_h.detach()
			state_c = state_c.detach()

			loss.backward()
			optimizer.step()

			if batch % 10 == 0:
				print(f'[epoch] {epoch + 1} / {max_epoch} | batch {batch + 1} / {len(dataloader)} | loss {loss.item():.2f}')


def predict(dataset, model, text, next_n_words=100):
	model.eval()

	words = text.split(' ')
	state_h, state_c = model.init_state(len(words))

	for i in range(next_n_words):
		x = torch.tensor([[dataset.word_to_ix[w] for w in words[i:]]])
		output, (state_h, state_c) = model(x, (state_h, state_c))

		last_word_probs = output[0][-1]
		p = F.softmax(last_word_probs, dim=0).detach().numpy()
		word_idx = np.random.choice(len(last_word_probs), p=p)
		words.append(dataset.ix_to_word[word_idx])

	return words


args = Args()
dataset = JokesDataset(seq_len=4)
model = Model(dataset)
text = 'Knock knock. Whos there?'

train(dataset, model, args)
print(predict(dataset, model, text=text))