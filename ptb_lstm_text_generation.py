import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext import data, datasets
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'> Using device: {DEVICE}')

TEXT = data.Field(lower=True, batch_first=True, eos_token='<eos>', pad_token=None)
train, val, test = datasets.PennTreebank.splits(TEXT)
TEXT.build_vocab(train)

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


def generate_text(model, text, next_n_words=100, ignore=['n', '$', '<unk>']):
	print(f'> Generating text with sequence: {text}\n')
	model.eval()

	words = text.split(' ')
	hidden1, hidden2 = None, None

	for i in range(next_n_words):
		x = torch.tensor([[TEXT.vocab.stoi[w] for w in words[i:]]])
		x = x.to(DEVICE)
		output, hidden1, hidden2 = model(x, hidden1, hidden2)

		last_word_probs = output[0][-1]
		p = nn.functional.softmax(last_word_probs, dim=0).detach().cpu().numpy()
		word_idx = np.random.choice(len(last_word_probs), p=p)

		while TEXT.vocab.itos[word_idx] in ignore:
			word_idx = np.random.choice(len(last_word_probs), p=p)

		words.append(TEXT.vocab.itos[word_idx])

	# print(' '.join(words).replace(' <eos> ', '.\n'))
	print(' '.join(words).replace(' <eos> ', '.\n').replace(" '", "'").replace(" n't", "n't"))


""" Set model
"""
PATH = os.path.join(os.path.dirname(__file__), '.checkpoint', 'checkpoint_e40.pth')

vocab_size = len(TEXT.vocab)
embedding_dim = 650
hidden_size = 650
dropout = 0.5

model = MyLSTM(vocab_size, embedding_dim, hidden_size, dropout).to(DEVICE)
model.load_state_dict(torch.load(PATH, map_location=DEVICE))


text = 'the meaning of life is'
generate_text(model, text, next_n_words=200)


""" Example
> Using device: cpu
> Generating text with sequence: the meaning of life is

the meaning of life is disciplined but it's nice only if we do direct print dollars says peter van macdonald of university and pennsylvania university.
in the 1970s it is in disarray overseas in the region.
the group joins its committee from directors of this insurance group.
white both corp. and william k. d. brown contributed to this article.
environmental centers are advertising is such a possibility of arias is the federal bureaucracy.
in the first missouri d.c. museum has been absurd from a community that hasn't been coastal with a very severe multiple of seeing the skills.
the extent of times is single-a-2 because and under the control of the coffee terminals that will constitute a live pickup in developing hundreds of patients to produce.
mr. thompson's move with some japanese company to do only with the likely grave of his changes.
the control into the advertising computer isn't going to display columbia he adds.
the u.s. achieved as a way for some of the domestic market.
in paris stocks were for technical the day.
in tokyo the nikkei index fell slightly points at between the two previous west
"""