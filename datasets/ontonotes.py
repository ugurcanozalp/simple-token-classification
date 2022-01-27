
import json
from typing import Any, List
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random 

lenp1 = lambda x: len(x)+1

def augment_text(text, lower_prob=0.1, upper_prob=0.1):
	p = random.random()
	if p < lower_prob:
		return text.lower()
	elif p < lower_prob + upper_prob:
		return text.upper()
	else:
		return text

class Ontonotes(Dataset):

	def __init__(self, tokenizer: Any,
			tag_names: List[str],
			data_path: str,
			max_tokens = 256,
			masking_ids = [],
			phase="train",
			iszh = False
		):
		self.tokenizer = tokenizer
		self.masking_ids = np.array(masking_ids, dtype=np.int64)
		self.tag_names = tag_names
		self.max_tokens = max_tokens
		self.tag_to_idx = {tag: i for i, tag in enumerate(self.tag_names)}
		self.is_training = phase == "train"
		self.iszh = iszh
		with open(data_path,'r') as f:
			data_text = f.read()

		self.data = []
		for sentence in filter(lambda x: len(x)>2, data_text.split('\n\n')):
			sample = []
			for wordline in sentence.split('\n'):
				if wordline=='':
					continue
				word, label = wordline.split('\t')
				sample.append((word, label))
			self.data.append(sample)

	def _annotate(self, word_start_locs, word_lengths, word_labels, s, e):
		tag = 'O'
		if s == e:
			return self.tag_to_idx.get(tag)
		for ws, wlen, wlab in zip(word_start_locs, word_lengths, word_labels):
			if s == ws and e <= ws+wlen:
				tag = wlab
				break
			elif (s>=ws and e<=ws+wlen):
				tag = "I-" + wlab[2:] if wlab.startswith("B-") else wlab
				break
		return self.tag_to_idx.get(tag, 0)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		sample = self.data[i]
		words, word_labels = zip(*sample)
		if self.iszh:
			text = "".join(words)
			word_lengths = np.array(list(map(len, words)))
		else:
			text = " ".join(words)
			word_lengths = np.array(list(map(lenp1, words)))
		if self.is_training:
			text = augment_text(text)
		word_start_locs = np.cumsum(word_lengths) - word_lengths
		tokenized = self.tokenizer(text, return_offsets_mapping=True, padding='max_length', truncation=True, max_length=self.max_tokens)
		token_spans = tokenized.pop('offset_mapping')
		token_ids = tokenized.pop('input_ids')
		begin_mask = np.array([(se[0] in word_start_locs and se[1]-se[0]>0) for se in token_spans], dtype=np.bool)
		input_ids = np.array(token_ids, dtype=np.int64)
		do_mask = (input_ids.reshape(-1, 1) == self.masking_ids).any(axis=-1)
		attention_mask = np.array(tokenized.pop('attention_mask'), dtype=np.int64)
		attention_mask[do_mask] = 0 # mask desired tokens
		tags_list = [self._annotate(word_start_locs, word_lengths, word_labels, s, e) for (s, e) in token_spans]
		tags = np.array(tags_list, dtype=np.int64)
		return input_ids, attention_mask, begin_mask, tags

if __name__ == "__main__":
	import transformers
	tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
	with open("data/ontonotes/tags.txt") as f:
		tag_names = [tag.strip() for tag in f]

	data_file = "data/ontonotes/chinesedev.txt"
	ds = Ontonotes(tokenizer, tag_names, data_file, iszh=True)
	input_ids, attention_mask, begin_mask, tags = ds[0]

	for i, t, b, a in zip(input_ids, tags, begin_mask, attention_mask):
		print(f"{ds.tokenizer.convert_ids_to_tokens(int(i))} - {ds.tag_names[t]} - {a} - {b}")
