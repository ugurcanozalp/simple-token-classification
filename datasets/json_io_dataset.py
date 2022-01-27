
import json
from typing import Any, List

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from nltk.tokenize import RegexpTokenizer

word_tokenizer = RegexpTokenizer(
		r'''(?x)          # set flag to allow verbose regexps
			(?:[A-Z]\.)+        # abbreviations, e.g. U.S.A. or U.S.A #
		  | (?:\d+\.)           # numbers
		  | \w+(?:[-.]\w+)*     # words with optional internal hyphens
		  | \$?\d+(?:.\d+)?%?   # currency and percentages, e.g. $12.40, 82%
		  | \.\.\.              # ellipsis, and special chars below, includes ], [
		  | [-\]\[.,;"'?():_`“”/°º‘’″…#$%()*+<>=@\\^_{}|~❑&§]
		'''
		)

class JsonIODataset(Dataset):

	def __init__(self, tokenizer: Any,
		tag_names: List[str],
		data_path: str,
		max_tokens = 128,
		masking_ids = [113, 108, 22, 128]):
		self.tokenizer = tokenizer
		self.masking_ids = np.array(masking_ids, dtype=np.int64)
		self.tag_names = tag_names
		self.max_tokens = max_tokens
		self.tag_to_idx = {tag: i for i, tag in enumerate(self.tag_names)}
		with open(data_path) as f:
			self.data = json.load(f)

	def _annotate(self, spans, s, e):
		tag = 'O'
		if s == e:
			return self.tag_to_idx.get(tag)
		for span in spans:
			if (s>=span['found'][0] and e<=span['found'][1]):
				tag = 'I-' + span['type']
				break
		return self.tag_to_idx.get(tag)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		sample = self.data[i]
		word_start_locs, _ = zip(*word_tokenizer.span_tokenize(sample['utterance']))
		tokenized = self.tokenizer(sample['utterance'], return_offsets_mapping=True, padding='max_length', truncation=True, max_length=self.max_tokens)
		token_spans = tokenized.pop('offset_mapping')
		token_ids = tokenized.pop('input_ids')
		begin_mask = np.array([(se[0] in word_start_locs and se[1]-se[0]>0) for se in token_spans], dtype=np.bool)
		input_ids = np.array(token_ids, dtype=np.int64)
		do_mask = (input_ids.reshape(-1, 1) == self.masking_ids).any(axis=-1)
		attention_mask = np.array(tokenized.pop('attention_mask'), dtype=np.int64)
		attention_mask[do_mask] = 0 # mask desired tokens
		tags_list = [self._annotate(sample['spans'], s, e) for (s, e) in token_spans]
		tags = np.array(tags_list, dtype=np.int64)
		return input_ids, attention_mask, begin_mask, tags

if __name__ == "__main__":
	import transformers
	tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
	with open("data/sss/tags.txt") as f:
		tag_names = [tag.strip() for tag in f]

	data_file = "data/sss/train_sss.json"
	ds = JsonIODataset(tokenizer, tag_names, data_file)
	input_ids, attention_mask, begin_mask, tags = ds[0]
