
from argparse import ArgumentParser
import os
import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from module import TokenClassification
from datasets import JsonIODataset, numpy_collate_fn

parser = ArgumentParser()
parser.add_argument('--test', action="store_true")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--tags_path', type=str, default="data/spider/tags.txt")
parser.add_argument('--scheme', type=str, default='io')
parser.add_argument('--ckpt', type=str, default="checkpoints/roberta-base.pt.ckpt")

parser = TokenClassification.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
dict_args = vars(args)
with open(args.tags_path) as f:
	tag_names = [tag.strip() for tag in f]

model = TokenClassification(tag_names=tag_names, masking_ids=[113, 108, 22, 128], **dict_args)
# define checkpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
	dirpath="./checkpoints",
	verbose=True,
	filename=args.arch+".pt",
	monitor="f1",
	save_top_k=1,
	save_weights_only=True,
	mode="max" # only pick max of `f1 score`
)

trainer = pl.Trainer(
	gpus=args.gpus,
	callbacks=[checkpoint_callback],
	gradient_clip_val=args.gradient_clip_val)

train_ds = ConcatDataset(
	[
		JsonIODataset(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/spider/train_spider.json'),
		JsonIODataset(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/spider/train_others.json'),
		JsonIODataset(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/spider/quote_a.json'),
		JsonIODataset(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/spider/custom.json')
	]
)

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
	num_workers=0, collate_fn=numpy_collate_fn)

val_ds = ConcatDataset(
	[
		JsonIODataset(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/spider/dev.json'),
		JsonIODataset(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/spider/quote_b.json')
	]
)

val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
	num_workers=0, collate_fn=numpy_collate_fn)

if args.test:
	model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state_dict"])
	trainer.test(model, test_dataloaders=[val_dl])
else:
	trainer.fit(model, train_dataloader=train_dl, val_dataloaders=[val_dl])

# python main.py --gradient_clip_val 1.0 --max_epochs 10 --min_epochs 3 --gpus 1
# python main.py --gradient_clip_val 1.0 --max_epochs 10 --min_epochs 3
