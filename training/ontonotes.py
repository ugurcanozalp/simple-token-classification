
from argparse import ArgumentParser
import os
import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from module import TokenClassification
from datasets import Ontonotes, numpy_collate_fn

parser = ArgumentParser()
parser.add_argument('--test', action="store_true")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--tags_path', type=str, default="data/ontonotes/tags.txt")
parser.add_argument('--scheme', type=str, default='bio')
parser.add_argument('--ckpt', type=str, default="checkpoints/xlm-roberta-base.pt.ckpt")

parser = TokenClassification.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
dict_args = vars(args)
with open(args.tags_path) as f:
	tag_names = [tag.strip() for tag in f]

model = TokenClassification(tag_names=tag_names, masking_ids=[], **dict_args)
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
		Ontonotes(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/ontonotes/englishtrain.txt', phase="train"),
		Ontonotes(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/ontonotes/chinesetrain.txt', phase="train", iszh=True),
		Ontonotes(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/ontonotes/arabictrain.txt', phase="train"),
		Ontonotes(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/ontonotes/turkishtrain.txt', phase="train")
	]
)

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
	num_workers=0, collate_fn=numpy_collate_fn)

val_ds = ConcatDataset(
	[
		Ontonotes(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/ontonotes/englishdev.txt', phase="val"),
		Ontonotes(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/ontonotes/chinesedev.txt', phase="val", iszh=True),
		Ontonotes(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/ontonotes/arabicdev.txt', phase="val"),
		Ontonotes(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/ontonotes/turkishdev.txt', phase="val")
	]
)

val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
	num_workers=0, collate_fn=numpy_collate_fn)

test_ds = ConcatDataset(
	[
		Ontonotes(tokenizer=model.tokenizer,
			tag_names=model.tag_names, data_path='data/ontonotes/arabictrain.txt', phase="test")
	]
)

test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
	num_workers=0, collate_fn=numpy_collate_fn)

if args.test:
	model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state_dict"])
	trainer.test(model, test_dataloaders=[test_dl])
else:
	trainer.fit(model, train_dataloader=train_dl, val_dataloaders=[val_dl])

# python ontonotes.py --arch xlm-roberta-base --gradient_clip_val 1.0 --max_epochs 10 --min_epochs 3 --gpus 1 --freeze_layers 4
# python ontonotes.py --arch xlm-roberta-base --gradient_clip_val 1.0 --max_epochs 10 --min_epochs 3
