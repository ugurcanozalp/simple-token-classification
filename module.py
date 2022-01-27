
from argparse import ArgumentParser
from typing import List
import pprint

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import transformers

from metrics import TokenClassificationMetric
from decoders import decoder_mapping

class TokenClassification(pl.LightningModule):
	def __init__(self,
			tag_names: List[str] = ['O'],
			scheme = "io",
			arch: str = "roberta-base",
			masking_ids = [],
			learning_rate: float = 2e-5,
			weight_decay: float = 0,
			freeze_layers: int = None,
			pretrained_backbone = True,
			*args,
			**kwargs
		):

		super(TokenClassification, self).__init__()
		self.tag_names = tag_names
		self.save_hyperparameters("learning_rate", "weight_decay",
			"freeze_layers")
		self.metrics = {"basic": TokenClassificationMetric(self.tag_names, scheme=scheme)}
		# Construct the tokenizer
		self.tokenizer = transformers.AutoTokenizer.from_pretrained(arch, use_fast=True)
		self.masking_ids = torch.tensor(masking_ids, dtype=torch.long)
		self.decoder = decoder_mapping[scheme]
		# Construct the neural network here, with specific arch
		backbone_config = transformers.AutoConfig.from_pretrained(
			arch,
			num_labels=len(self.tag_names),
			id2label=self.tag_names,
			label2id={label: i for i, label in enumerate(self.tag_names)},
		)
		if pretrained_backbone:
			self.net = transformers.AutoModelForTokenClassification.from_pretrained(arch, config=backbone_config)
		else:
			self.net = transformers.AutoModelForTokenClassification.from_config(backbone_config)

		#self.net.classifier.bias.data[0] = 3.0 # bias towards O tag
		self.load_state_dict(torch.load("checkpoints/xlm-roberta-base.pt-v1.ckpt")["state_dict"])
		
		for param in self.net.roberta.embeddings.parameters():
			param.requires_grad = False

		for param in self.net.roberta.encoder.layer[:self.hparams.freeze_layers].parameters():
			param.requires_grad = False		



	def configure_optimizers(self):
		optimizer_grouped_parameters = [
			{
				"params": self.net.parameters(),
				"weight_decay_rate": 0,
				"lr": self.hparams.learning_rate,
			}
		]
		optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
		return optimizer

	@staticmethod
	def loss_fcn(preds, targets, mask):
		return nn.functional.cross_entropy(preds[mask], targets[mask])

	def forward(self, input_ids, attention_mask):
		o = self.net(input_ids, attention_mask)
		return o.logits

	def training_step(self, batch, batch_idx):
		input_ids, attention_mask, begin_mask, tags = batch
		logits = self.forward(input_ids, attention_mask)
		loss = self.loss_fcn(logits, tags, begin_mask)
		tensorboard_logs = {'train_batch_loss': loss}
		for metric, value in tensorboard_logs.items():
			self.log(metric, value, prog_bar=False)
		return {"loss": loss}

	def validation_step(self, batch, batch_idx):
		input_ids, attention_mask, begin_mask, tags = batch
		logits = self.forward(input_ids, attention_mask)
		loss = self.loss_fcn(logits, tags, begin_mask)
		preds = logits.argmax(-1)
		self.metrics["basic"].update(preds, tags, begin_mask)
		tensorboard_logs = {'val_batch_loss': loss}
		for metric, value in tensorboard_logs.items():
			self.log(metric, value, prog_bar=False)
		return {"loss": loss}

	def test_step(self, batch, batch_idx):
		input_ids, attention_mask, begin_mask, tags = batch
		logits = self.forward(input_ids, attention_mask)
		loss = self.loss_fcn(logits, tags, begin_mask)
		preds = logits.argmax(-1)
		self.metrics["basic"].update(preds, tags, begin_mask)
		tensorboard_logs = {'test_batch_loss': loss}
		for metric, value in tensorboard_logs.items():
			self.log(metric, value, prog_bar=False)
		return {"loss": loss}

	def validation_epoch_end(self, outputs):
		total_loss = sum(output["loss"].cpu().item() for output in outputs)
		self.log("total_val_loss", total_loss)
		metrics = self.metrics["basic"].compute()
		for name, value in metrics.items():
			self.log(name, value)
		self.metrics["basic"].reset()

	def test_epoch_end(self, outputs):
		total_loss = sum(output["loss"].cpu().item() for output in outputs)
		self.log("total_test_loss", total_loss)
		metrics = self.metrics["basic"].compute()
		for name, value in metrics.items():
			self.log(name, value)
		self.metrics["basic"].reset()

	@staticmethod
	def add_model_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--arch', type=str, default='roberta-base')
		parser.add_argument('--learning_rate', type=float, default=2e-5)
		parser.add_argument('--weight_decay', type=float, default=0)
		parser.add_argument('--freeze_layers', type=int, default=0)
		return parser

	@torch.no_grad()
	def predict(self, text: str):
		tokenized = self.tokenizer(text, return_offsets_mapping=True, padding='max_length', truncation=True)
		spans = tokenized.pop('offset_mapping')
		token_ids = tokenized.pop('input_ids')
		input_ids = torch.tensor(token_ids, dtype=torch.long)
		do_mask = (input_ids.unsqueeze(-1) == self.masking_ids).any(axis=-1)
		attention_mask = torch.tensor(tokenized.pop('attention_mask'), dtype=torch.long)
		attention_mask[do_mask] = 0 # mask desired tokens
		output = self.forward(input_ids.unsqueeze(0), attention_mask.unsqueeze(0)).argmax(-1)[0].tolist()
		return self.decoder(text, output, spans, self.tag_names)

if __name__=="__main__":
	sd = torch.load("checkpoints/xlm-roberta-base.pt-v1.ckpt", map_location="cpu")["state_dict"]
	with open("data/ontonotes/tags.txt") as f:
		tag_names = [tag.strip() for tag in f]
	model = TokenClassification(tag_names=tag_names, arch='xlm-roberta-base', scheme='bio', pretrained_backbone=False)
	model.load_state_dict(sd)
	model.eval()
	text = 'What is the word in the world?'
	text = """Kuzeydoğu eyaletlerinden kalkan ve Kaliforniya istikametine doğru ilerleyen dört yolcu uçağı, el-Kaide üyesi olan 19 kişi tarafından uçuş sırasında kaçırıldı. Hava korsanları, beşli üç grup ve dörtlü tek grup olarak organize edilmişti. Hedefini vuran ilk uçak American Airlines'ın 11 sefer sayılı uçuşuydu. Uçak saat 08.46'da Aşağı Manhattan'daki Dünya Ticaret Merkezi'nin kuzey kulesine çarptı. İlk saldırı, uçakta bulunan 92 kişinin tamamının ve saldırının etkilediği bölgedeki 1000'den fazla kişinin ölmesiyle sonuçlandı. 17 dakika sonrasında, saat 9.03'de, Ticaret Merkezi'nin güney kulesine United Airlines'ın 175 sefer sayılı uçuşu çarptı. İkinci saldırı, uçaktaki 65 kişinin tamamının ve etkilenen bölgede yer alan tahmini 1000'den fazla kişinin ölmesiyle sonuçlandı. İki saat içinde 110 katlı her iki bina da çökerken 7 Dünya Ticaret Merkezi'nin de arasında bulunduğu çevre yapıların bazısı yıkıldı, bazılarıysa hasar gördü. """
	result = model.predict(text)
	pprint.pprint(result)
