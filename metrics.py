
from typing import List

from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from seqeval.scheme import Token, Prefix, Tag, IOB2
import torch
from torchmetrics import Metric

class IO(Token):
	allowed_prefix = Prefix.I | Prefix.O
	start_patterns = {
		(Prefix.O, Prefix.I, Tag.ANY)
	}
	inside_patterns = {
		(Prefix.I, Prefix.I, Tag.SAME),
	}
	end_patterns = {
		(Prefix.I, Prefix.O, Tag.ANY),
		(Prefix.I, Prefix.I, Tag.DIFF),
	}

_scheme_mapping = {
	"io": IO,
	"bio": IOB2
}

class TokenClassificationMetric(Metric):
	def __init__(self, tag_names: List, scheme: str, dist_sync_on_step: bool = False):
		super().__init__(dist_sync_on_step=dist_sync_on_step)
		self.tag_names = tag_names
		self.tag_to_idx = {tag_name: i for i, tag_name in enumerate(self.tag_names)}
		self.scheme = _scheme_mapping[scheme]
		self.add_state("preds", default=[], dist_reduce_fx=None)
		self.add_state("targets", default=[], dist_reduce_fx=None)

	def update(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
		self.preds.append([self.tag_names[pred.item()] for pred in preds[mask]])
		self.targets.append([self.tag_names[target.item()] for target in targets[mask]])

	def compute(self):
		accuracy = accuracy_score(self.targets, self.preds)
		precision = precision_score(self.targets, self.preds, mode='strict', scheme=self.scheme, average='micro')
		recall = recall_score(self.targets, self.preds, mode='strict', scheme=self.scheme, average='micro')
		f1 = f1_score(self.targets, self.preds, mode='strict', scheme=self.scheme, average='micro')
		report = classification_report(self.targets, self.preds, scheme=self.scheme, mode='strict')
		conf = confusion_matrix(sum(self.targets, []), sum(self.preds, []), self.tag_names)
		print(report)
		print(conf)
		#return f1
		return {
			"accuracy": accuracy,
			"precision": precision,
			"recall": recall,
			"f1": f1,
		}
