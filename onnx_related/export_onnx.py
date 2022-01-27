from argparse import ArgumentParser
import json
import os
from tqdm import tqdm
from shutil import copyfile

import pytorch_lightning as pl
import torch

from module import TokenClassification

parser = ArgumentParser()
parser.add_argument('--arch', type=str, default='roberta-base')
parser.add_argument('--ckpt', type=str, default="checkpoints/roberta-base.pt.ckpt")
parser.add_argument('--tags_path', type=str, default="data/spider/tags.txt")
parser.add_argument('--quantize', action='store_true')

args = parser.parse_args()
with open(args.tags_path) as f:
    tag_names = [tag.strip() for tag in f]

model = TokenClassification(tag_names = tag_names, arch = args.arch)

sd = torch.load(args.ckpt, map_location="cpu")
model.load_state_dict(sd["state_dict"])
model.eval()

text = "How many singers do we have?"
input_batches = model.tokenizer(text, return_offsets_mapping=True, return_tensors='pt')
inp = (input_batches['input_ids'], input_batches['attention_mask'])

deployment_path = os.path.join("deployment", args.arch)
os.mkdir(deployment_path)
onnx_path = os.path.join(deployment_path, "tuned-"+args.arch+".onnx")
model.to_onnx(
    onnx_path, inp, export_params=True,
    opset_version=11,
    input_names = ['input_ids', 'attention_mask'],   # the model's input names
    output_names = ['logits'], # the model's output names
    dynamic_axes={'input_ids' : {0 : 'batch_size', 1: 'sequence_size'}, 'attention_mask': {0 : 'batch_size', 1: 'sequence_size'}, 'logits' : {0 : 'batch_size', 1: 'sequence_size'}}
)

if args.quantize:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quant_path = os.path.join(deployment_path, "tuned-"+args.arch+"-quantized.onnx")
    quantized_model = quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QUInt8)

tags_path = os.path.join(deployment_path, "tags.txt")
with open(tags_path, "w") as f:
    for tag in tag_names:
        f.write(tag+'\n')