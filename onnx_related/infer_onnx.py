import numpy as np
import pprint
import onnxruntime as ort
from tokenizers import Tokenizer 
from decoders import decoder_mapping

class TokenClassificationInference:

	def __init__(self, onnx_path, tokenizer_path, tags_path, masking_ids = [113, 108, 22, 128], scheme="io"):
		self.tokenizer = Tokenizer.from_file(tokenizer_path)
		self.ort_session = ort.InferenceSession(onnx_path)
		self.masking_ids = masking_ids
		with open(tags_path) as f:
			self.tag_names = [tag.strip() for tag in f]
		self.decoder = decoder_mapping[scheme]

	def __call__(self, text):
		tokens = self.tokenizer.encode(text)
		input_ids = np.array(tokens.ids, dtype=np.int64)
		do_mask = (input_ids.reshape(-1, 1) == self.masking_ids).any(axis=-1)
		attention_mask = np.array(tokens.attention_mask, dtype=np.int64)
		attention_mask[do_mask] = 0 # mask desired tokens
		spans = tokens.offsets
		# compute ONNX Runtime output prediction
		ort_inputs = {'input_ids': input_ids.reshape(1, -1), 'attention_mask': attention_mask.reshape(1, -1)}
		ort_outs = self.ort_session.run(None, ort_inputs)
		output = ort_outs[0][0].argmax(-1).tolist()
		return self.decoder(text, output, spans, self.tag_names)

if __name__ == '__main__':
	model = TokenClassificationInference('deployment/xlm-roberta-base/tuned-xlm-roberta-base-quantized.onnx', 
		'deployment/xlm-roberta-base/xlm-roberta_tokenizer.json', 'deployment/xlm-roberta-base/tags.txt', scheme="bio")
	text = 'The September 11 attacks, also commonly referred to as 9/11, were a series of four coordinated terrorist attacks by the militant Islamist terrorist group al-Qaeda against the United States on the morning of Tuesday, September 11, 2001. On that morning, four commercial airliners traveling from the northeastern U.S. to California were hijacked mid-flight by 19 al-Qaeda terrorists. The hijackers were organized into three groups of five hijackers and one group of four. Each group had one hijacker who had received flight training and took over control of the aircraft. Their explicit goal was to crash each plane into a prominent American building, causing mass casualties and partial or complete destruction of the targeted buildings. Two of the planes hit the Twin Towers of the World Trade Center, and a third hit the west side of the Pentagon. A fourth plane was intended to crash into a target in Washington, D.C., but instead crashed into a field near Shanksville, Pennsylvania, following a passenger revolt.'
	import time 
	t0 = time.perf_counter()
	for i in range(100):
	    result = model(text)
	dt = time.perf_counter() - t0
	pprint.pprint(result)
	print(f'Elapsed time: {dt} seconds..')

for text in texts:
	result = model(text)
	print(text)
	pprint.pprint(result)
