import torch 

def numpy_collate_fn(batch):
	input_ids, attention_mask, begin_mask, tags = zip(*batch)
	input_ids = torch.tensor(input_ids).contiguous()
	attention_mask = torch.tensor(attention_mask).contiguous()
	begin_mask = torch.tensor(begin_mask).contiguous()
	tags = torch.tensor(tags).contiguous()
	return input_ids, attention_mask, begin_mask, tags
