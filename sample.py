import torch 

state_dict = torch.load('vq_ds8_c2i.pt')
quantizer_only = {}
quantizer_only['model'] = {k:v for k,v in state_dict['model'].items() if 'encoder'in k or 'quantize' in k or ("post" not in k and "quant_conv" in k) }
torch.save(quantizer_only, 'quantizer_ only.pt')
print(quantizer_only['model'].keys())