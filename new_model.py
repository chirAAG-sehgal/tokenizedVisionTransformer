from model import tokenizer, tokenizerConfig
from PIL import Image
import torch 
import numpy as np
from transformer import VisionTransformer
from torchsummary import summary

from model import tokenizer, tokenizerConfig
from PIL import Image
import torch 
import numpy as np
from transformer import VisionTransformer
from torchsummary import summary

class NewModel(torch.nn.Module):  # Inherit from torch.nn.Module
    def __init__(self):
        super(NewModel, self).__init__()  # Call the constructor of torch.nn.Module
        config = tokenizerConfig()
        self.model_tokenizer = tokenizer(config).to('cuda')
        self.model_tokenizer.load_state_dict(torch.load('quantizer_only.pt')['model'], strict=False)
        self.model_tokenizer.eval()
        self.vit = VisionTransformer().to('cuda')
    
    def forward(self, x):
        x = self.model_tokenizer(x)
        x = self.vit(x)
        return x

if __name__ == "__main__":
    model = NewModel()
    model.load_state_dict(torch.load('weights/model_epoch_5.pth'))
    output = model(torch.zeros((1,3,256,256),dtype=torch.float32,device="cuda"))
    print(output)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
