from preprocess.matting.models import StyleMatte
import numpy as np 
import torch
from torchvision import transforms
import cv2

class MattingModel(torch.nn.Module):
    def __init__(self, checkpoint = f"stylematte.pth", device='cuda'):
        super().__init__()
        model = StyleMatte()
        model = model.to(device)
        state_dict = torch.load(checkpoint, map_location=f'{device}')
        model.load_state_dict(state_dict)
        model.eval()
        self.checkpoint = checkpoint
        self.device = device
        self.model = model
    
    def forward(self, img):
        h,w,_ = img.shape
        if h%8!=0 or w%8!=0:
            img=cv2.copyMakeBorder(img, 8-h%8, 0, 8-w%8, 0, cv2.BORDER_REFLECT)
    
        tensor_img = torch.from_numpy(img).permute(2, 0, 1).to(self.device)
        input_t = tensor_img
        input_t = input_t/255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        input_t = normalize(input_t)
        input_t = input_t.unsqueeze(0).float()
        with torch.no_grad():
            out = self.model(input_t)
        result = out[0][:,-h:,-w:].cpu().numpy()
        return result[0]