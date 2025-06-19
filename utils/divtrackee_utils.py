
import cv2
import torch
from torchvision import transforms
from models import irse, ir152, facenet
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
from models.stylegan2.model import Generator
import os
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import cv2
import torch
import clip
from PIL import Image
from models import irse, ir152, facenet
import torch.nn.functional as F
import numpy as np
import glob

def preprocess(im, mean, std, device):
    if len(im.size()) == 3:
        im = im.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    elif len(im.size()) == 4:
        im = im.transpose(1, 3).transpose(2, 3)

    mean = torch.tensor(mean).to(device)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std).to(device)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    im = (im - mean) / std
    return im

def read_img(data_dir, mean, std, device):
    img = cv2.imread(data_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    img = torch.from_numpy(img).to(torch.float32).to(device)
    img = preprocess(img, mean, std, device)
    return img

from facenet_pytorch import MTCNN
mtcnn = MTCNN(image_size=1024, margin=0, post_process=False, select_largest=False, device='cuda')
def alignment(image):
    boxes, probs = mtcnn.detect(image)
    return boxes[0]

def trans():
    return transforms.Compose([transforms.ToTensor()])


class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.model.eval();
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda").view(1,3,1,1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda").view(1,3,1,1)

    def forward(self, image, text):
        #image = image.add(1).div(2)
        image = image.sub(self.mean).div(self.std)
        image = self.face_pool(image)
        similarity = 1 - self.model(image, text)[0]/ 100
        return similarity

def cos_simi(emb_1, emb_2):
    return torch.mean(torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1))

def cal_adv_loss(source_resize, target_resize):
    cos_loss = (1-cos_simi(source_resize, target_resize))
    return cos_loss

def cal_guide_loss(source_resize, target_resize, delta=0.2):
    guide_loss = torch.maximum(torch.tensor(0.0, device=source_resize.device), (1-cos_simi(source_resize, target_resize)) - delta)
    return guide_loss
    
def get_target():
    lfw_dir = "../dataset/lfw"  

    image_paths = []
    for root, dirs, files in os.walk(lfw_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    
    target_path = np.random.choice(image_paths)
    image = Image.open(target_path)
    boxes = alignment(image)
    if boxes is None:
        return get_target()

    x1, y1, x2, y2 = [int(b) for b in boxes]
    face_img = image.crop((x1, y1, x2, y2))

    face_img = face_img.resize((1024, 1024), Image.LANCZOS)
    
    transform = trans()
    face_tensor = transform(face_img)
    
    return face_tensor

class EmbeddingQueue:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.queue_facenet = []  
        self.queue_152 = []  
        self.queue_50 = []  
    
    def enqueue(self, emb_facenet, emb_152, emb_50):
        self.queue_facenet.append(emb_facenet.detach())
        self.queue_152.append(emb_152.detach())
        self.queue_50.append(emb_50.detach())
        
        if len(self.queue_facenet) > self.max_size:
            self.queue_facenet.pop(0)
            self.queue_152.pop(0)
            self.queue_50.pop(0)
    
    def get_queue_loss(self, current_facenet, current_152, current_50):
        if len(self.queue_facenet) == 0:
            return torch.tensor(0.0, device=current_facenet.device)
        
        total_loss = torch.tensor(0.0, device=current_facenet.device)
        
        for q_facenet, q_152, q_50 in zip(
            self.queue_facenet, self.queue_152, self.queue_50):
            
            loss_facenet = cal_adv_loss(current_facenet, q_facenet)
            loss_152 = cal_adv_loss(current_152, q_152)
            loss_50 = cal_adv_loss(current_50, q_50)
            
            total_loss += (loss_facenet + loss_152 + loss_50)

        return total_loss / len(self.queue_facenet) 