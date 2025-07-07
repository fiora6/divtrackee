import torch
import torch.nn.functional as F
from torch import nn

import matplotlib.pyplot as plt
from utils import *
from models.stylegan2.model import Generator
from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image
from models import irse, ir152, facenet

import numpy as np
import glob
import torchvision
from utils.divtrackee_utils import *

import criteria.clip_loss as clip_loss
import criteria.nce_loss as nce_loss
from torchvision import transforms



class FaceRecognitionModels:
    def __init__(self, device='cuda'):
        self.device = device
        self.fr_model_facenet = self.load_facenet()
        self.fr_model_152 = self.load_ir152()
        self.fr_model_50 = self.load_irse50()

    def load_facenet(self):
        fr_model_facenet = facenet.InceptionResnetV1(num_classes=8631, device=self.device)
        try:
            fr_model_facenet.load_state_dict(torch.load('./models/facenet.pth'))
        except Exception as e:
            print(f"[Error] Failed to load facenet model: {e}")
            raise
        fr_model_facenet.to(self.device)
        fr_model_facenet.eval()
        return fr_model_facenet

    def load_ir152(self):
        fr_model_152 = ir152.IR_152((112, 112))
        try:
            fr_model_152.load_state_dict(torch.load('./models/ir152.pth'))
        except Exception as e:
            print(f"[Error] Failed to load ir152 model: {e}")
            raise
        fr_model_152.to(self.device)
        fr_model_152.eval()
        return fr_model_152

    def load_irse50(self):
        fr_model_50 = irse.Backbone(50, 0.6, 'ir_se')
        try:
            fr_model_50.load_state_dict(torch.load('./models/irse50.pth'))
        except Exception as e:
            print(f"[Error] Failed to load irse50 model: {e}")
            raise
        fr_model_50.to(self.device)
        fr_model_50.eval()
        return fr_model_50
 
class DivTrackee:
    def __init__(self,args):
        self.augment = transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5)
        self.num_aug = args.num_aug
        self.nce_loss = nce_loss.NCELoss('cuda', clip_model="ViT-B/32")
        self.source_text = args.source_text
        self.description = args.makeup_prompt
        self.face_models = FaceRecognitionModels()
        self.fr_model_facenet = self.face_models.fr_model_facenet  
        self.fr_model_152 = self.face_models.fr_model_152
        self.fr_model_50 = self.face_models.fr_model_50
        
        self.steps = args.steps
        self.path = sorted(glob.glob(args.data_dir+'/*.png'))
        self.generators = sorted(glob.glob(args.checkpoint_dir+'/*.pt'))
        self.latents = torch.load(args.latent_path).unsqueeze(1)
        self.noi = torch.load(args.noise_path)
        self.trans = trans()
        self.noise_optimize = args.noise_optimize
        self.margin = args.margin
        
        self.lat_hyp = args.lambda_lat 
        self.c_hyp = args.lambda_clip
        self.adv_hyp = args.lambda_adv
        self.embedding_queue = EmbeddingQueue() 
        self.queue_hyp = args.lambda_queue
        self.output_dir = args.output_dir
    

    def process_latent(self, latent, noi):
        latent = latent.cuda()
        latent_cl = latent.clone().detach()
        latent.requires_grad = True

        noisss = []
        noiss = noi
        for nois in noiss:
            if nois.shape[2] < 512:
                nois.requires_grad = True
            else:
                nois.requires_grad = False            
            noisss.append(nois)
        return latent, latent_cl, noisss
    
    def normalize_and_interpolate(self, img, size):
        if img.dim() == 3:
            img = img.unsqueeze(0)
        return F.interpolate((img - 0.5) * 2, size=size, mode='bilinear').cuda()
    
    #calculate target img embeddings
    def get_target_embeddings(self):
        target = get_target()
        with torch.no_grad():
            target_embbeding_facenet = self.fr_model_facenet(self.normalize_and_interpolate(target, size=(160, 160)))
            target_embbeding_152 = self.fr_model_152(self.normalize_and_interpolate(target, size=(112, 112)))
            target_embbeding_50 = self.fr_model_50(self.normalize_and_interpolate(target, size=(112, 112)))
        return target_embbeding_facenet, target_embbeding_152, target_embbeding_50
    
    #calculate source img embeddings
    def get_source_embedding(self, path):
        try:
            img = Image.open(path)
        except Exception as e:
            print(f"[Error] Failed to open image {path}: {e}")
            return None, None, None, None
        bb_src1 = alignment(img)
        img_src1 = self.trans(img).unsqueeze(0)[:,:,round(bb_src1[1])-self.margin:round(bb_src1[3])+self.margin,round(bb_src1[0])-self.margin:round(bb_src1[2])+self.margin]
        norm_source_src1 = self.normalize_and_interpolate(img_src1, size=(112, 112))
        norm_source_facenet_src1 = self.normalize_and_interpolate(img_src1, size=(160, 160))

        with torch.no_grad():    
            source_embbeding_facenet_ = self.fr_model_facenet(norm_source_facenet_src1)
            source_embbeding_152_ = self.fr_model_152(norm_source_src1)
            source_embbeding_50_ = self.fr_model_50(norm_source_src1)
        return bb_src1, source_embbeding_facenet_.detach(), source_embbeding_152_.detach(), source_embbeding_50_.detach()
    
    
    def get_image_gen(self, latent, noisss, g_ema):
            
        with torch.no_grad():
            img_org, _ = g_ema([latent], input_is_latent=True, noise=noisss)

        img_org_ = img_org.detach().clone()
        img_org_ = ((img_org_+1)/2).clamp(0,1)
        img_org_ = img_org_.repeat(self.num_aug,1,1,1)

        return img_org_
    
    #calculate latent code adversarial loss
    def get_adv_loss(self, img_gen, source_embbeding_facenet_, source_embbeding_152_, source_embbeding_50_, target_embbeding_facenet, target_embbeding_152, target_embbeding_50):
        norm_source = self.normalize_and_interpolate(img_gen, size=(112, 112))
        norm_source_facenet = self.normalize_and_interpolate(img_gen, size=(160, 160))

        source_embbeding_facenet = self.fr_model_facenet(norm_source_facenet)
        source_embbeding_152 = self.fr_model_152(norm_source)
        source_embbeding_50 = self.fr_model_50(norm_source)

        adv_loss_facenet_sim = cal_adv_loss(source_embbeding_facenet, source_embbeding_facenet_)
        adv_loss_152_sim = cal_adv_loss(source_embbeding_152, source_embbeding_152_)
        adv_loss_50_sim = cal_adv_loss(source_embbeding_50, source_embbeding_50_)
        
        adv_loss_facenet = cal_guide_loss(source_embbeding_facenet, target_embbeding_facenet.detach())
        adv_loss_152 = cal_guide_loss(source_embbeding_152, target_embbeding_152.detach())
        adv_loss_50 = cal_guide_loss(source_embbeding_50, target_embbeding_50.detach())
        
        return adv_loss_facenet_sim, adv_loss_152_sim, adv_loss_50_sim, adv_loss_facenet, adv_loss_152, adv_loss_50
    
    #calculate latent code total loss
    def calculate_loss(self, l2_loss, c_loss, adv_loss_facenet_sim, adv_loss_152_sim, adv_loss_50_sim, adv_loss_facenet, adv_loss_152, adv_loss_50, queue_loss):
        dis_loss =  adv_loss_facenet_sim+adv_loss_152_sim+adv_loss_50_sim
        sim_loss =  adv_loss_facenet+adv_loss_152+adv_loss_50
        adv_loss = 0.6*sim_loss - dis_loss 
        loss =  self.lat_hyp * l2_loss+self.adv_hyp*adv_loss+self.c_hyp*c_loss + self.queue_hyp*queue_loss
        return loss
    
    
    def run(self):
        
        for ff, (latent, path) in enumerate(tqdm(zip(self.latents, self.path), total=len(self.latents), desc="Optimizing")):
            #load random target embeddings
            target_embbeding_facenet, target_embbeding_152, target_embbeding_50 = self.get_target_embeddings()
            #load fine-tuned generator
            with torch.no_grad():
                g_ema = torch.load(self.generators[ff]).eval() 
   
            _,latent_cl, noisss = self.process_latent(latent, self.noi[ff]) 
            #load initial inversed image
            img_org_ = self.get_image_gen(latent, noisss,g_ema) 
            
            optimizer = torch.optim.Adam([latent] + (noisss if self.noise_optimize else []), lr=0.01)
            
            bb_src1, source_embbeding_facenet_, source_embbeding_152_, source_embbeding_50_ = self.get_source_embedding(path)

            for i in range(self.steps):
                
                optimizer.zero_grad()
                
                img_gen_, _ = g_ema([latent], input_is_latent=True, noise=noisss)
                img_gen_ = ((img_gen_+1)/2).clamp(0,1)

                img_gen_aug = torch.cat([self.augment(img_gen_) for i in range(self.num_aug)], dim=0)                     
                c_loss = self.nce_loss(img_org_, self.source_text,img_gen_aug, self.description).sum()

                l2_loss = ((latent_cl - latent) ** 2).sum()
                #crop and normalize the image
                img_gen = img_gen_[:,:,round(bb_src1[1])-self.margin:round(bb_src1[3])+self.margin,round(bb_src1[0])-self.margin:round(bb_src1[2])+self.margin]
                norm_source = self.normalize_and_interpolate(img_gen, size=(112, 112))
                norm_source_facenet = self.normalize_and_interpolate(img_gen, size=(160, 160))

                source_embbeding_facenet = self.fr_model_facenet(norm_source_facenet)
                source_embbeding_152 = self.fr_model_152(norm_source)
                source_embbeding_50 = self.fr_model_50(norm_source)

                queue_loss = self.embedding_queue.get_queue_loss(
                    source_embbeding_facenet,
                    source_embbeding_152,
                    source_embbeding_50
                )
                adv_loss_facenet_sim, adv_loss_152_sim, adv_loss_50_sim, adv_loss_facenet, adv_loss_152, adv_loss_50 = self.get_adv_loss(img_gen, source_embbeding_facenet_, source_embbeding_152_, source_embbeding_50_, target_embbeding_facenet, target_embbeding_152, target_embbeding_50)
                                            
                
                loss = self.calculate_loss(l2_loss, c_loss, adv_loss_facenet_sim, adv_loss_152_sim, adv_loss_50_sim, adv_loss_facenet, adv_loss_152, adv_loss_50, queue_loss)               
                loss.backward()
                
                latent.grad[0][0:8] = torch.zeros(8,512) 

                optimizer.step()
                #update queue
                if (i+1) % self.steps == 0:
                    self.embedding_queue.enqueue(
                        source_embbeding_facenet,
                        source_embbeding_152,
                        source_embbeding_50
                    )
                    torchvision.utils.save_image(img_gen_, f"{self.output_dir}/{str(ff)+'_'+str(i).zfill(5)}.png", normalize=True, range=(0, 1))

      
