import math
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import einops
from tqdm import tqdm 
import numpy as np
from PIL import Image
import clip
import os

import utils.image as imutils

DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')


class Dinov2Features:
    
    def __init__(self, 
                crop_size,
                save_folder,
                model_tag = "dinov2_vitl14_reg",
                device = DEVICE,
                
    ):
        self.model = torch.hub.load(
            'facebookresearch/dinov2', model_tag)
        self.model = self.model.eval().to(device)

        self.img_size = crop_size
        self.patch_size = self.model.patch_size # patchsize=14
        self.patch_w = self.patch_h = self.img_size // self.patch_size
        assert crop_size % self.patch_size == 0, \
            f"crop size ({crop_size}) should be multiple of model patch size (={self.patch_size})"
        
        self.preprocess = transforms.Compose([
                transforms.CenterCrop(crop_size), #should be multiple of model patch_size                 
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.2)
        ])

        self.device = device
        self.feat_dim = 1024 # vitl14
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def reconstruct_shape(self, feat):
        # feat: (B, H*W, C)
        feat = einops.rearrange(feat, '(h w) c -> h w c', h=self.patch_h, w=self.patch_w)
        feat = feat.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)
        return feat

    @torch.no_grad()
    def extract(self, image_paths, reconstruct_shape=False):
        for img_path in tqdm(image_paths):
            fname = img_path.split('/')[-1].split('.')[0] + '.npy'
            save_path = os.path.join(
                self.save_folder, fname)
            
            image = Image.open(img_path).convert('RGB')

            image_input = self.preprocess(image)
            image_input = image_input.to(self.device).unsqueeze(0)

            feat = self.model.forward_features(image_input)
            feat = feat['x_norm_patchtokens'][0].cpu().numpy()
            
            if reconstruct_shape:
                feat = self.reconstruct_shape(feat)

            if self.device == torch.device("cuda"):
                torch.cuda.empty_cache()

            np.save(save_path, feat)

    @torch.no_grad()
    def extract_obj_prior(self, images, segms):
        batch_feats = []
        for img, seg in tqdm(list(zip(images, segms))):
            H, W = img.shape[:2]

            obj_ids = sorted(np.unique(seg))
            obj_masks = imutils.seg_mask_to_binary(seg)

            img_feats = torch.zeros((H, W, self.feat_dim), 
                device=self.device).float()

            for obj_mask, img_mask in zip(obj_masks, img_masks):
                img_mask = np.zeros_like(img)
                img_mask[obj_mask==True] = img[obj_mask==True]

                # crop around mask center
                x, y, w, h = imutils.get_mask_bbox(obj_mask)
                res = math.ceil(max(w, h) / self.patch_size) * self.patch_size

                img_input = transforms.Compose([
                        transforms.CenterCrop(res), #should be multiple of model patch_size                 
                        transforms.ToTensor(),
                        transforms.Normalize(mean=0.5, std=0.2)
                ])(img_mask)
                    
                feat = self.model.forward_features(img_input.unsqueeze(0))[0]
                feat = feat['x_norm_clstoken'] # (C,)

                _mask = torch.from_numpy(obj_mask).to(self.device) 
                img_feats[_mask == True] = feat

            batch_feats.append(img_feats.cpu())

            if self.device == torch.device("cuda"):
                torch.cuda.empty_cache()

        return batch_feats

    @torch.no_grad()
    def extract_obj_prior_multiview(self, images, segms, objIDs):
        # images: List[List[Image]], (B, V, H, W, 3)
        # segms: List[List[np.float32]], (B, V, H, W)
        # objIDs: List[List[int]], (B, K)
        
        batch_obj_feats = []

        for image_views, seg_views, obj_ids in tqdm(list(zip(images, segms, objIDs))):
            
            obj_feats = {}

            # average embedding for each object accross views
            for obj_id in obj_ids:

                aggr_feat = []

                for img, seg in zip(image_views, seg_views):
                    
                    obj_mask = seg == obj_id
                    
                    img_mask = np.zeros_like(img)
                    img_mask[obj_mask==True] = img[obj_mask==True]

                    x, y, w, h = imutils.get_mask_bbox(obj_mask)
                    res = math.ceil(max(w, h) / self.patch_size) * self.patch_size

                    img_input = transforms.Compose([
                            transforms.CenterCrop(res), #should be multiple of model patch_size                 
                            transforms.ToTensor(),
                            transforms.Normalize(mean=0.5, std=0.2)
                    ])(Image.fromarray(img_mask)).to(self.device)

                    feat = self.model.forward_features(img_input.unsqueeze(0))
                    feat = feat['x_norm_clstoken'][0] # (C,)

                    aggr_feat.append(feat.cpu())
                    if self.device == torch.device("cuda"):
                        torch.cuda.empty_cache()

                obj_feats[obj_id] = torch.stack(aggr_feat).mean(0)

            batch_obj_feats.append(obj_feats)

        return batch_obj_feats



class CLIPFeatures:
    
    def __init__(self, 
                save_folder,
                crop_size=840,
                patch_size=168,
                patch_stride=168,
                backbone = "ViT-L/14@336px",
                device = DEVICE,
                batch_size = 8
    ):
        self.model, self.preprocess = clip.load(
            backbone, device=device, jit=False)
        self.model = self.model.eval().to(device)

        self.img_size = crop_size
        self.patch_size = patch_size 
        self.patch_stride = patch_stride 
        self.patch_h = self.patch_w = self.img_size // self.patch_size

        self.device = device
        self.feat_dim = 1024 # RN50
        self.batch_size = batch_size
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def patchify_image(self, image):
        H, W, _ = image.shape
        R = self.img_size
        patch_size = self.patch_size

        startx = H // 2 - (R // 2)
        starty = W // 2 - (R // 2)
        cropped_image = image[startx:startx+R, starty:starty+R]  # Center crop
        
        patches = cropped_image.reshape(R // patch_size, patch_size, R // patch_size, patch_size, 3)
        patches = patches.swapaxes(1, 2).reshape(-1, patch_size, patch_size, 3)
        
        return patches, cropped_image

    def reconstruct_shape(self, feat):
        # feat: (H*W, C)
        feat = einops.rearrange(feat, '(h w) c -> h w c',
            h=self.patch_h, w=self.patch_w)
        feat = feat.unsqueeze(2).expand(-1, -1, self.patch_size, -1)
        feat = feat.unsqueeze(1).expand(-1, self.patch_size, -1, -1, -1)
        feat = einops.rearrange(feat, 'h p1 w p2 c -> (h p1) (w p2) c')
        return feat

    @torch.no_grad()
    def forward_features(self, images):
        for img in images:
            img = np.array(img)
            patches, _ = self.patchify_image(img)
               
    @torch.no_grad()
    def extract(self, image_paths, batch_size=None, reconstruct_shape=False):
        batch_size = batch_size or self.batch_size

        image_input_list = []
        save_paths = []
        for img_path in image_paths:
            fname = img_path.split('/')[-1].split('.')[0] + '.npy'

            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            
            patches, _ = self.patchify_image(img) # (P, h, w, 3)
            patches = [self.preprocess(
                Image.fromarray(im)) for im in patches] # (P, 3, 224, 224)
            image_input_list.append(patches)
            save_paths.append(os.path.join(
                self.save_folder, fname))
        

        for save_path, patch_input in tqdm(list(zip(save_paths, image_input_list))):
            n_patches = len(patch_input)
            n_batches = math.ceil(n_patches / batch_size)

            patch_feats = None
            for bi in range(n_batches):
                start_idx = bi * batch_size 
                end_idx = min((bi+1) * batch_size, n_patches)
                
                image_input = torch.stack(
                    patch_input[start_idx : end_idx]
                ).to(self.device)

                feat = self.model.encode_image(image_input) # (B, C)
                feat = F.normalize(feat, dim=-1)
                
                if patch_feats is None:
                    patch_feats = feat
                else:
                    patch_feats = torch.cat(
                        [patch_feats, feat], dim=0)
                del feat

                if self.device == torch.device("cuda"):
                    torch.cuda.empty_cache()

            patch_feats = patch_feats.cpu()
            if reconstruct_shape:
                patch_feats = self.reconstruct_shape(patch_feats).cpu() # (H, W, C)
            
            np.save(save_path, patch_feats.numpy())

    @torch.no_grad()
    def extract_obj_prior(self, images, segms):
        batch_feats = []
        for img, seg in tqdm(list(zip(images, segms))):
            H, W = img.shape[:2]

            obj_ids = sorted(np.unique(seg))
            obj_masks = imutils.seg_mask_to_binary(seg)

            img_feats = torch.zeros((H, W, self.feat_dim), 
                device=self.device).float()

            for obj_mask, img_mask in zip(obj_masks, img_masks):
                img_mask = np.zeros_like(img)
                img_mask[obj_mask==True] = img[obj_mask==True]

                # crop around mask center
                x, y, w, h = imutils.get_mask_bbox(obj_mask)
                res = math.ceil(max(w, h) / self.patch_size) * self.patch_size

                img_input = transforms.Compose([
                        transforms.CenterCrop(res), #should be multiple of model patch_size                 
                        transforms.ToTensor(),
                        transforms.Normalize(mean=0.5, std=0.2)
                ])(img_mask)
                    
                feat = self.model.forward_features(img_input.unsqueeze(0))[0]
                feat = feat['x_norm_clstoken'] # (C,)

                _mask = torch.from_numpy(obj_mask).to(self.device) 
                img_feats[_mask == True] = feat

            batch_feats.append(img_feats.cpu())

            if self.device == torch.device("cuda"):
                torch.cuda.empty_cache()

        return batch_feats

    @torch.no_grad()
    def extract_obj_prior_multiview(self, images, segms, objIDs):
        # images: List[List[Image]], (B, V, H, W, 3)
        # segms: List[List[np.float32]], (B, V, H, W)
        # objIDs: List[List[int]], (B, K)
        
        batch_obj_feats = []

        for image_views, seg_views, obj_ids in tqdm(list(zip(images, segms, objIDs))):
            
            obj_feats = {}

            # average embedding for each object accross views
            for obj_id in obj_ids:

                aggr_feat = []

                for img, seg in zip(image_views, seg_views):
                    
                    obj_mask = seg == obj_id
                    
                    img_mask = np.ones_like(img) * 255
                    img_mask[obj_mask==True] = img[obj_mask==True]

                    # x, y, w, h = imutils.get_mask_bbox(obj_mask)
                    # res = math.ceil(max(w, h) / self.patch_size) * self.patch_size

                    # img_input = transforms.Compose([
                    #         transforms.CenterCrop(res), #should be multiple of model patch_size                 
                    # ])(Image.fromarray(img_mask))
                    img_input = Image.fromarray(img_mask)

                    img_input = self.preprocess(img_input).to(self.device)

                    feat = self.model.encode_image(img_input.unsqueeze(0))
                    feat = F.normalize(feat, dim=-1)[0]

                    aggr_feat.append(feat.cpu())
                    if self.device == torch.device("cuda"):
                        torch.cuda.empty_cache()

                obj_feats[obj_id] = torch.stack(aggr_feat).mean(0)

            batch_obj_feats.append(obj_feats)

        return batch_obj_feats
