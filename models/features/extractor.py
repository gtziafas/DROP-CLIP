import os
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import einops
from tqdm import tqdm 
import numpy as np
from PIL import Image
import gc
import cv2

import utils.image as imutils
import utils.transforms as tutils

# import tensorflow as tf2
# import tensorflow.compat.v1 as tf
# from models.features.openseg import extract_openseg_img_feature
from models.features.clip import clip
from models.features.dino.dinov2_vit_extractor import DINOv2Featurizer


DEVICE = torch.device('cuda') if torch.cuda.is_available else 'cpu'


def noop():
    def _noop(x):
        return x
    return _noop


# def image_preprocess(
#     img_crop,
#     img_resize,
#     center_crop=False,
#     norm_mean=[0.5, 0.5, 0.5],
#     norm_std=[0.1, 0.1, 0.1],
#     ):
#     return transforms.Compose([
#         transforms.CenterCrop(img_crop) if img_crop is not None else noop,
#         transforms.Resize(img_resize, interpolation=transforms.InterpolationMode.BICUBIC),
#         transforms.CenterCrop(img_resize) if center_crop else noop,
#         transforms.ToTensor(),
#         transforms.Normalize(mean=norm_mean, std=norm_std)
#     ])

def image_preprocess(
    img_crop,
    img_resize,
    center_crop=False,
    norm_mean=[0.5, 0.5, 0.5],
    norm_std=[0.1, 0.1, 0.1],
    ):
    
    compose_list = []
    if img_crop is not None:
        compose_list.append(transforms.CenterCrop(img_crop))
    compose_list.append(transforms.Resize(img_resize, interpolation=transforms.InterpolationMode.BICUBIC))
    if center_crop:
        compose_list.append(transforms.CenterCrop(img_resize))
    compose_list.append(transforms.ToTensor())
    compose_list.append(transforms.Normalize(mean=norm_mean, std=norm_std))
    return transforms.Compose(compose_list)
    

def clip_preprocess(img_crop, img_resize, center_crop=True):
    return image_preprocess(
        img_crop, img_resize, center_crop, norm_mean=[0.48145466, 0.4578275, 0.40821073], norm_std=[0.26862954, 0.26130258, 0.27577711]
    )


def dinov2_preprocess(img_crop, img_resize, center_crop=True):
    return image_preprocess(
        img_crop, img_resize, center_crop, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]
    )



class FeatureExtractor(object):
    """
    Base class for extracting features from 2D images with a pretrained model
    """
    def __init__(
        self,
        vision_model,
        preprocess,
        forward_key,
        feat_dim,
        device = DEVICE,
        batch_size = 8,
        save_folder = None
    ):
        self.preprocess = preprocess
        self.model = vision_model
        self.batch_size = batch_size
        self.device = device
        self.feat_dim = feat_dim
        self.save_folder = save_folder
        self.call = getattr(self.model, forward_key)

    @torch.no_grad()
    def extract(self, images, device=None, batch_size=None):
        if isinstance(images[0], str):
            # image path - load
            images = [Image.open(path).convert('RGB') for path in images]

        device = device or self.device
        batch_size = batch_size or self.batch_size
        
        preprocessed_images = torch.stack([self.preprocess(image) for image in images])
        preprocessed_images = preprocessed_images.to(device)  # (b, 3, h, w)
        embeddings = []
        for i in range(0, len(preprocessed_images), batch_size):
        # for i in tqdm(
        #     range(0, len(preprocessed_images), batch_size),
        #     desc="Extracting CLIP features",
        # ):
            batch = preprocessed_images[i : i + batch_size]
            embeddings.append(self.call(batch))
        embeddings = list(torch.cat(embeddings, dim=0))

        # Delete and clear memory to be safe
        del preprocessed_images
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        return embeddings

    def save(self, images, save_name, device=None, batch_size=None):
        assert self.save_folder is not None, 'Set save_folder attribute first'
        save_path = os.path.join(self.save_folder, save_name + '.npy')
        embeddings = self.extract(images, device, batch_size).cpu().numpy()
        np.save(embeddings, save_path)


    @torch.no_grad()
    def extract_obj_prior(self, images, segms, obj_ids, device=None, batch_size=None):
        device = device or self.device
        batch_size = batch_size or self.batch_size

        preprocessed_images = []
        collect_map = [] 
        ind = 0
        for img, seg in list(zip(images, segms)):
            # keep only masked region with gray-ish background
            objs = obj_ids[ind]
            background = np.ones_like(img) * np.array([200,200, 200], dtype=np.uint8)
            for obj in objs:
                obj_mask = seg == obj
                img_mask = background.copy()
                img_mask[obj_mask==True] = img[obj_mask==True]

                img_input = self.preprocess(Image.fromarray(img_mask))
                preprocessed_images.append(img_input)

                collect_map.append(ind)
                
            ind += 1

        preprocessed_images = torch.stack(preprocessed_images).to(device)
        collect_map = torch.as_tensor(collect_map, device=device)
        unique_groups = torch.unique(collect_map)
        
        embeddings = []
        for i in range(0, len(preprocessed_images), batch_size):
        # for i in tqdm(
        #     range(0, len(preprocessed_images), batch_size),
        #     desc="Extracting CLIP features",
        # ):
            batch = preprocessed_images[i : i + batch_size]
            embeddings.append(self.call(batch))
        embeddings = torch.cat(embeddings, dim=0)
        
        # Split to a list of all images, each K objects
        embeddings_grouped = [embeddings[collect_map == ind] for ind in unique_groups]
        # Delete and clear memory to be safe
        del preprocessed_images, embeddings
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        return embeddings_grouped


class Dinov2Extractor(FeatureExtractor):
    def __init__(self, 
        model_name="dinov2_vits14_reg", 
        patch_size=14,
        img_crop=None,
        img_resize=(336,448)    ,
        center_crop=None,
        feat_dim=384,
        mode="patch",
        device = DEVICE,
        batch_size = 12,
        save_folder = None,
    ):
        self.patch_size = patch_size
        self.orig_size = img_crop
        self.img_size = img_resize
        self.patch_h = img_resize[0] // patch_size
        self.patch_w = img_resize[1] // patch_size 
        self.feat_dim = feat_dim

        model = DINOv2Featurizer(model_name, device=device)
        preprocess = dinov2_preprocess(
            img_crop=img_crop, img_resize=img_resize, center_crop=center_crop)
        print(f"Loaded Dinov2 model {model_name}")

        if mode == "cls":
            forward_key = "get_cls_token"
        elif mode == "patch":
            forward_key = "get_patch_tokens"
        else:
            raise ValueError("Set mode to either ['cls', 'patch']")
        self.mode = mode

        super().__init__(vision_model=model, preprocess=preprocess, forward_key=forward_key, 
            feat_dim=feat_dim, device=device, batch_size=batch_size, save_folder=save_folder)

    def set_mode(self, mode):
        if mode == "cls":
            forward_key = "get_cls_token"
        elif mode == "patch":
            forward_key = "get_patch_tokens"
        else:
            raise ValueError("Set mode to either ['cls', 'patch']")

        self.mode = mode
        self.forward_key = forward_key
        self.call = getattr(self.model, forward_key)

        return mode, forward_key

class OpenSegExtractor:
    def __init__(self, saved_model_path, feat_dim=768, img_resize=336, device=DEVICE):
        self.feat_dim = feat_dim
        self.img_size = [img_resize, img_resize]
        self.device = device
        self.model = tf2.saved_model.load(saved_model_path,
                    tags=[tf.saved_model.tag_constants.SERVING],)
        self.text_emb = tf.zeros([1, 1, self.feat_dim])

    def extract(self, image_dirs, device=None):
        device = device or self.device
        features = []
        for img_dir in tqdm(image_dirs):
            feat = extract_openseg_img_feature(
                img_dir, self.model, self.text_emb, img_size=self.img_size).to(device)
            features.append(feat)
        return features


class ClipExtractor(FeatureExtractor):  
    """
    Extract CLIP visual features from 2D images. Currently supports two modes:
        - "cls": Classic cls token embedding from last layer
        - "patch": Patch embeddings from last layer using MaskCLIP reparameterization trick
    """

    NEGATIVE_PROMPT_GENERIC = ["object", "thing", "texture", "stuff"]
    SOFTMAX_TEMP = 0.1
    
    def __init__(self, 
                model_name = "ViT-L/14@336px",
                patch_size=14,
                img_crop=840,
                img_resize=336,
                center_crop=336,
                feat_dim=768,
                mode="cls",
                visual_prompt = ["crop", "mask-blur", "mask-gray"],
                crop_num_levels = 3,
                crop_expansion_ratio = 0.1,
                blur_kernel = 31,
                device = DEVICE,
                batch_size = 8,
                save_folder = None,
    ):
        self.visual_prompt = visual_prompt
        self.crop_num_levels = crop_num_levels
        self.crop_expansion_ratio = crop_expansion_ratio
        self.blur_kernel = blur_kernel

        self.patch_size = patch_size
        self.orig_size = img_crop
        self.img_size = img_resize
        self.patch_h = img_resize[0] // patch_size
        self.patch_w = img_resize[1] // patch_size 
        model, _ = clip.load(model_name, device=device)
        model = model.eval().to(device)
        preprocess = clip_preprocess(
            img_crop=img_crop, img_resize=img_resize, center_crop=center_crop)
        print(f"Loaded CLIP model {model_name}")

        if mode == "cls":
            forward_key = "encode_image"
        elif mode == "patch":
            forward_key = "get_patch_encodings"
        else:
            raise ValueError("Set mode to either ['cls', 'patch']")
        self.mode = mode

        super().__init__(vision_model=model, preprocess=preprocess, forward_key=forward_key, 
            feat_dim=feat_dim, device=device, batch_size=batch_size, save_folder=save_folder)

    def make_prompt(self, image, binary_mask):
        
        def obtain_background_color(image, binary_mask):
            image_region = image[binary_mask==True]
            expected_color = image_region.mean(0)
            # Determine if the expected color is closer to white or black
            white = np.array([255, 255, 255], dtype=np.uint8)
            black = np.array([0, 0, 0], dtype=np.uint8)
            if np.linalg.norm(expected_color - white) < np.linalg.norm(expected_color - black):
                return black  # Closer to white, return black
            else:
                return white  # Closer to black, return white

        image_prompt = []
        use_color = obtain_background_color(image, binary_mask)
        target_ratio = float(image.shape[1] / image.shape[0])

        if "crop" in self.visual_prompt:
            # get crops of multiple resolutions
            for level in range(self.crop_num_levels):
                x1, y1, x2, y2 = imutils.mask2box_multi_level(
                    binary_mask, level, self.crop_expansion_ratio)
                crop = image[y1:y2, x1:x2]
                crop = imutils.add_borders_to_image(
                    crop, target_ratio, use_color)
                image_prompt.append(crop)

        if "crop-mask" in self.visual_prompt:
            # get crops of multiple resolutions
            background = np.ones_like(image) * use_color
            img_mask = background.copy()
            img_mask[binary_mask==True] = image[binary_mask==True]
            for level in range(self.crop_num_levels):
                x1, y1, x2, y2 = imutils.mask2box_multi_level(
                    binary_mask, level, self.crop_expansion_ratio)
                crop = img_mask[y1:y2, x1:x2]
                crop = imutils.add_borders_to_image(
                    crop, target_ratio, use_color)
                image_prompt.append(crop)

        if "mask-blur" in self.visual_prompt:
            background = cv2.GaussianBlur(
                image.copy(), (self.blur_kernel,self.blur_kernel), 0)
            img_mask = background.copy()
            img_mask[binary_mask==True] = image[binary_mask==True]
            image_prompt.append(img_mask)

        if "mask-gray" in self.visual_prompt:
            gray = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
            background = cv2.merge([gray, gray, gray])
            img_mask = background.copy()
            img_mask[binary_mask==True] = image[binary_mask==True]
            image_prompt.append(img_mask)

        if "mask-out" in self.visual_prompt:
            # use fixed color background, depending on mask color
            background = np.ones_like(image) * use_color
            img_mask = background.copy()
            img_mask[binary_mask==True] = image[binary_mask==True]
            image_prompt.append(img_mask)

        return image_prompt


    @torch.no_grad()
    def extract_obj_prior(self, images, segms, obj_ids, device=None, batch_size=None):
        # obj_ids: List of obj ids per view
        device = device or self.device
        batch_size = batch_size or self.batch_size

        preprocessed_images = []
        #collect_map = [] 
        embeddings = []
        ind = 0

        for img, seg in list(zip(images, segms)):
            objs = obj_ids[ind]
            
            existing_ids = np.unique(seg)[1:]
            n_objects = 0
            img_input_view = []
            L = None
            for obj in objs:
                if obj not in existing_ids:
                    continue
                obj_mask = seg == obj
                image_prompt = self.make_prompt(img, obj_mask)
                if L is None:
                    L = len(image_prompt)
                else:
                    assert L == len(image_prompt)

                img_input = [self.preprocess(Image.fromarray(im)) for im in image_prompt]
                #preprocessed_images.extend(img_input)
                img_input_view.extend(img_input)

                del img_input
                n_objects += 1
                #collect_map.append(ind)
            
            img_input_view = torch.stack(img_input_view).to(device)

            embeddings_view = []
            for i in range(0, img_input_view.shape[0], batch_size):
                batch = img_input_view[i : i + batch_size]
                embeddings_view.append(self.call(batch)) # (K*L, C)
            
            embeddings_view = torch.cat(embeddings_view, dim=0)
            embeddings_view = einops.rearrange(
                embeddings_view, "(k l) c -> k l c", 
                k=n_objects, 
                l=L,
            ).mean(1)

            embeddings.append(embeddings_view)

            del img_input_view, embeddings_view
            if device == 'cuda':
                torch.cuda.empty_cache()
            #gc.collect()

            #print('finished an image')
            ind += 1

        return embeddings
        # collect_map = torch.as_tensor(collect_map, device=device)
        # unique_groups = torch.unique(collect_map)
        # embeddings_grouped = [embeddings[collect_map == ind] for ind in unique_groups]

        return embeddings_grouped
        # preprocessed_images = torch.stack(preprocessed_images)
        # collect_map = torch.as_tensor(collect_map, device=device)
        # unique_groups = torch.unique(collect_map)
        
        # embeddings = []
        # for i in range(0, len(preprocessed_images), batch_size):
        # # for i in tqdm(
        # #     range(0, len(preprocessed_images), batch_size),
        # #     desc="Extracting CLIP features",
        # # ):
        #     batch = preprocessed_images[i : i + batch_size]
        #     embeddings.append(self.call(batch))
        # embeddings = torch.cat(embeddings, dim=0)

        # embeddings_grouped = einops.rearrange(
        #     embeddings, "(v k l) c -> v k l c", 
        #     v=len(images),
        #     k=len(obj_ids), 
        #     l=-1
        # )
        # embeddings_grouped = embeddings_grouped.mean(2) # average over prompts
        
        # # # Split to a list of all images, each K objects
        # # embeddings_grouped = [embeddings[collect_map == ind] for ind in unique_groups]

        # # Delete and clear memory to be safe
        # del preprocessed_images, embeddings
        # if device == 'cuda':
        #     torch.cuda.empty_cache()
        # gc.collect()
        #return embeddings

    def set_mode(self, mode):
        if mode == "cls":
            forward_key = "encode_image"
        elif mode == "patch":
            forward_key = "get_patch_encodings"
        else:
            raise ValueError("Set mode to either ['cls', 'patch']")

        self.mode = mode
        self.forward_key = forward_key
        self.call = getattr(self.model, forward_key)

        return mode, forward_key

    @torch.no_grad()
    def compute_similarity(self, vis_feat_norm, qpos, qneg = False, softmax_temp=None):
        softmax_temp = softmax_temp or self.SOFTMAX_TEMP
        
        # Encode positive query
        qpos = clip.tokenize(qpos).to(self.device)
        qpos = self.model.encode_text(qpos)
        qpos /= qpos.norm(dim=-1, keepdim=True)

        # Encode generic negative query if included
        if qneg:
            qneg = self.NEGATIVE_PROMPT_GENERIC
            qneg = clip.tokenize(qneg).to(self.device)
            qneg = self.model.encode_text(qneg)
            qneg /= qneg.norm(dim=-1, keepdim=True)
        
            # Use paired softmax method with positive and negative texts
            text_embs = torch.cat([qpos, qneg], dim=0)
            raw_sims = vis_feat_norm @ text_embs.T

            # Broadcast positive label similarities to all negative labels
            pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
            pos_sims = pos_sims.broadcast_to(neg_sims.shape)
            paired_sims = torch.cat([pos_sims, neg_sims], dim=-1)

            # Compute paired softmax
            probs = (paired_sims / softmax_temp).softmax(dim=-1)[..., :1]
            torch.nan_to_num_(probs, nan=0.0)
            sims, _ = probs.min(dim=-1, keepdim=True)
            
            return sims

        else:
            return vis_feat_norm @ qpos.T
    

    @torch.no_grad()
    def get_similarities(self, vis_feats, text_query, qneg=False, norm_vis_feat=True):
        sims=[]
        for clip_feat in vis_feats:
            if norm_vis_feat:
                clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
            if self.mode == "patch":
                # clip_feat: (P, P, C)
                clip_feat = tutils.reconstruct_feature_map(clip_feat, (self.orig_size, self.orig_size, 3))
            sim = self.compute_similarity(clip_feat, qpos=text_query, qneg=qneg).squeeze()
            sims.append(sim)

        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        return sims


    @torch.no_grad()
    def compute_similarity_1(self, vis_feat_norm, qpos, qneg = None, softmax_temp=None, method="paired"):
        softmax_temp = softmax_temp or self.SOFTMAX_TEMP

        # Encode positive query
        qpos = clip.tokenize(qpos).to(vis_feat_norm.device)
        qpos = self.model.encode_text(qpos)
        qpos /= qpos.norm(dim=-1, keepdim=True)

        # Encode generic negative query if included
        if qneg is not None:
            assert isinstance(qneg, list), "qneg argument should be list or None"
            if not len(qneg):
                qneg = self.NEGATIVE_PROMPT_GENERIC
            
            qneg = clip.tokenize(qneg).to(vis_feat_norm.device)
            qneg = self.model.encode_text(qneg)
            qneg /= qneg.norm(dim=-1, keepdim=True)

            # Use paired softmax method with positive and negative texts
            text_embs = torch.cat([qpos, qneg], dim=0)
            raw_sims = vis_feat_norm @ text_embs.T

            if method == "paired":
                # Broadcast positive label similarities to all negative labels
                pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
                pos_sims = pos_sims.broadcast_to(neg_sims.shape)
                paired_sims = torch.cat([pos_sims, neg_sims], dim=-1)

                # Compute paired softmax
                probs = (paired_sims / softmax_temp).softmax(dim=-1)[..., :1]
                torch.nan_to_num_(probs, nan=0.0)
                sims, _ = probs.min(dim=-1, keepdim=True)
                return sims

            elif method == "argmax":
                return raw_sims

        else:
            return vis_feat_norm @ qpos.T
    

    @torch.no_grad()
    def predict(self, vis_feats, qpos, method, threshold=None, norm_vis_feat=True, qneg=None):
        method = method
        threshold = threshold
        norm_vis_feat = norm_vis_feat

        if norm_vis_feat:
            vis_feats /= vis_feats.norm(dim=-1, keepdim=True)

        sims = self.compute_similarity_1(
            vis_feats, qpos, qneg, method=method).squeeze()

        if qneg is None or (qneg is not None and method == "paired"):     
            if sims.max() != sims.min():
                    sims_norm = (sims - sims.min()) / (sims.max() - sims.min())
            else:
                    sims_norm = sims / sims.max()

            pred = sims_norm > threshold
            return pred, sims_norm.float()

        elif qneg is not None and method == "argmax":
            sims_dif = sims[:,0] - sims[:, 1:].mean(-1)

            if sims.max() != sims.min():
                sims_norm = (sims_dif - sims_dif.min()) / (sims_dif.max() - sims_dif.min())
            else:
                sims_norm = sims_dif / sims_dif.max()

            pred = torch.max(sims, 1)[1] == 0 # 0->positive
            return pred, sims_norm.float()


