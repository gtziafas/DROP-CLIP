import torch

from models.features.clip import clip

DEVICE = torch.device('cuda') if torch.cuda.is_available else 'cpu'


class ClipSimilarity(object):  

  NEGATIVE_PROMPT_GENERIC = ["object", "thing", "texture", "stuff"]
  SOFTMAX_TEMP = 0.1
  
  def __init__(self, 
        model_name = "ViT-L/14@336px",
        method="paired",
        threshold=0.7,
        norm_vis_feat=True,
        device=DEVICE,
  ):
    self.device = device
    self.threshold = threshold
    self.method = method
    self.norm_vis_feat = norm_vis_feat
    self.model, _ = clip.load(model_name, device=device)
    self.model = self.model.eval().to(device)
    print(f"Loaded CLIP model {model_name}")

  @torch.no_grad()
  def compute_similarity(self, vis_feat_norm, qpos, qneg = None, softmax_temp=None, method="paired"):
    softmax_temp = softmax_temp or self.SOFTMAX_TEMP

    # Encode positive query
    qpos = clip.tokenize(qpos).to(self.device)
    qpos = self.model.encode_text(qpos)
    qpos /= qpos.norm(dim=-1, keepdim=True)

    # Encode generic negative query if included
    if qneg is not None:
      assert isinstance(qneg, list), "qneg argument should be list or None"
      if not len(qneg):
        qneg = self.NEGATIVE_PROMPT_GENERIC
      
      qneg = clip.tokenize(qneg).to(self.device)
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
  def predict(self, vis_feats, qpos, qneg=None, norm_vis_feat=None, method=None, threshold=None):
    method = method or self.method
    threshold = threshold or self.threshold
    norm_vis_feat = norm_vis_feat or self.norm_vis_feat

    if norm_vis_feat:
      vis_feats /= vis_feats.norm(dim=-1, keepdim=True)

    sims = self.compute_similarity(
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


