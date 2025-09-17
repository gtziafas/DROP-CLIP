import torch
import torch.nn.functional as F
import numpy as np
import gc
from typing import *

import utils.transforms as tutils
from utils.geometry import remove_invisible_points
#import utils.projections as proj
from models.similarity import ClipSimilarity
from models.features.clip import tokenize 



class MultiviewFeatureFusion:
	def __init__(
		self,
		camera_intrinsic: Dict[str, float],
		visibility_threshold: float = 0.05,
		image_size: Tuple[float] = (480, 640),
		patch_size: int = 14,
		feature_size: int = 768,
		use_visibility: bool = True,
		use_similarity: bool = True,
		use_sim_kernel: Optional[str] = None,
		use_obj_prior: bool = True, 
		norm_feat: bool = True,
		device='cuda'
	):
		self.visibility_threshold = visibility_threshold
		self.height, self.width = image_size
		self.feature_size = feature_size
		self.patch_size = patch_size
		self.camera_intrinsic = camera_intrinsic
		self.K = np.asarray([
			[camera_intrinsic['fx'], 0, camera_intrinsic['cx']],
			[0, camera_intrinsic['fy'], camera_intrinsic['cy']],
			[0,  0,  1]

		])
		self.device = device
		self.use_obj_prior = use_obj_prior
		self.norm_feat = norm_feat
		self.use_visibility = use_visibility
		self.use_similarity = use_similarity
		if self.use_similarity:
			assert use_sim_kernel is not None, "Remember to set similarity kernel for `use_similarity=True`"
			#self.sim_kernel = self.set_sim_kernel(use_sim_kernel)
			self.sim_method = use_sim_kernel

		# 2D coordinate transforms
		self.coord_tf = tutils.CoordTransform2d(
			image_size, patch_size)

	# def set_sim_kernel(self, method, eps=1e-6):
	# 	if method == "max":
	# 		return lambda pos, neg: torch.clip(
	# 			pos - torch.max(neg, dim=-1)[0], eps).squeeze().float() 
	# 	elif method == "mean":
	# 		return lambda pos, neg: torch.clip(
	# 			pos - neg.mean(-1), eps).squeeze().float()
	# 	else:
	# 		raise ValueError('Please set method in [mean, max]')

	def calculate_sim(self, pos, neg, eps=1e-6):
		if self.sim_method == 'max':
			return torch.clip(
				pos - torch.max(neg, dim=-1)[0], eps).squeeze().float() 
		elif self.sim_method == 'mean':
			return torch.clip(
				pos - neg.mean(-1), eps).squeeze().float()
		else:
			raise ValueError('Please set method in [mean, max]')

	@staticmethod
	def _cvt_o3d_coords(pts):
		pts[:, 1] = -pts[:, 1]
		pts[:, 2] = -pts[:, 2]
		return pts

	def get_visibility_mask(self, points, depths, camera_poses, device=None):
		K = self.K
		device = device or self.device
		n_pts = points.shape[0]
		n_views = len(depths)
		visibility_mask = torch.zeros((n_views, n_pts), dtype=int)
		
		for v, (depth, camera_pose) in enumerate(list(zip(depths, camera_poses))):

			projected_points = np.zeros((n_pts, 2), dtype=int)

			# *******************************************************************************************************************
			# STEP 1: get the projected points
			# Get the coordinates of the projected points in the i-th view (i.e. the view with index idx)
			points_camera = tutils.transform_pointcloud_to_camera_frame(
				points, camera_pose)
			points_camera = self._cvt_o3d_coords(points_camera)
			projected_points_not_norm = (self.K @ points_camera.T).T

			# Get the mask of the points which have a non-null third coordinate to avoid division by zero
			mask = (projected_points_not_norm[:, 2] != 0) # don't do the division for point with the third coord equal to zero
			# Get non homogeneous coordinates of valid points (2D in the image)
			projected_points[mask] = np.column_stack([[projected_points_not_norm[:, 0][mask]/projected_points_not_norm[:, 2][mask], 
				projected_points_not_norm[:, 1][mask]/projected_points_not_norm[:, 2][mask]]]).T
			projected_points = torch.from_numpy(projected_points).to(device)

			sensor_depth = torch.from_numpy(depth.copy()).to(device)
			point_depth = torch.from_numpy(projected_points_not_norm[:,2]).to(device)
			pi = projected_points.T
			inside_mask = (projected_points[:,0] >= 0) * (projected_points[:,1] >= 0) \
								* (projected_points[:,0] < self.width) \
								* (projected_points[:,1] < self.height)

			# *******************************************************************************************************************
			# STEP 2: occlusions computation
			# Depth of the points of the pointcloud, projected in the i-th view, computed using the projection matrices
			# Compute the visibility mask, true for all the points which are visible from the i-th view
			visible_points_view = ((torch.abs(sensor_depth[pi[1][inside_mask], pi[0][inside_mask]]
											- point_depth[inside_mask]) <= \
											self.visibility_threshold)).bool()
			inside_mask[inside_mask == True] = visible_points_view
			
			visibility_mask[v] = inside_mask

		return visibility_mask

	@staticmethod
	def reconstruct_per_obj_feat(pc, label, feat, obj_ids):
		with torch.no_grad():
			mv_pc = torch.zeros((pc.shape[0], feat.shape[-1]), dtype=torch.float32)
			for i, obj in enumerate(obj_ids):
				if i == 0:
					continue
				ids = np.argwhere(label == obj)
				mv_pc[ids, :] = feat[i] 
		return mv_pc
	
	@torch.no_grad()
	def aggregate_features(
		self, 
		points, 
		depths, 
		seg_masks,
		camera_poses, 
		mv_features, 
		query_embeddings=None,
		device=None
	):
		device = device or self.device
		K = self.K
		n_pts = points.shape[0]
		n_views = len(depths)
		
		visibility_mask = torch.zeros((n_views, n_pts), dtype=torch.long, device=device)
		
		similarity_mask = None
		if self.use_similarity:
			assert query_embeddings is not None, 'Must provide query embeddings for using similarity.'
			similarity_mask = torch.zeros((n_views, n_pts), dtype=torch.float32, device=device)

		sum_features = torch.zeros((n_pts, self.feature_size), dtype=torch.float32, device=device)
		
		for v, (depth, camera_pose, feat2d, seg) in enumerate(list(zip(depths, camera_poses, mv_features, seg_masks))):
			# *******************************************************************************************************************
			# STEP 0 (Optional): get similarity maps from 2D feature maps
			# Upsample feature maps to original resolution
			feat2d = F.interpolate(
				feat2d.permute(2,0,1).unsqueeze(0), 
				size=(self.height, self.width), 
				mode="bicubic", 
				align_corners=False
			).squeeze().permute(1,2,0)
			if self.norm_feat:
				feat2d /= feat2d.norm(dim=-1, keepdim=True)

			if self.use_similarity:			
				# Compute similarity map for each feature with all scene queries
				raw_sim_map = feat2d.float() @ query_embeddings.T # (H, W, Q)
				sim_metric = torch.zeros((self.height, self.width), dtype=torch.float32, device=device)
				
				# Iterate over object IDs to compute metric at each region
				for obj_id in range(len(query_embeddings)):
					obj_mask_2d = seg == obj_id
					sim_map_obj = raw_sim_map[obj_mask_2d] # (h',w',1)

					# obtain positive - negative queries for obj ID
					qpos_idx = obj_id
					qneg_idx = torch.as_tensor(
						[obj for obj in range(len(query_embeddings)) if obj != obj_id]
					).long().to(device)

					sim_pos = sim_map_obj[:, qpos_idx] # (Nobj, 1)
					sim_neg = sim_map_obj[:, qneg_idx] # (Nobj, Q-1)

					# compute metric based on relative similarity kernel
					sim_metric[obj_mask_2d] = self.calculate_sim(sim_pos, sim_neg) # (Nobj,)

			# *******************************************************************************************************************
			# STEP 1: get the projected points
			# Get the coordinates of the projected points in the i-th view (i.e. the view with index idx)
			projected_points = np.zeros((n_pts, 2), dtype=int)
			points_camera = tutils.transform_pointcloud_to_camera_frame(
				points, camera_pose)
			points_camera = self._cvt_o3d_coords(points_camera)
			projected_points_not_norm = (self.K @ points_camera.T).T
			
			# Get the mask of the points which have a non-null third coordinate to avoid division by zero
			mask = (projected_points_not_norm[:, 2] != 0) # don't do the division for point with the third coord equal to zero
			# Get non homogeneous coordinates of valid points (2D in the image)
			projected_points[mask] = np.column_stack([[projected_points_not_norm[:, 0][mask]/projected_points_not_norm[:, 2][mask], 
				projected_points_not_norm[:, 1][mask]/projected_points_not_norm[:, 2][mask]]]).T
			projected_points = torch.from_numpy(projected_points).to(device)

			sensor_depth = torch.from_numpy(depth.copy()).to(device)
			point_depth = torch.from_numpy(projected_points_not_norm[:,2]).to(device)
			pi = projected_points.T
			inside_mask = (projected_points[:,0] >= 0) * (projected_points[:,1] >= 0) \
								* (projected_points[:,0] < self.width) \
								* (projected_points[:,1] < self.height)

			# *******************************************************************************************************************
			# STEP 2: occlusions computation
			# Depth of the points of the pointcloud, projected in the i-th view, computed using the projection matrices
			# Compute the visibility mask, true for all the points which are visible from the i-th view
			visible_points_view = ((torch.abs(sensor_depth[pi[1][inside_mask], pi[0][inside_mask]]
										- point_depth[inside_mask]) <= \
										self.visibility_threshold)).bool()
			inside_mask[inside_mask == True] = visible_points_view
			visibility_mask[v] = inside_mask

			# *******************************************************************************************************************
			# STEP 3: project 2D features to 3D based on visibility/similarity mask
			indices = (inside_mask == 1).nonzero(as_tuple=True)[0]
			_pi = pi[:, indices]
			xs, ys = _pi[0,:], _pi[1,:]
			
			if self.use_similarity:
				similarity_mask[v][inside_mask==1] = sim_metric[ys, xs]
				feat3d = feat2d[ys, xs] * sim_metric[ys, xs].unsqueeze(1)
			else:
				feat3d = feat2d[ys, xs]
			
			sum_features[inside_mask==1] += feat3d

			del feat3d
			if device == 'cuda':
				torch.cuda.empty_cache()
			# gc.collect()

		return sum_features, visibility_mask, similarity_mask

	def fuse_points(self, points, colors, labels, depths, seg_masks, camera_poses, mv_features, query_embeddings, device=None):
		sum_features, visibility_mask, similarity_mask = self.aggregate_features(
			points, depths, seg_masks, camera_poses, mv_features, query_embeddings, device)

		# remove invisible points
		_visible_mask = visibility_mask.sum(0) > 0
		points = points[_visible_mask.cpu().numpy()]
		colors = colors[_visible_mask.cpu().numpy()]
		labels = labels[_visible_mask.cpu().numpy()]
		visibility_mask = visibility_mask[:, _visible_mask]
		sum_features = sum_features[_visible_mask]
		if self.use_similarity:
			similarity_mask = similarity_mask[:, _visible_mask]
		
		dividend = visibility_mask.sum(0) if not self.use_similarity else similarity_mask.sum(0)
	
		sum_features /= dividend.unsqueeze(1)

		return (sum_features, visibility_mask, similarity_mask), (points, colors, labels)

	@torch.no_grad()
	def fuse_obj_prior(self, points, colors, labels, depths, seg_masks, camera_poses, mv_features, query_embeddings, return_obj=False, device=None):
		visibility_mask = self.get_visibility_mask(points, depths, camera_poses)
		
		# remove invisible points
		_visible_mask = visibility_mask.sum(0) > 0
		points = points[_visible_mask.cpu().numpy()]
		colors = colors[_visible_mask.cpu().numpy()]
		labels = labels[_visible_mask.cpu().numpy()]
		visibility_mask = visibility_mask[:, _visible_mask]

		device = device or self.device
		
		# obtain per-object and per-view feature 
		n_objects = query_embeddings.shape[0] # incl. table
		#n_objects = max([np.unique(seg).max() for seg in seg_masks]) + 1 # incl. table
		n_views = len(mv_features)
		
		mv_feats_obj = torch.zeros(
			(n_objects, n_views, 768), dtype=torch.float32, device=device)
		weight_obj = torch.zeros(
			(n_objects, n_views), dtype=torch.float32, device=device)

		# if self.use_visibility:
		# 	vis_weight_obj = torch.zeros(
		# 		(n_objects, n_views), dtype=torch.float32, device=device)

		# if self.use_similarity:
		# 	sim_weight_obj = torch.zeros(
		# 		(n_objects, n_views), dtype=torch.float32, device=device)

		for v in range(n_views):
			feat_v = mv_features[v]
			seg = seg_masks[v]

			obj_ids_2d = np.unique(seg)[1:].tolist()
			#query_embeddings_v = query_embeddings[torch.as_tensor(obj_ids_2d).to(device)]

			if self.use_similarity:
				feat_v_norm = feat_v / feat_v.norm(dim=-1, keepdim=True)
				sim_map = feat_v_norm.float() @ query_embeddings.T #(K,Q)
				sim_map_norm = (sim_map - sim_map.min())/(sim_map.max() - sim_map.min())
			
			for i, obj in enumerate(obj_ids_2d):

				weight_obj[obj, v] = 1.0
				
				if self.use_visibility:
					n_pts_obj_in_view = (seg == obj).sum()
					weight_obj[obj, v] = float(n_pts_obj_in_view)

				if self.use_similarity:
					qpos_idx = obj
					qneg_idx = torch.as_tensor(
						[o for o in range(len(query_embeddings)) if o != obj]).long().to(device)
					
					sim_obj_v = sim_map_norm[i] # (Q,)
					sim_obj_v = self.calculate_sim(sim_obj_v[qpos_idx], sim_obj_v[qneg_idx])

					weight_obj[obj, v] = sim_obj_v.item()
				
				mv_feats_obj[obj, v] = feat_v[i]
		
		mv_feats_obj = torch.einsum("kvc,kv->kc", mv_feats_obj, weight_obj) /  weight_obj.sum(1).unsqueeze(-1)

		if not return_obj:
			mv_feats = self.reconstruct_per_obj_feat(
				points, labels, mv_feats_obj.float().cpu(), list(range(n_objects)))
		else:
			mv_feats = mv_feats_obj

		return (mv_feats, weight_obj, visibility_mask), (points, colors, labels)

	@torch.no_grad()
	def fuse(self, *args, **kwargs):
		if self.use_obj_prior:
			return self.fuse_obj_prior(*args, **kwargs)
		else:
			return self.fuse_points(*args, **kwargs)