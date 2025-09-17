import copy
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from PIL import Image
import json
import os
import csv
import scipy
from io import StringIO
from collections import Counter
from tqdm import tqdm 
import h5py 
import warnings
warnings.filterwarnings("ignore")

from utils.geometry import *
from utils.transforms import *
from utils.viz import *
from utils.image import *

# import open3d.visualization as vis 

np.random.seed(31)

PALLETE = np.concatenate([np.zeros((1,3)), np.ones((1,3)) * 0.2 , np.random.rand(100, 3)], axis=0)
PALLETE_MAP = {k-1:v for k,v in enumerate(list(PALLETE))}


# views alignment pcd data <-> image data
# VIEWS_MAPPING = {
#     1: 9,
#     2: 4,
#     3: 6,
#     4: 5,
#     5: 1,
#     6: 3,
#     7: 2,
#     8: 7,
#     9: 8
# }


def obtain_seg_info(scene):
    seg_masks, all_obj_ids_2d = [], []
    for view_id, stuff in scene['views'].items():
        cls_ids, binary_masks, colors = zip(*stuff['annos'])
        global_ids = [col_to_ins_dict[x]["ins_id"] for x in colors]
        seg_ins_2d = imutils.binary_masks_to_seg(np.stack(binary_masks), np.asarray(global_ids))
        seg_masks.append(seg_ins_2d)
        all_obj_ids_2d.append(global_ids)
    return seg_masks, all_obj_ids_2d


def fuse_multiview_features(pc, multiview_features, camera_poses, camera_intrinsic):
    n_pts = pc.shape[0]
    feat_size = multiview_features.shape[-1]
    sum_features = torch.empty((n_pts, feat_size), dtype=float, device=multiview_features.device)
    for feat, camera_pose in zip(multiview_features, camera_poses):
        mapping = find_sth() # (M, 2)
        feat[mapping, :] # (M, C)


def get_sims_3d(feats_norm, qpos, background, qneg=False, thr=0.96):
    sims = CLIP.compute_similarity(feats_norm, qpos, qneg).squeeze().cpu().numpy()
    sims_norm = (sims - sims.min()) / (sims.max() - sims.min())
    cmap = plt.get_cmap("turbo")
    heatmap = cmap(sims_norm)[:,:3]
    sims_thr = background.copy()
    sims_thr[sims_norm > thr, 0] = 1.0
    sims_thr[sims_norm > thr, 1] = 0.0
    sims_thr[sims_norm > thr, 2] = 0.0
    return sims_thr, heatmap



def get_sims(text_query, clip_features, qneg=False):
    sims=[]
    for clip_feat in clip_features:
        clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
        clip_feat = einops.rearrange(clip_feat, "(h w) c -> h w c", h=24, w=24)
        clip_feat = reconstruct_feature_map(clip_feat, (840,840,3))
        sim = compute_similarity(clip_feat, qpos=text_query, qneg=qneg).squeeze()
        sims.append(sim)
        torch.cuda.empty_cache()
    return sims






def reconstruct_feature_map(feat, image_shape):
   H, W, _ = image_shape
   patch_h, patch_w, C = feat.shape
   scale_h, scale_w = patch_h / H, patch_w / W

   # Create a grid of coordinates in the original image
   y = torch.arange(H).unsqueeze(1).expand(H, W).float()  # Shape (H, W)
   x = torch.arange(W).unsqueeze(0).expand(H, W).float()  # Shape (H, W)

   # Scale the grid according to the feature map
   y_ = (y * scale_h).long()
   x_ = (x * scale_w).long()

   # Use the scaled coordinates to index into the feature map
   reconstructed = feat[y_, x_]

   return reconstructed


def load_multiview(root, split, scene_id):
    filepaths = sorted([os.path.join(dataset.root, split, "MultiviewDINOv2", f) for f in os.listdir(os.path.join(dataset.root, split, "MultiviewDINOv2"))])
    feats = {}
    for fp in filepaths:
        sid = fp.split('/')[-1].split('_')[0]
        vid = fp.split('/')[-1].split('_')[1].split('.')[0]
        if sid == scene_id:
            feats[int(vid)] = np.load(fp).reshape(60, 60, -1)
    return feats


def load_processed(root, split, scene_id, view_id):
    f = h5py.File(os.path.join(root, split, 'processed', f'{scene_id}_{view_id}.h5py'), 'r')
    return f


VIEWS_MAPPING = {
    1: 9,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8
}


def main(data_root, split, scene_ids=None):
    camera_info = np.load(os.path.join(data_root, "camera_info.npy"), allow_pickle=True).item()
    
    objects_json = json.load(open(f'{data_root}/{split}/objects_16k.json'))
    
    if scene_ids is None:
        scene_ids = next(os.walk(os.path.join(data_root, split, 'Points')))[1]
    
    stuff = {}
    for scene_id in scene_ids:
        #objs = json.load(open(f'{data_root}/{split}/{scene_id}/objects.json'))
        objs = objects_json[scene_id]

        stuff[scene_id] = {}
        filtered_cloud = None
        for view in range(1,10):
            data = np.load(f'{data_root}/{split}/Points/{scene_id}/{scene_id}_view_{view}.p', allow_pickle=True)
            img = Image.open(f'{data_root}/{split}/RGBImages/{scene_id}_{VIEWS_MAPPING[view]}.jpg').convert('RGB')
            
            if filtered_cloud is None:
                filtered_cloud = data['scene_cloud']

            l = np.array([PALLETE_MAP[x+1] for x in data['view_cloud_label']])   
        
            _objs = objs[str(view)]
            stuff[scene_id][view] = {
                'image': img, 
                'pc_xyz': data['view_cloud'],
                'pc_label': (data['view_cloud_label'] + 1),
                'pc_anno': l,
                'pc_rgb': data['view_cloud_color'],
                'state': []
            }
            for j, o in enumerate(_objs):
                stuff[scene_id][view]['state'].append({})
        
                pose = np.eye(4)
                pose[:3, :3] = R.from_quat(o['6D_pose'][3:]).as_matrix()
                pose[:3, -1] = np.array(o['6D_pose'][:3])
        
                pose_camera = np.matmul(camera_info['extrinsic'][view], pose)
        
                stuff[scene_id][view]['state'][j]['obj_id'] = o['obj_id']
                stuff[scene_id][view]['state'][j]['pose'] = pose_camera
                stuff[scene_id][view]['state'][j]['bbox'] = objs[str(VIEWS_MAPPING[view])][j]['bbox']
                stuff[scene_id][view]['state'][j]['label'] = o['model_name']
                stuff[scene_id][view]['state'][j]['model_id'] = o['model_id']
                stuff[scene_id][view]['state'][j]['category_id'] = o['category']

        pc_full = aggregate_views(stuff[scene_id])
        point_indices = find_closest_indices(pc_full['xyz'], filtered_cloud)
        stuff[scene_id]['pc_full'] = {'xyz': pc_full['xyz'][point_indices],
            'rgb': pc_full['rgb'][point_indices], 'label': pc_full['label'][point_indices],
            'anno': pc_full['anno'][point_indices]}
    return stuff             


def resolve_spatial_relation(source, target, relation):
    source_min_bound = source.get_min_bound()
    source_max_bound = source.get_max_bound()
    target_min_bound = target.get_min_bound()
    target_max_bound = target.get_max_bound()
    
    if relation == "left":
        return source_max_bound[0] < target_min_bound[0]
    elif relation == "right":
        return source_min_bound[0] > target_max_bound[0]
    elif relation == "behind":
        return (source_min_bound[0] <= target_max_bound[0] and
                source_max_bound[0] >= target_min_bound[0] and
                source_min_bound[1] > target_max_bound[1])
    elif relation == "front":
        return (source_min_bound[0] <= target_max_bound[0] and
                source_max_bound[0] >= target_min_bound[0] and
                source_min_bound[1] < target_min_bound[1])
    elif relation == "rear left":
        return source_max_bound[0] < target_min_bound[0] and source_min_bound[1] > target_max_bound[1]
    elif relation == "rear right":
        return source_min_bound[0] > target_max_bound[0] and source_min_bound[1] > target_max_bound[1]
    elif relation == "front left":
        return source_max_bound[0] < target_min_bound[0] and source_min_bound[1] < target_min_bound[1]
    elif relation == "front right":
        return source_min_bound[0] > target_max_bound[0] and source_min_bound[1] < target_min_bound[1]
    elif relation == "on":
        # Heuristic for 'on': source overlaps in x and y with target, and is above target
        return (source_min_bound[0] < target_max_bound[0] and
                source_max_bound[0] > target_min_bound[0] and
                source_min_bound[1] < target_max_bound[1] and
                source_max_bound[1] > target_min_bound[1] and
                source_max_bound[2] > target_max_bound[2])
    else:
        raise ValueError("Unknown relation")



def compute_box3d_iou(boxes):
    # Extract the min and max bounds of each box
    min_bounds = np.array([box.get_min_bound() for box in boxes])
    max_bounds = np.array([box.get_max_bound() for box in boxes])

    # Compute the intersection min and max bounds for all pairs
    int_min_bounds = np.maximum(min_bounds[np.newaxis, : :], min_bounds[:, np.newaxis, :])
    int_max_bounds = np.minimum(max_bounds[np.newaxis, : :], max_bounds[:, np.newaxis, :])

    # Compute the intersection volumes
    int_dims = np.maximum(0, int_max_bounds - int_min_bounds)
    int_volumes = np.prod(int_dims, axis=-1)

    # Compute the volumes of each box
    box_dims = max_bounds - min_bounds
    box_volumes = np.prod(box_dims, axis=-1)

    # Compute the union volumes for all pairs
    union_volumes = box_volumes[np.newaxis, :] + box_volumes[:, np.newaxis] - int_volumes

    # Compute the IoU for all pairs
    iou = int_volumes / union_volumes

    # Set the IoU of non-intersecting pairs to 0
    iou[int_volumes == 0] = 0

    return iou



def compute_pairwise_relations(boxes, 
                               frontal_threshold=0.3, 
                               vertical_threshold = 0.3, 
                               next_threshold=0.125,
                               iou_threshold=0.1
    ):
    # Extract bounds for all boxes
    min_bounds = np.array([box.get_min_bound() for box in boxes])
    max_bounds = np.array([box.get_max_bound() for box in boxes])
    centers = np.array([box.get_center() for box in boxes])

    # Calculate the pairwise comparisons between min and max bounds
    # x_overlap = np.logical_and(
    #     min_bounds[np.newaxis, :, 0] < centers[:, np.newaxis, 0],
    #     max_bounds[np.newaxis, :, 0] > min_bounds[:, np.newaxis, 0]
    # )
    # y_overlap = np.logical_and(
    #     min_bounds[np.newaxis, :, 1] < centers[:, np.newaxis, 1],
    #     max_bounds[np.newaxis, :, 1] > min_bounds[:, np.newaxis, 1]
    # )
    x_overlap = ~ np.logical_or(
        min_bounds[np.newaxis, :, 0] > max_bounds[:, np.newaxis, 0],
        max_bounds[np.newaxis, :, 0] < min_bounds[:, np.newaxis, 0]
    )
    y_overlap = ~ np.logical_or(
        min_bounds[np.newaxis, :, 1] > max_bounds[:, np.newaxis, 1],
        max_bounds[np.newaxis, :, 1] < min_bounds[:, np.newaxis, 1]
    )
    np.fill_diagonal(x_overlap, False)
    np.fill_diagonal(y_overlap, False)

    # Calculate pair-wise Intersection-over-Union for 3D boxes
    ious = compute_box3d_iou(boxes)

    # Initialize a dictionary to store the results
    results = {}

    # Compute pairwise relations using vectorized operations
    results["left"] = centers[np.newaxis, :, 0] < min_bounds[:, np.newaxis, 0]
    results["right"] = centers[np.newaxis, :, 0] > max_bounds[:, np.newaxis, 0]
    results["behind"] = np.logical_and.reduce((
        x_overlap,
        #~y_overlap,
        np.abs(centers[np.newaxis, :, 1] - centers[:, np.newaxis, 1]) <= frontal_threshold,
        max_bounds[:, np.newaxis, 0] > centers[np.newaxis, :, 0],
        centers[np.newaxis, :, 0] > min_bounds[:, np.newaxis, 0],
        min_bounds[np.newaxis, :, 1] > centers[:, np.newaxis, 1],
        #centers[np.newaxis, :, 1] 
    ))
    results["front"] = np.logical_and.reduce((
        x_overlap,
        #~y_overlap,
        np.abs(centers[np.newaxis, :, 1] - centers[:, np.newaxis, 1]) <= frontal_threshold,
        max_bounds[:, np.newaxis, 0] > centers[np.newaxis, :, 0],
        centers[np.newaxis, :, 0] > min_bounds[:, np.newaxis, 0],
        max_bounds[np.newaxis, :, 1] < centers[:, np.newaxis, 1]
    ))
    results["rear_left"] = np.logical_and.reduce((
        #~x_overlap,
        #~y_overlap,
        np.abs(centers[np.newaxis, :, 0] - centers[:, np.newaxis, 0]) <= vertical_threshold,
        np.abs(centers[np.newaxis, :, 1] - centers[:, np.newaxis, 1]) <= vertical_threshold,
        #np.linalg.norm(centers[np.newaxis, :, :] - centers[:, np.newaxis, :], axis=2) < vertical_threshold,
        centers[np.newaxis, :, 0] < min_bounds[:, np.newaxis, 0],
        centers[np.newaxis, :, 1] > max_bounds[:, np.newaxis, 1],
        min_bounds[np.newaxis, :, 1] > min_bounds[:, np.newaxis, 1]
    ))
    results["rear_right"] = np.logical_and.reduce((
        #~x_overlap,
        #~y_overlap,
        np.abs(centers[np.newaxis, :, 0] - centers[:, np.newaxis, 0]) <= vertical_threshold,
        np.abs(centers[np.newaxis, :, 1] - centers[:, np.newaxis, 1]) <= vertical_threshold,
        #np.linalg.norm(centers[np.newaxis, :, :] - centers[:, np.newaxis, :], axis=2) < vertical_threshold,
        centers[np.newaxis, :, 0] > max_bounds[:, np.newaxis, 0],
        centers[np.newaxis, :, 1] > max_bounds[:, np.newaxis, 1],
        min_bounds[np.newaxis, :, 1] > min_bounds[:, np.newaxis, 1]
    ))
    results["front_left"] = np.logical_and.reduce((
        #~x_overlap,
        #~y_overlap,
        np.abs(centers[np.newaxis, :, 0] - centers[:, np.newaxis, 0]) <= vertical_threshold,
        np.abs(centers[np.newaxis, :, 1] - centers[:, np.newaxis, 1]) <= vertical_threshold,
        #np.linalg.norm(centers[np.newaxis, :, :] - centers[:, np.newaxis, :], axis=2) < vertical_threshold,
        centers[np.newaxis, :, 0] < min_bounds[:, np.newaxis, 0],
        centers[np.newaxis, :, 1] < min_bounds[:, np.newaxis, 1],
        max_bounds[np.newaxis, :, 1] < max_bounds[:, np.newaxis, 1]
    ))
    results["front_right"] = np.logical_and.reduce((
        #~x_overlap,
        #~y_overlap,
        np.abs(centers[np.newaxis, :, 0] - centers[:, np.newaxis, 0]) <= vertical_threshold,
        np.abs(centers[np.newaxis, :, 1] - centers[:, np.newaxis, 1]) <= vertical_threshold,
        #np.linalg.norm(centers[np.newaxis, :, :] - centers[:, np.newaxis, :], axis=2) < vertical_threshold,
        centers[np.newaxis, :, 0] > max_bounds[:, np.newaxis, 0],
        centers[np.newaxis, :, 1] < min_bounds[:, np.newaxis, 1],
        max_bounds[np.newaxis, :, 1] < max_bounds[:, np.newaxis, 1]
    ))
    results["on"] = np.logical_and.reduce((
        ious > iou_threshold,
        min_bounds[np.newaxis, :, 0] < centers[:, np.newaxis, 0],
        max_bounds[np.newaxis, :, 0] > centers[:, np.newaxis, 0],
        min_bounds[np.newaxis, :, 1] < centers[:, np.newaxis, 1],
        max_bounds[np.newaxis, :, 1] > centers[:, np.newaxis, 1],
        max_bounds[np.newaxis, :, 2] > max_bounds[:, np.newaxis, 2]
    ))
    results["next"] = np.logical_and.reduce((
        ious <= 0.35,
        np.linalg.norm(centers[np.newaxis, :, :] - centers[:, np.newaxis, :], axis=2) < next_threshold
    ))
    # Set diagonal to False since an object cannot have a spatial relation with itself
    for relation in results.keys():
        np.fill_diagonal(results[relation], False)

    return results

def get_targets(relations, R, i): return [x for x in np.where(relations[R][i])[0].tolist()]




# def extract_shapenetcore_entries(data):
#     def extract(entry, results, depth, parents):
#         ID = entry['metadata']['name']
#         label = entry['metadata']['label']
#         title = entry['li_attr']['title']
#         text = entry['text']

#         if ID not in results:
#             results[ID] = {'parents': parents.copy(), 'depth': depth}
#         results[ID]['label'] = label
#         results[ID]['title'] = title
#         results[ID]['text'] = text

#         for child in entry.get('children', []):
#             extract(child, results, depth + 1, parents + [ID])

#     results = {}
#     for entry in data:
#         extract(entry, results, 0, [])
#     return results


def extract_shapenetcore_metadata(metadata_dir):
    
    def split_string(s):
        # Use csv.reader to handle commas inside quotes
        f = StringIO(s)
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            return row

    txtfiles = [f for f in os.listdir(metadata_dir) if f.endswith('txt') and f not in ['bowl.txt', 'bicycle.txt']]
    data = {}
    for txtfile in txtfiles:
        with open(os.path.join('ShapeNetCore-metadata', txtfile), 'r') as f:
            lines = f.readlines()
            k = txtfile.split('.txt')[0]
            model_name = ' '.join(k.split('_'))
            for l in lines[1:]:
                toks = split_string(l)
                model_id = toks[0].split('.')[1]
                if model_id in data.keys() and model_name != data[model_id]['model_name']:
                    data[model_id]['model_name'].append(model_name)
                    continue
                categories = toks[1]
                names = toks[2]
                special = toks[-2]
                data[model_id] = {
                    'model_id': model_id, 
                    'category_id': categories, 
                    'attributes': names, 
                    'special': special, 
                    'model_name': [model_name]
                }
    return data


def make_regrad_concept_catalog(datasets):
    
    not_found = {}; unique = {}
    for split, dataset in zip(['train','seen_val','unseen_val'], datasets):
        not_found[split] = {}
        unique[split] = {}
        for k, views in dataset.items():
            objs = views['1'] # same always
            #IDs = [x['model_id'] for x in objs]
            for o in objs:
                i = o['model_id']
                #if i not in metadata.keys() and i not in not_found.keys():
                _e = {'model_id': i,
                    'label': o['model_name'],
                    'category_id': o['category'],
                    'parents': o['parent_list'],
                    'source': o['source']}
                if i not in metadata.keys() and i not in not_found.keys():
                    not_found[split][i] = _e
    
                else:
                    if i not in unique and i in metadata.keys():
                        unique[split][i] = _e

    catalog = []
    for split in unique.keys():
        for model_id, entry in unique[split].items():
            #print(entry)
            #print(metadata[model_id])
            assert entry['category_id'] == metadata[model_id]['category_id'], (entry['category_id'], metadata[model_id]['category_id'])
            catalog.append({
                'model_id': model_id,
                'synset_id': entry['category_id'],
                'category': entry['label'],
                'attributes': metadata[model_id]['attributes'],
                'special': metadata[model_id]['special'],
                'parents': entry['parents'],
                'source': entry['source']}
            )
        for model_id, entry in not_found[split].items():
            assert entry['category_id'] == metadata[model_id]['category_id'],(entry['category_id'], metadata[model_id]['category_id'])
            catalog.append({
                'model_id': model_id,
                'synset_id': entry['category_id'],
                'category': entry['label'],
                'attributes': None,
                'special': None,
                'parents': entry['parents'],
                'source': entry['source']}
             )
    return catalog



# COLOR CLASSIFICATION
from scipy.signal import find_peaks
from sklearn.neighbors import KNeighborsClassifier

# Define the fixed color points
color_points = {
    'red': [
        [128, 0, 0],
        [139, 0, 0],
        [165, 42, 42],
        [178, 34, 34],
        [220, 20, 60],
        [255, 0, 0],
        [255, 69, 0]
    ],
    'orange': [
        [255, 99, 71],
        [255, 127, 80],
        [255, 140, 0],
        [255, 165, 0],
    ],
    'yellow': [
        [255, 215, 0],
        [255, 255, 0],
        [217, 165, 32],
        [207, 185, 63],
        [221, 203, 35],
    ],
    'green': [
        [154, 205, 50],
        [85, 107, 47],
        [107, 142, 35],
        [124, 252, 0],
        [173, 255, 47],
        [0, 100, 0],
        [0, 128, 0],
        [34, 139, 34],
        [0, 255, 0],
        [50, 205, 50],
        [46, 139, 87],
        [60, 179, 113]
    ],
    'black': [
        [0, 0, 0],
        [10, 10, 10],
        [20, 20, 20],
        [45, 45, 45]
    ],
    'gray': [
        [75, 75, 75],
        [128, 128, 128],
        [105, 105, 105],
        [150, 150, 150],
        [179, 179, 179]
    ],
    'white': [
        [222, 222, 222],
        [245, 245, 245],
        [200, 200, 200],
        [255, 255, 255],
    ],
    'pink': [
        [221, 160, 221],
        [255, 182, 193],
        [255, 192, 203],
        [255, 20, 147],
        #[188, 162, 188 ]
    ],
    'purple': [
        [75, 0, 130],
        [128, 0, 128],
        [186, 85, 211],
        [153, 50, 204],
        [148, 0, 211],
        [139, 0, 139],
        [147, 112, 219],
    ],
    'brown': [
        [139, 69, 19],
        [160, 82, 45],
        [210, 105, 30],
        [205, 133, 63],
        [207, 159, 119],
        [100, 50, 35]
    ],
    'blue': [
        [25, 25, 112],
        [0, 0, 128],
        [0, 0, 205],
        [0, 0, 255],
        [65, 105, 225],
        [70, 130, 180],
        [100, 149, 237],
        [30, 144, 255],
        [121, 124, 230]
    ],
    'cyan': [
        [0, 191, 255],
        [0, 255, 255],
        [0, 206, 209],
        [64, 224, 208],
        [72, 209, 204]
    ]
}

class ColorClassifier:
    def __init__(self, n_neighbors=1, n_bins=256, method="expected"):
        # Prepare data for k-NN classifier
        self.n_bins = n_bins
        X_train = []
        y_train = []
        for color_name, color_pts in color_points.items():
            for color in color_pts:
                #hist = self.compute_histogram(np.array([color]))
                #X_train.append(hist)
                X_train.append(np.float32(color) / 255.)
                y_train.append(color_name)

        # Create and train k-NN classifier
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.knn.fit(X_train, y_train)
        self.method = method

    def expected_color(self, img):
        if len(img.shape) > 2:
            img = np.reshape(img, (-1, 3))
        n_pts = img.shape[0]

        hist = self.compute_histogram(img)
        hist_r, hist_g, hist_b = np.split(hist, 3)

        # convert histogram to prob densities and take expectation
        exp_r = np.sum(np.arange(self.n_bins) * (hist_r / n_pts))
        exp_g = np.sum(np.arange(self.n_bins) * (hist_g / n_pts))
        exp_b = np.sum(np.arange(self.n_bins) * (hist_b / n_pts))

        return [exp_r, exp_g, exp_b]

    def most_dominant_color(self, img, range_width=10):
        if len(img.shape) > 2:
            img = np.reshape(img, (-1, 3))

        # most frequent color
        res = np.unique(img, axis=0, return_counts=True)
        dominant = res[0][np.argmax(res[1]), :]

        lower_bound = np.array([dominant[0] - range_width, dominant[1] - range_width, dominant[2] - range_width])
        upper_bound = np.array([dominant[0] + range_width, dominant[1] + range_width, dominant[2] + range_width])

        # Clip the bounds to be within valid color range
        lower_bound = np.clip(lower_bound, 0, 255)
        upper_bound = np.clip(upper_bound, 0, 255)

        # Find all points within the range
        points_in_range = np.all((img >= lower_bound) & (img <= upper_bound), axis=-1)

        # If no points return just average
        if points_in_range.sum() > 0:
            return img[points_in_range].mean(0)
        else:
            return img.mean(0)

    def compute_histogram(self, color):
        hist_sampled_r = np.histogram(color[:, 0], bins=self.n_bins, range=(0, 255))[0] 
        hist_sampled_g = np.histogram(color[:, 1], bins=self.n_bins, range=(0, 255))[0] 
        hist_sampled_b = np.histogram(color[:, 2], bins=self.n_bins, range=(0, 255))[0] 
        hist_sampled_rgb = np.concatenate((hist_sampled_r, hist_sampled_g, hist_sampled_b))
        return hist_sampled_rgb

    def predict(self, images, masks):
        # Isolate the RGB points inside the mask for each object
        masked_images = [image[mask == True] for image, mask in zip(images, masks)]

        if self.method == "histogram":
            X_test = np.array([self.compute_histogram(image) for image in masked_images]) / 255.
        elif self.method == "average":
            X_test = np.array([image.mean(0) for image in masked_images]) / 255.
        elif self.method == "dominant":
            X_test = np.array([self.most_dominant_color(image) for image in masked_images]) / 255.
        elif self.method == "expected":
            X_test = np.array([self.expected_color(image) for image in masked_images]) / 255.

        # Predict the color
        predicted_color = self.knn.predict(X_test)
        return predicted_color


def predict_colors_dataset(stuff, skip=[]):
    results = {}
    for scene_id, scene in stuff.items():
        #print(scene_id)
        #print()
        if scene_id in skip:
           continue
        scene_t = transform_scene_to_camera_frame(scene, camera_info)
        objIDs = [o['obj_id'] for o in scene_t[1]['state']] #objIDs same accross views
        predictions = []
        for v in range(1,10):
            #print(f'View {v}')
            img = np.array(scene_t[v]['image'].copy())
            bboxes = [o['bbox'] for o in scene_t[v]['state']]
            seg = cv2.imread(os.path.join(DATA_ROOT, split, 'SegmentationImages', f'{scene_id}_{VIEWS_MAPPING[v]}.png'), cv2.IMREAD_UNCHANGED)
            _objIDs = [o['obj_id'] for o in scene_t[v]['state']] #objIDs same accross views
            assert objIDs == _objIDs
            masks = [seg==objID for objID in objIDs]
            _images = [img.copy()[y1:y2,x1:x2] for x1,y1,x2,y2 in bboxes]
            _masks = [m.copy()[y1:y2,x1:x2] for (x1,y1,x2,y2), m in zip(bboxes, masks)]
            predicted_color = CC.predict(_images, _masks)
            predictions.append(predicted_color)
        majority_vote = [Counter(col).most_common(1)[0][0] for col in zip(*predictions)]
        results[scene_id] = {objID: vote for obID, vote in zip(objIDs, majority_vote)}
    return results


def test_relations_multiview(scene_t, skip=[]):
    _, _, all_boxes, _, all_meshes = extract_geometries_multiview(scene_t)
    _, scene_objIDs, scene_labels = zip(*[(j, x['obj_id'], x['model_name']) for j,x in enumerate(scene_t['state'])])
    all_rels = [compute_pairwise_relations(all_boxes[v]) for v in range(10)]
    for view in range(1,10):
        _pc_anno = to_o3d(scene_t['views'][view]['pc_xyz'], scene_t['views'][view]['pc_rgb'])
        print(f'Doing view {view}')
        for rel in all_rels[view].keys():
            if rel in skip:
                continue
            print(rel)
            for source in range(len(scene_labels)):
                targets = get_targets(all_rels[view], rel, source)
                draw_meshes = paint_meshes_rel(all_meshes[view], source, targets)
                ii = paint_image_rel(
                    scene_t['views'][view]['image'], 
                    scene_t['views'][view]['RGB_boxes'],
                    scene_objIDs[source], 
                    [scene_objIDs[t] for t in targets]
                )
                imshow(ii)
                #Image.fromarray(ii).show()
                o3d_viewer([_pc_anno] + draw_meshes)
        print('---' * 48)



