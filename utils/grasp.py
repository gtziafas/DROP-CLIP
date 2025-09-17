import numpy as np
import random
import copy
import cv2

import gripper_models
import utils.viz as viz


def generate_2D_masks(grasp_rectangles, H, W, width_factor=150):
    pos_out = np.zeros((H, W))
    ang_out = np.zeros((H, W))
    wid_out = np.zeros((H, W))
    for rect in grasp_rectangles:
        center_x, center_y, w_rect, h_rect, theta, _ = rect
        
        # Get 4 corners of rotated rect
        # Convert from our angle represent to opencv's
        r_rect = ((center_x, center_y), (w_rect/2, h_rect), -(theta+180))
        box = cv2.boxPoints(r_rect)
        box = np.intp(box)

        rr, cc = polygon(box[:, 0], box[:,1])

        mask_rr = rr < W
        rr = rr[mask_rr]
        cc = cc[mask_rr]

        mask_cc = cc < H
        cc = cc[mask_cc]
        rr = rr[mask_cc]


        pos_out[cc, rr] = 1.0
        ang_out[cc, rr] = theta * np.pi / 180
        # Adopt width normalize accoding to class 
        wid_out[cc, rr] = np.clip(w_rect, 0.0, width_factor) / width_factor
    
    qua_out = gaussian(pos_out, 3, preserve_range=True)
    ang_out = gaussian(ang_out, 2, preserve_range=True)
    wid_out = gaussian(wid_out, 3, preserve_range=True)
    
    return {'pos': pos_out, 
            'qua': qua_out, 
            'ang': ang_out, 
            'wid': wid_out
            }


def grasp_rects_to_tuples(grasp_rectangles):
    # grasp_rectangles: (M, 4, 2)
    grasp_rectangles = np.stack(grasp_rectangles, axis=0)
    M  = grasp_rectangles.shape[0]
    p1, p2, p3, p4 = np.split(grasp_rectangles, 4, axis=1)
    
    center_x = (p1[..., 0] + p3[..., 0]) / 2
    center_y = (p1[..., 1] + p3[..., 1]) / 2
    
    width  = np.sqrt((p1[..., 0] - p4[..., 0]) * (p1[..., 0] - p4[..., 0]) + (p1[..., 1] - p4[..., 1]) * (p1[..., 1] - p4[..., 1]))
    height = np.sqrt((p1[..., 0] - p2[..., 0]) * (p1[..., 0] - p2[..., 0]) + (p1[..., 1] - p2[..., 1]) * (p1[..., 1] - p2[..., 1]))
    
    theta = np.arctan2(p4[..., 0] - p1[..., 0], p4[..., 1] - p1[..., 1]) * 180 / np.pi
    theta = np.where(theta > 0, theta - 90, theta + 90)

    target = np.tile(np.array([[target]]), (M,1))

    return np.concatenate([center_x, center_y, width, height, theta, target], axis=1)


class Grasp2D:
    def __init__(self, center, angle, quality, width, height=None, deg=False):
        self.center = center
        self.theta = angle if deg else np.rad2deg(angle)
        self.q = quality
        self.width = width
        self.height = height or 2 * self.width

    def as_rect(self):
        center_x, center_y, width, height, theta = [int(x) for x in self.as_tuple()]
        box = ((center_x, center_y), (width, height), -(theta+180))
        box = cv2.boxPoints(box)
        box = np.intp(box)
        return box

    def as_tuple(self):
        return [self.center[0], self.center[1], self.width, self.height, self.theta]

    def __repr__(self):
        rep = f"Grasp center: {self.center[0], self.center[1]}\n"
        rep += f"Grasp angle: {self.theta}\n"
        rep += f"Grasp quality: {self.q}\n"
        rep += f"Grasp width: {self.width}\n"
        rep += f"Grasp height: {self.height}\n\n"
        return rep


class SceneGrasps2D:

    def __init__(self, grasps_input, input_type="dict"):
        if input_type == "rect":
            grasps_input = grasp_rects_to_tuples(grasps_input)
            self.grasps = [Grasp2D(g) for g in grasps_input]
        elif input_type == "dict":
            grasps_input = [
                (
                    g['center'], 
                    g['angle'],
                    g['quality'], 
                    g['width'], 
                    g['height'] if 'height' in g.keys() else None
                )
            for g in grasps_input]
            self.grasps = [Grasp2D(*g) for g in grasps_input]
    
    @property
    def centers(self):
        return [g.center for g in self.grasps]

    @property
    def angles(self):
        return [g.theta for g in self.grasps]

    @property
    def qualities(self):
        return [g.q for g in self.grasps]

    @property
    def widths(self):
        return [g.width for g in self.grasps]

    def __len__(self):
        return len(self.grasps)

    def get_rects(self):
        return [g.as_rect() for g in self.grasps]

    def get_masks(self, height, width, width_factor=150):
        rects = [g.as_rect() for g in self.get_rects()]
        out = generate_2D_masks(rects, height, width, width_factor)
        out = {k: v.squeeze() for k, v in out.items()}
        return out

    def __iter__(self):
        return self.__iter__([g for g in self.grasps])


class SceneGrasps:
    
    def __init__(self, indices, poses, scores, labels):
        self._poses = np.array(poses)
        self._labels = np.array(labels)
        self._scores = np.array(scores)
        self._indices = np.array(indices)

    @property
    def poses(self):
        return self._poses

    @property
    def scores(self):
        return self._scores

    @property
    def labels(self):
        return self._labels

    @property
    def indices(self):
        return self._indices

    @property
    def size(self):
        return self.__len__()
    
    def __len__(self):
        return self.poses.shape[0]

    def __iter__(self):
        return iter([self.poses, self.scores, self.labels, self.indices])

    def __repr__(self):
        rep = f"Grasp Poses: {self.poses}\n"
        rep += f"Scores: {self.scores}\n"
        rep += f"Labels: {self.labels}\n"
        return rep

    def _filter(self, filtered_indices):
        self._poses = self.poses[filtered_indices, ...]
        self._scores = self.scores[filtered_indices, ...]
        self._labels = self.labels[filtered_indices, ...]
        self._indices = self.indices[filtered_indices, ...]
        
    def filter(self, filtered_indices):
        poses = self.poses[filtered_indices, ...]
        scores = self.scores[filtered_indices, ...]
        labels = self.labels[filtered_indices, ...]
        indices = self.indices[filtered_indices, ...]
        return Grasps(indices, poses, scores, labels)

    def _filter_by_score(self, score_thresh):
        filt = np.argwhere(self.scores > 3 * score_thresh).squeeze()
        self._filter(filt)

    def filter_by_score(self, score_thresh):
        filt = np.argwhere(self.scores > 3 * score_thresh).squeeze()
        return self.filter(filt)

    def _filter_by_labels(self, obj_ids):
        if isinstance(obj_ids, int):
            obj_ids = [obj_ids]

        mask = np.zeros(self.__len__(), dtype=bool)
        for obj in obj_ids:
            mask[self.labels==obj] = True
        
        self._filter(mask)

    def filter_by_labels(self, obj_ids):
        if isinstance(obj_ids, int):
            obj_ids = [obj_ids]

        mask = np.zeros(self.__len__(), dtype=bool)
        for obj in obj_ids:
            mask[self.labels==obj] = True
        
        return self.filter(mask)
    
    def select_topk(self, k):
        order = np.argsort(self.scores)[::-1][:min(k, self.size)]
        return self._filter(order)
        
    def sample(self, population):
        choices = np.array(
            random.sample(range(self.size), min(population, self.size)))
        return self._filter(choices)

    def _select_topk(self, k):
        order = np.argsort(self.scores)[::-1][:min(k, self.size)]
        self._filter(order)
        
    def _sample(self, population):
        choices = np.array(
            random.sample(range(self.size), min(population, self.size)))
        self._filter(choices)

    def to_meshes(self, use_gripper_mesh=False, gripper_type="marker"):
        # render specific gripper type mesh
        if use_gripper_mesh:
            gripper_mesh = gripper_models.make(gripper_type)

        # just use a coord frame mesh
        else:
            gripper_mesh = viz.get_coord_frame(scale=0.05)
        
        meshes = [copy.deepcopy(gripper_mesh).transform(p) for p in self.poses]

        return meshes
