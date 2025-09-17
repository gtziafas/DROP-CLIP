import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import glob
import json
import h5py
import trimesh
import numpy as np
import torch.utils.data as data
from pycocotools import mask as maskUtils
import utils.image as imutils

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)


class BlenderDataset(data.Dataset):
    def __init__(self, root, models_root, grasp_root, split):
        self.root = root
        self.split = split
        self.models_root = models_root
        self.grasp_root = grasp_root
        self.scene_ids = sorted(next(os.walk(os.path.join(self.root, self.split)))[1])
        print(f"{len(self.scene_ids)} scenes in total")
        

        self.metadata = json.load(open(os.path.join(root, 'annos.meta.coco.json')))
        # self.names_map = json.load(open(os.path.join(root, 'name_prompt_map.json')))
        # self.prompt_embeddings_map = np.load(os.path.join(root, 'Prompt_Emb_Blender.npy'), allow_pickle=True).item()
        self.cls_embedding_table = np.load(os.path.join(root, 'cls_embedding_table.npy'))
        self.id_to_name = {0: 'table', 
                      **{x['id'] + 1 : x['name'] for x in self.metadata['categories']}}
        self.name_to_id = {v:k for k, v in self.id_to_name.items()}
        # self.id_to_embedding = {0: self.prompt_embeddings_map['table'], 
        #               **{x['id'] + 1: self.prompt_embeddings_map[self.id_to_name[x['id']]] for x in self.metadata['categories']}}
        
        self.name_to_id = {v:k for k, v in self.id_to_name.items()}    

        # self.camera_intrinsic = {
        #   'height': 480,
        #   'width': 640,
        #   'fx': 444.4,
        #   'fy': 444.4,
        #   'cx': 320.0,
        #   'cy': 240.0
        # }

    def read_rgb_image(self, fpath):
        return np.ascontiguousarray(cv2.imread(fpath)[:, :, ::-1])
    
    
    def read_seg_image(self, fpath):
        return cv2.imread(fpath)
    

    def read_depth_image(self, fpath):
        # OpenEXR file provides 
        return cv2.imread(fpath, cv2.IMREAD_UNCHANGED)[:, :, 0].astype(np.float32)


    def load_json(self, fpath):
        return json.load(open(fpath, "r"))


    def anno_to_mask(self, anno, h, w):
        def anno_to_rle(anno, h, w):
            segm = anno['segmentation']
            if type(segm) == list:
                # polygon -- a single object might consist of multiple parts
                # we merge all parts into one mask rle code
                rles = maskUtils.frPyObjects(segm, h, w)
                rle = maskUtils.merge(rles)
            elif type(segm['counts']) == list:
                # uncompressed RLE
                rle = maskUtils.frPyObjects(segm, h, w)
            else:
                # rle
                rle = anno['segmentation']
            
            return rle

        rle = anno_to_rle(anno, h, w)
        m = maskUtils.decode(rle)

        return m

    @staticmethod
    def obtain_seg_info(scene):
        col_to_ins_dict = scene['col_to_ins']
        seg_masks, all_obj_ids_2d = [], []
        for view_id, stuff in scene['views'].items():
            cls_ids, binary_masks, colors = zip(*stuff['annos'])
            global_ids = [col_to_ins_dict[x] for x in colors]
            seg_ins_2d = imutils.binary_masks_to_seg(np.stack(binary_masks), np.asarray(global_ids))
            seg_masks.append(seg_ins_2d)
            all_obj_ids_2d.append(global_ids)
        return seg_masks, all_obj_ids_2d
    

    def load_grasps(self, filename):
        """Load transformations and qualities of grasps from a JSON file from the dataset.

        Args:
            filename (str): HDF5 or JSON file name.

        Returns:
            np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
            np.ndarray: List of binary values indicating grasp success in simulation.
        """
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            T = np.array(data["transforms"])
            success = np.array(data["quality_flex_object_in_gripper"])
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            T = np.array(data["grasps/transforms"])
            success = np.array(data["grasps/qualities/flex/object_in_gripper"])
            obj_scale = data["object/scale"][()]
        else:
            raise RuntimeError("Unknown file ending:", filename)
        return T, success, obj_scale


    def create_gripper_marker(self, color=[0, 0, 255], tube_radius=0.001, sections=6):
        """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

        Args:
            color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
            tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
            sections (int, optional): Number of sections of each cylinder. Defaults to 6.

        Returns:
            trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
        """
        cfl = trimesh.creation.cylinder(
            radius=0.002,
            sections=sections,
            segment=[
                [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
                [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
            ],
        )
        cfr = trimesh.creation.cylinder(
            radius=0.002,
            sections=sections,
            segment=[
                [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
                [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
            ],
        )
        cb1 = trimesh.creation.cylinder(
            radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
        )
        cb2 = trimesh.creation.cylinder(
            radius=0.002,
            sections=sections,
            segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
        )

        tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
        tmp.visual.face_colors = color

        return tmp


    # Assume scene_id as input
    def __getitem__(self, index):
        data_root = os.path.join(self.root, self.split, f"{index:06d}")
        rgb_images = sorted(glob.glob(f"{data_root}/image.{index:06d}.rgb.*.png"))
        depth_images = sorted(glob.glob(f"{data_root}/image.{index:06d}.raw_depth.*.exr"))
        seg_images = sorted(glob.glob(f"{data_root}/image.{index:06d}.iseg.*.png"))
        
        annos = self.load_json(f"{data_root}/annos.{index:06d}.coco.json")
        camera_poses = self.load_json(f"{data_root}/cameras.{index:06d}.json")
        objects_init = self.load_json(f"{data_root}/objects.init.{index:06d}.json")
        objects_final = self.load_json(f"{data_root}/objects.{index:06d}.json")

        base_scale = objects_init[-1]['base_scale']

        self.camera_intrinsic = {
            'height': 480,
            'width': 640,
            'fx': 444.44444444 * (base_scale / 10),
            'fy': 444.44444444 * (base_scale / 10),
            'cx': 319.5,
            'cy': 239.5,
        }
        self.world_scale = base_scale

        ins_dict = {}

        """
        Hex ID -> object name / path to the object model -> prompt
               -> instance ID -> alignment between multiple views -> annotation in each view
        """
        for obj_init, obj_final in zip(objects_init[:-1], objects_final):
            
            assert obj_init["color"]["hex"] not in ins_dict.keys()
            model_path = "/".join(obj_init["path"].split("/")[2:4])
            concept_path = f"{self.models_root}/{model_path}/concept.json"
            if obj_init["source"] == "gazebo":
                grasps = None
                scores = None
            else:
                model_id = obj_init["path"].split("/")[-2]
                #grasp_f = glob.glob(f"{self.grasp_root}/*_{model_id}_*.h5")[0]
                #grasps, scores, grasp_scale = self.load_grasps(grasp_f)

            ins_dict[obj_init["color"]["hex"]] = {
                "ins_id": len(ins_dict.keys()) + 1, # ins_id 0 is for table
                "cls_name": obj_init["cls_name"],
                "path": model_path, # This is the path to the model dir, you can load prompt using this
                "concepts": json.load(open(concept_path, "r")) if os.path.exists(concept_path) else "",
                #"grasps": grasps,
                #"grasp_scores": scores,
                "size": obj_final["size"],
                "pose": obj_final["pose"],
                "bbox": obj_final["bbox"],
                "rotation": obj_final["rotation"],
                #"grasp_scale": grasp_scale,
                "object_scale": obj_init["sim_scale"]
            }

        img_name_to_id = {x["file_name"]:x["id"] for x in annos["images"]}

        scene_data = {'views': {}}
        for rgb_f, depth_f, seg_f in zip(rgb_images, depth_images, seg_images):
            assert rgb_f.split(".")[-2] == depth_f.split(".")[-2] == seg_f.split(".")[-2]
            view_id = rgb_f.split(".")[-2]
            image_id = img_name_to_id[rgb_f.split("/")[-1]]
            rgb = self.read_rgb_image(rgb_f)
            depth = self.read_depth_image(depth_f)
            h, w, c = rgb.shape
            
            seg_img = self.read_seg_image(seg_f)
            
            _annos = []
            for x in annos["annotations"]:
                if x["image_id"] == image_id:
                    bin_m = self.anno_to_mask(x, h, w)
                    hex_id = x["seg_color_hex"] 
                    cls_name = ins_dict[hex_id]["cls_name"]
                    _annos.append([cls_name, bin_m, hex_id]) # we dont need cls_id anymore, so just append cls_name here
                    
            scene_data['views'][view_id] = {
                "camera": camera_poses[view_id],
                "annos": _annos,
                "rgb": rgb,
                "depth": depth,
                "ins_seg": seg_img,
                "imgpaths": rgb_f,
            }
        
        scene_data["objects_info"] = {
            0: 'table',
            **{
                v["ins_id"]:{
                    "cls_name": v["cls_name"],
                    #"_cls_name_2": v["concepts"]["cls"],
                    "concepts": v["concepts"]["concepts"],
                    "hex_id": k,
                    "path": v["path"],
                    #"grasps": v["grasps"],
                    #"grasp_scores": v["grasp_scores"],
                    "size": v["size"],
                    "pose": v["pose"],
                    "bbox": v["bbox"],
                    #"grasp_scale": v["grasp_scale"],
                    "rotation": v["rotation"]
                } for k, v in ins_dict.items()}
            }

        scene_data["queries"] = {0: 'table',
            **{v["ins_id"]:v["cls_name"] for v in ins_dict.values()}} # instance ID -> class name
        scene_data["col_to_ins"] = {'#000000': 0,
            **{k:v["ins_id"] for k,v in ins_dict.items()}} # Hex code to instance ID
        scene_data["ins_to_cls"] = {0: self.name_to_id['table'],
            **{v["ins_id"]:self.name_to_id[v["cls_name"]] for v in ins_dict.values()}} # instance ID -> class id
        scene_data["camera_intrinsic"] = self.camera_intrinsic
        return scene_data


