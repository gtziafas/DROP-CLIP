from models.distil import DisNet
from models.distil.disnet import state_dict_remove_moudle
from data.dataset import build_dataset
import utils.config as config
import torch
from tqdm import tqdm
from utils.projections import apply_pca
from utils.viz import *
import os
import MinkowskiEngine as ME

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def make_viz(weights_path, config_path, save_folder, objects_path_train, objects_path_val=None):
  save_path  = os.path.join(save_folder, weights_path.split('/')[-2])
  os.makedirs(save_path, exist_ok=True)

  cfg = config.load_cfg_from_cfg_file(config_path) 
  cfg['objects_train_path'] = objects_path_train
  if objects_path_val is not None:
  	cfg['objects_path_val'] = objects_path_val

  model = DisNet(cfg).to(DEVICE).eval()
  chp = torch.load(weights_path)
  weights = state_dict_remove_moudle(chp['state_dict'])
  model.load_state_dict(weights)

  use_color = cfg['use_color']
  cfg['use_color'] = True
  train_data, _, _ = build_dataset(cfg)
  loader = torch.utils.data.DataLoader(
    train_data, shuffle=False, batch_size=1, collate_fn=train_data.collate_fn)
  viz = []
  for data in tqdm(loader):
    sinput = ME.SparseTensor(
                coordinates=data["coords"],
                features=data["input_features"] if use_color else data['input_features'][:,:3],
                device="cuda"
            ).float()
    batch_shapes = [x.shape[0] for x in sinput.decomposed_features]

    with torch.no_grad():
      if cfg.use_cls_head:
        out, out_cls = model(sinput)
      else:
        out = model(sinput)

    out /= out.norm(dim=-1, keepdim=True)

    xyz = data['input_features'][:,:3].numpy()
    rgb = data['input_features'][:, 3:].numpy()
    targets = data['output_features'].numpy()
    label = data['labels'].numpy()
    anno = np.array([PALLETE_MAP[x] for x in label])

    p_o3d = to_o3d(xyz, rgb)
    y_o3d = to_o3d(xyz, anno).translate([.75, 0, 0])
    targets_o3d = to_o3d(xyz, apply_pca(targets, seed=0)).translate([0, -.75, 0])
    feat_o3d = to_o3d(xyz, apply_pca(out.cpu().numpy(), seed=0)).translate([.75, -.75, 0])
    merged = p_o3d + y_o3d + targets_o3d + feat_o3d

    ID = data['scene_ids'][0]
    o3d.io.write_point_cloud(os.path.join(save_path, f'merged_{ID}.pcd'), merged)



if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-w', '--weights_path', help='path to pretrained model weights', type=str, default=None)
  parser.add_argument('-c', '--config_path', help='path to configuration', type=str, default=None)
  parser.add_argument('-s', '--save_folder', help='path to save results', type=str, default=None)
  parser.add_argument('-to', '--objects_path_train', help='path to objects json (train)', type=str, default=None)
  parser.add_argument('-vo', '--objects_path_val', help='path to objects json (val)', type=str, default=None)
  kwargs = vars(parser.parse_args())
  make_viz(**kwargs)