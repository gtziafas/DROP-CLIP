import numpy as np 
import open3d as o3d 
import scipy
import trimesh
from collections import Counter

import utils.image as imutils
import utils.transforms as tutils
#from utils.projections import rgbd_to_pointcloud_o3d
# from utils.viz import *

def to_o3d(points, colors=None, scale=1.0):
    x = o3d.geometry.PointCloud()
    x.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        x.colors = o3d.utility.Vector3dVector(colors)
        x = x.scale(scale, x.get_center())
    return x


def rgbd_to_pointcloud_o3d(rgb, depth, camera_intrinsics, depth_scale=1.0, depth_trunc=25.0):
    intr = o3d.camera.PinholeCameraIntrinsic(
        width=rgb.shape[1], height=rgb.shape[0], 
        fx=camera_intrinsics['fx'],
        fy=camera_intrinsics['fy'],
        cx=camera_intrinsics['cx'],
        cy=camera_intrinsics['cy'],
    )
    rgb_image = o3d.geometry.Image(rgb)
    depth_image = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image, depth_scale, depth_trunc, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intr)
    return pcd

# def plane_removal(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
#     plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
#                                              ransac_n=ransac_n,
#                                              num_iterations=num_iterations)
#     [a, b, c, d] = plane_model
#     #print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
#     inlier_cloud = pcd.select_by_index(inliers)
#     inlier_cloud.paint_uniform_color([1.0, 0, 0])
#     return inlier_cloud

def plane_removal(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)

    [a, b, c, d] = plane_model

    # Select the outlier points (not on the plane)
    outlier_indices = np.setdiff1d(np.arange(len(pcd.points)), inliers)
    outlier_cloud = pcd.select_by_index(outlier_indices)

    return outlier_cloud

def extract_geometries(scene, view=0):
    
    def process(pts, labels, objID, color, use_rect=False):
        mask = np.where(labels == objID)
        pc = pts[mask]
        pc_filt = pc_outlier_removal(pc.copy(), eps = 0.05, min_points=50)
        try:
            box = to_o3d(pc_filt).get_axis_aligned_bounding_box()
            box_r = to_o3d(pc_filt).get_oriented_bounding_box()
            use_pc = pc_filt.copy()
        except:
            #o3d_viewer([to_o3d(pc), to_o3d(pc_filt).translate([0.5, 0, 0])])
            box = to_o3d(pc).get_axis_aligned_bounding_box()
            box_r = to_o3d(pc).get_oriented_bounding_box()
            use_pc = pc.copy()
        mesh = draw_box_outline(box if not use_rect else box_r, color)
        return mask, use_pc, box, box_r, mesh

    # all scenes have same object IDs
    _, scene_objIDs, scene_labels = zip(
        *[(j, x['obj_id'], x['model_name']) for j,x in enumerate(scene['state'])])

    if view > 0:
        # use view-specific point-cloud
        pc_xyz = scene['views'][view]['pc_xyz'].copy()
        pc_label = scene['views'][view]['pc_label'].copy()
    else:
        # use full point-cloud
        pc_xyz = scene['aggr']['pc_xyz'].copy()
        pc_label = scene['aggr']['pc_label'].copy()

    meshes = []
    boxes = []
    boxes_r = []
    pcs = []
    masks = []
    for j, (objID, l) in enumerate(list(zip(scene_objIDs, scene_labels))):
        col = PALLETE_MAP[objID]
        mask, pc, box, box_r, mesh = process(pc_xyz, pc_label, objID, col)
        meshes.append(mesh)
        boxes.append(box)
        boxes_r.append(box_r)
        pcs.append(pc)
        masks.append(mask)
    return masks, pcs, boxes, boxes_r, meshes


def extract_geometries_multiview(scene):
    all_masks, all_pcs, all_boxes, all_rects, all_meshes = [], [], [], [], []
    for view in range(10):
        geoms = extract_geometries(scene, view)
        all_masks.append(geoms[0])
        all_pcs.append(geoms[1])
        all_boxes.append(geoms[2])
        all_rects.append(geoms[3])
        all_meshes.append(geoms[4])
    return all_masks, all_pcs, all_boxes, all_rects, all_meshes


def aggregate_views_blender_new(scene, camera_intrinsic, depth_trunc=25.0, voxel_size=None):
    def _cvt_blender_coord(pts):
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]
        return pts

    col_to_ins_dict = scene['col_to_ins']
    all_points = []
    all_colors = []
    all_labels_ins = []
    all_valid_m = []
    T_cam = np.eye(4)
    T_cam[1,1] = -1
    T_cam[2,2] = -1

    all_pc = None
    for view_id, stuff in scene['views'].items():
        rgb = stuff['rgb']
        depth = stuff['depth'].astype(np.float32)

        valid_m = depth < depth_trunc # Remove background points

        # segmentation mask -> project to 3D for valid points only
        obj_name, binary_masks, colors = zip(*stuff['annos'])
        obj_name, global_ids = zip(*zip(obj_name, np.asarray([col_to_ins_dict[x] for x in colors])))

        #global_ids = [col_to_ins_dict[x]["ins_id"] for x in colors]
        #seg_ins_2d = imutils.binary_masks_to_seg(np.stack(binary_masks), np.asarray(global_ids))
        seg_ins_2d = imutils.binary_masks_to_seg(np.stack(binary_masks), np.asarray([col_to_ins_dict[x] for x in colors]))
        seg_ins = seg_ins_2d[valid_m].flatten()

        # forward project 2D -> 3D
        pc = rgbd_to_pointcloud_o3d(rgb, depth, camera_intrinsic, depth_trunc=depth_trunc)

        pc_cam = pc.transform(T_cam)

        # # points in camera coordinates, revert axes from open3d convension to numpy image convension 
        # pts_cam = _cvt_blender_coord(np.asarray(pc.points))

        # Transform from camera coord to world frame
        world2cam = np.asarray(stuff['camera']['world_matrix']).copy().astype(np.float64)
        pc_world = pc_cam.transform(world2cam)

        if all_pc is None:
            all_pc = pc_world
        else:
            all_pc += pc_world

        # all_points.append(np.asarray(pc_world.points))
        # all_colors.append(np.asarray(pc_world.colors))
        all_labels_ins.append(seg_ins)

    # concatenate points accross views
    # all_points_np = np.concatenate(all_points, axis=0)
    # all_colors_np = np.concatenate(all_colors, axis=0)
    #all_labels_cls_np = np.concatenate(all_labels_cls, axis=0)
    all_labels_ins_np = np.concatenate(all_labels_ins, axis=0)

    if voxel_size is None:
        return np.asarray(all_pc.points), np.asarray(all_pc.colors), all_labels_ins_np 

    else:
        # all_pc_o3d = to_o3d(
        #     all_points_np, all_colors_np)

        # Voxel downsample the aggregated pointclouds, with voxel size 0.02m
        all_pc_down, cube_ids, original_indices = all_pc.voxel_down_sample_and_trace(
            min_bound=all_pc.get_min_bound(), max_bound=all_pc.get_max_bound(), voxel_size=voxel_size)

        all_points_down = np.asarray(all_pc_down.points)
        all_colors_down = np.asarray(all_pc_down.colors)

        # find indices in original to get downsampled labels
        #all_labels_ins_down, all_labels_cls_down = [], []
        all_labels_ins_down = []
        for old_indices in original_indices:
            #old_labels_cls = all_labels_cls_np[old_indices]
            old_labels_ins = all_labels_ins_np[old_indices]
            #all_labels_cls_down.append(Counter(old_labels_cls).most_common()[0][0])
            all_labels_ins_down.append(Counter(old_labels_ins).most_common()[0][0])
        #all_labels_cls_down = np.stack(all_labels_cls_down)  
        all_labels_ins_down = np.stack(all_labels_ins_down)  


        return all_points_down, all_colors_down, all_labels_ins_down

def aggregate_views_regrad(scene):
    pc_full = None
    for v in list(scene.keys()):
        if pc_full is None:
            pc_full = {'xyz': scene[v]['pc_xyz'], 'rgb': scene[v]['pc_rgb'], 'label': scene[v]['pc_label'], 'anno':scene[v]['pc_anno']}
            continue
        pc_full['xyz'] = np.concatenate([pc_full['xyz'], scene[v]['pc_xyz']], axis=0)
        pc_full['rgb'] = np.concatenate([pc_full['rgb'], scene[v]['pc_rgb']], axis=0)
        pc_full['label'] = np.concatenate([pc_full['label'], scene[v]['pc_label']], axis=0)
        pc_full['anno'] = np.concatenate([pc_full['anno'], scene[v]['pc_anno']], axis=0)
    return pc_full


def aggregate_views_blender(scene, camera_intrinsic, depth_trunc=25.0, voxel_size=None):
    def _cvt_blender_coord(pts):
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]
        return pts

    col_to_ins_dict = scene['col_to_ins']
    all_points = []
    all_colors = []
    all_labels_ins = []
    all_valid_m = []

    for view_id, stuff in scene['views'].items():
        rgb = stuff['rgb']
        depth = stuff['depth'].astype(np.float32)

        valid_m = depth < depth_trunc # Remove background points

        # segmentation mask -> project to 3D for valid points only
        obj_name, binary_masks, colors = zip(*stuff['annos'])
        obj_name, global_ids = zip(*zip(obj_name, np.asarray([col_to_ins_dict[x] for x in colors])))

        #global_ids = [col_to_ins_dict[x]["ins_id"] for x in colors]
        #seg_ins_2d = imutils.binary_masks_to_seg(np.stack(binary_masks), np.asarray(global_ids))
        seg_ins_2d = imutils.binary_masks_to_seg(np.stack(binary_masks), np.asarray([col_to_ins_dict[x] for x in colors]))
        seg_ins = seg_ins_2d[valid_m].flatten()

        # forward project 2D -> 3D
        pc = rgbd_to_pointcloud_o3d(rgb, depth, camera_intrinsic, depth_trunc=depth_trunc)

        # points in camera coordinates, revert axes from open3d convension to numpy image convension 
        pts_cam = _cvt_blender_coord(np.asarray(pc.points))

        # Transform from camera coord to world frame
        world2cam = np.asarray(stuff['camera']['world_matrix']).copy().astype(np.float64)
        pts_w = tutils.transform_pointcloud_to_world_frame(pts_cam, world2cam)

        all_points.append(np.asarray(pts_w))
        all_colors.append(np.asarray(pc.colors))
        all_labels_ins.append(seg_ins)

    # concatenate points accross views
    all_points_np = np.concatenate(all_points, axis=0)
    all_colors_np = np.concatenate(all_colors, axis=0)
    #all_labels_cls_np = np.concatenate(all_labels_cls, axis=0)
    all_labels_ins_np = np.concatenate(all_labels_ins, axis=0)

    if voxel_size is None:
        return all_points_np, all_colors_np, all_labels_ins_np 

    else:
        all_pc_o3d = to_o3d(
            all_points_np, all_colors_np)

        # Voxel downsample the aggregated pointclouds, with voxel size 0.02m
        all_pc_down, cube_ids, original_indices = all_pc_o3d.voxel_down_sample_and_trace(
            min_bound=all_pc_o3d.get_min_bound(), max_bound=all_pc_o3d.get_max_bound(), voxel_size=voxel_size)

        all_points_down = np.asarray(all_pc_down.points)
        all_colors_down = np.asarray(all_pc_down.colors)

        # find indices in original to get downsampled labels
        #all_labels_ins_down, all_labels_cls_down = [], []
        all_labels_ins_down = []
        for old_indices in original_indices:
            #old_labels_cls = all_labels_cls_np[old_indices]
            old_labels_ins = all_labels_ins_np[old_indices]
            #all_labels_cls_down.append(Counter(old_labels_cls).most_common()[0][0])
            all_labels_ins_down.append(Counter(old_labels_ins).most_common()[0][0])
        #all_labels_cls_down = np.stack(all_labels_cls_down)  
        all_labels_ins_down = np.stack(all_labels_ins_down)  

        return all_points_down, all_colors_down, all_labels_ins_down


def remove_table_mask(points, colors, labels, table_id=0):
    no_table_mask = labels !=0
    points = points[no_table_mask]
    colors = colors[no_table_mask]
    labels = labels[no_table_mask]
    return points, colors, labels


def remove_invisible_points(points, colors, labels, visibility_mask):
    _visible_mask = visibility_mask.sum(0) > 0
    points = points[_visible_mask]
    colors = colors[_visible_mask]
    labels = labels[_visible_mask]
    visibility_mask = visibility_mask[:, _visible_mask]
    return points, colors, labels, visibility_mask


def find_existing_points(pc_aggr, pc1):
    # Convert numpy arrays to open3d point clouds
    pcd_aggr = o3d.geometry.PointCloud()
    pcd_aggr.points = o3d.utility.Vector3dVector(pc_aggr)
    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(pc1)

    # Build a KDTree for the aggregated point cloud
    kdtree = o3d.geometry.KDTreeFlann(pcd_aggr)

    # Search for each point in pc1 within the aggregated point cloud
    mask = np.zeros(len(pc_aggr), dtype=int)
    for point in pc1:
        [_, idx, _] = kdtree.search_knn_vector_3d(point, 1)
        mask[idx] = 1

    return mask


def find_indices(pc1, pc2):
  """
  Finds the indices in pc1 that exist in pc2.

  Args:
      pc1: A numpy array of shape (M, 3) representing the first pointcloud.
      pc2: A numpy array of shape (K, 3) representing the second pointcloud,
          which is a subset of the first pointcloud.

  Returns:
      A numpy array of shape (L,) containing the indices in pc1 that exist in pc2.
  """

  # Efficiently compute the boolean intersection using broadcasting
  intersection = np.all(pc1[:, None] == pc2[None, :], axis=2)

  # Flatten the boolean intersection array and return the nonzero indices
  return np.flatnonzero(intersection)


def pc_voxel_down(pc, voxel_size=0.0075):
    pc = to_o3d(pc).voxel_down_sample(voxel_size=voxel_size)
    return np.array(pc.points)
    

def remove_stat_outlier(points, n_pts=25, ratio=2.0):
    p = to_o3d(points)
    cl, ind = p.remove_statistical_outlier(nb_neighbors=n_pts, std_ratio=ratio)
    p1 = p.select_by_index(ind)
    return np.array(p1.points), ind


def pc_outlier_removal(pc, eps = 0.05, min_points = 15, voxel_size=0.02):
    # Set the parameters for DBSCAN
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_points: The number of samples in a neighborhood for a point to be considered as a core point.
    pcd = to_o3d(pc)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Apply DBSCAN clustering
    #with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    _, ind = voxel_down_pcd.remove_radius_outlier(nb_points=min_points, radius=eps)
    
    # Filter out outlier points
    #max_label = labels.max()
    return ind
    # inlier_points = points[labels >= 0]
    # #outlier_points = points[labels < 0]

    # return inlier_points


def pc_points_within_sphere(pc, sphere_radius, origin=[0, 0, 0.5]):
    O = np.array([origin]).repeat(pc.shape[0], 0)
    dist = np.linalg.norm((pc - O), axis=-1)
    mask = dist < sphere_radius
    return mask


def find_closest_indices(full_pc, filtered_pc, eps=None):
    # Create a KDTree for the full point cloud
    kdtree = scipy.spatial.cKDTree(full_pc)
    
    # Query the KDTree for the closest point in the full point cloud for each point in the filtered point cloud
    distances, indices = kdtree.query(filtered_pc)
    
    # Return only indices within threshold distance if desired
    if eps is not None:
        indices = indices[np.argwhere(distances <= eps)]
        
    return indices


def find_closest_point_z_axis_chunked(pcA, pcB, chunk_size=1000):
    # Initialize the output array with the same shape as pcB
    pcB_new = np.copy(pcB)
    # Determine the number of chunks needed
    num_chunks = int(np.ceil(pcB.shape[0] / chunk_size))

    for i in range(num_chunks):
        # Calculate the start and end indices for the current chunk
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, pcB.shape[0])
        # Extract the current chunk from pcB
        pcB_chunk = pcB[start_idx:end_idx]

        # Compute the cost matrix for the current chunk
        cost_matrix_chunk = np.abs(pcA[:, np.newaxis, 0] - pcB_chunk[:, 0]) + np.abs(pcA[:, np.newaxis, 1] - pcB_chunk[:, 1])
        # Find the index of the minimum cost for each point in the chunk
        min_cost_indices_chunk = np.argmin(cost_matrix_chunk, axis=0)
        # Assign the z values from pcA to the current chunk based on the minimum cost indices
        pcB_new[start_idx:end_idx, 2] = pcA[min_cost_indices_chunk, 2]

    return pcB_new


def trimesh_to_o3d(tri_mesh):
    # Extract vertices and faces from trimesh mesh
    vertices = np.array(tri_mesh.vertices)
    faces = np.array(tri_mesh.faces)

    # Create Open3D mesh
    open3d_mesh = o3d.geometry.TriangleMesh()

    # Set vertices and faces
    open3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    open3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    return open3d_mesh