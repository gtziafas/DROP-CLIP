from gripper_models.franka_panda.make import make_franka_mesh
from utils.geometry import trimesh_to_o3d
import os
import numpy as np
import trimesh
import open3d as o3d


def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
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

    tmp = trimesh_to_o3d(tmp)
    tmp.paint_uniform_color(color)

    # implicit_transform = np.array([
    #     [0, 0, 1, -0.06],
    #     [0, 1, 0, 0,],
    #     [-1, 0, 0, -0.01],
    #     [0, 0, 0, 1]
    # ])
    implicit_transform = np.array([
       [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00, -0.06],
       [ 1.00000000e+00,  3.33066907e-16,  0.00000000e+00, -0.01],
       [-3.33066907e-16,  1.00000000e+00,  5.55111512e-17, -0.01],
       [0, 0, 0, 1],
    ])
    tmp.transform(implicit_transform)
    
    return tmp


def make(gripper_type):
    if gripper_type == "franka_panda":
        mesh = trimesh_to_o3d(make_franka_mesh(
            os.path.join(os.getcwd(), 'gripper_models/franka_panda/meshes')))
        mesh.paint_uniform_color([0.4, 0.4, 0.4])
        #mesh.scale(0.25, mesh.get_center())
        theta = np.pi / 2
        R = np.array([
            [np.cos(theta), 0, np.sin(theta), 0.025],
            [0, 1, 0, -0.01],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1],
        ])
        mesh.transform(R)
        mesh.scale(1.25, center=mesh.get_center())
        return mesh

    elif gripper_type == 'robotiq_2f_140':
        mesh = trimesh.load(os.path.join('gripper_models/robotiq_2f_140/robotiq_arg2f_140.obj'))
        mesh = trimesh_to_o3d(mesh).paint_uniform_color([0,1,1])
        #hmodel_o3d = hmodel_o3d.scale(1, center=hmodel_o3d.get_center())
        theta = np.pi / 2
        R = np.array([
                    [np.cos(theta), 0, np.sin(theta), -0.0],
                    [0, 1, 0, 0.0],
                    [-np.sin(theta), 0, np.cos(theta), 0],
                    [0, 0, 0, 1],
        ])
        mesh.transform(R)
        #mesh.scale(1.25, center=mesh.get_center())
        return mesh

    elif gripper_type == "marker":
        return create_gripper_marker(color=[0.4, 0.4, 0.4])

    else:
        raise ValueError(f"Unknown gripper type {gripper_type}. Check gripper_models/.")
