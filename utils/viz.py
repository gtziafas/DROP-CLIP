import copy
import cv2
import numpy as np
import open3d as o3d
#from open3d.web_visualizer import draw
from matplotlib import pyplot as plt 

import warnings
warnings.filterwarnings("ignore")

# import open3d.visualization as vis 
import utils.projections as proj

def to_o3d(points, colors=None, normals=None, scale=1.0):
    x = o3d.geometry.PointCloud()
    x.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        x.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        x.normals = o3d.utility.Vector3dVector(normals)
    x = x.scale(scale, x.get_center())
    return x


PALLETE = np.array([
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
       [2.00000000e-01, 2.00000000e-01, 2.00000000e-01],
       [0.01897779, 0.98651468, 0.09738904],
       [0.53350548, 0.26298251, 0.9601228 ],
       [0.19629425, 0.42507458, 0.33849926],
       [0.20808288, 0.40207791, 0.51957192],
       [0.80200273, 0.27820731, 0.73292514],
       [0.96839393, 0.98492235, 0.20483592],
       [0.54878112, 0.86824018, 0.26262122],
       [0.61845413, 0.95864834, 0.56319795],
       [0.2024566 , 0.54534184, 0.25383653],
       [0.90641528, 0.26322816, 0.39312142],
       [0.28979289, 0.65575112, 0.5045464 ],
       [0.44887392, 0.42652772, 0.1449758 ],
       [0.07295724, 0.82962005, 0.61854373],
       [0.75599124, 0.18524666, 0.55930736],
       [0.52426257, 0.71267499, 0.69488294],
       [0.78031708, 0.57022642, 0.07769594],
       [0.15858803, 0.68825184, 0.96999511],
       [0.92109541, 0.52293084, 0.24276273],
       [0.66077861, 0.53030237, 0.59786725],
       [0.24413826, 0.76645894, 0.62987864],
       [0.71211083, 0.65554378, 0.7463561 ],
       [0.33377819, 0.08047042, 0.89425172],
       [0.71583059, 0.17852382, 0.39639506],
       [0.53575894, 0.08250583, 0.05178909],
       [0.47964034, 0.77436185, 0.54568445],
       [0.19251496, 0.66641699, 0.36067149],
       [0.44014663, 0.68304682, 0.26989905],
       [0.32318882, 0.31751388, 0.1886325 ],
       [0.10380822, 0.89578403, 0.18021793],
       [0.07960468, 0.44477548, 0.13727231],
       [0.34982554, 0.43836722, 0.40700478],
       [0.29413655, 0.83238094, 0.33033832],
       [0.27477381, 0.60419043, 0.04834325],
       [0.6074918 , 0.78612273, 0.99223941],
       [0.84326872, 0.7806221 , 0.20317598],
       [0.08692968, 0.80206732, 0.49343417],
       [0.06073446, 0.50080976, 0.8624674 ],
       [0.86462371, 0.19041732, 0.49227738],
       [0.95879871, 0.65516213, 0.45896927],
       [0.18222867, 0.69365152, 0.40732584],
       [0.3706478 , 0.68809223, 0.20147523],
       [0.12007848, 0.62680726, 0.83760729],
       [0.84301866, 0.57219948, 0.77179068],
       [0.32131165, 0.62410709, 0.33199207],
       [0.16693778, 0.82203463, 0.45640388],
       [0.72880466, 0.6721055 , 0.58970972],
       [0.10486954, 0.764784  , 0.63506786],
       [0.39743283, 0.99861897, 0.49961081],
       [0.01509293, 0.74206056, 0.29371792],
       [0.52107921, 0.86717633, 0.47218746],
       [0.79299878, 0.66156159, 0.209883  ],
       [0.28132676, 0.88621188, 0.579432  ],
       [0.79455495, 0.73367278, 0.61003907],
       [0.16906779, 0.73690148, 0.33655934],
       [0.55864982, 0.20729471, 0.04360255],
       [0.2509452 , 0.3674491 , 0.48853552],
       [0.49301502, 0.71459002, 0.31824289],
       [0.07556905, 0.93274676, 0.65736986],
       [0.03345206, 0.75680554, 0.04240944],
       [0.5296778 , 0.60520836, 0.68701369],
       [0.74818271, 0.46970833, 0.7064058 ],
       [0.68616522, 0.78705385, 0.00990659],
       [0.35315021, 0.38743765, 0.28922997],
       [0.06342607, 0.62594179, 0.94697244],
       [0.50758232, 0.69024067, 0.90851563],
       [0.45782603, 0.2188143 , 0.22640122],
       [0.39510758, 0.52048675, 0.72660804],
       [0.72515075, 0.48418169, 0.07370792],
       [0.49814651, 0.81696686, 0.55947755],
       [0.60620536, 0.91538162, 0.64161796],
       [0.10312688, 0.39118986, 0.2430239 ],
       [0.3427938 , 0.21052051, 0.22790802],
       [0.61704971, 0.43798848, 0.65307264],
       [0.67102583, 0.293662  , 0.24685417],
       [0.86352195, 0.81947565, 0.58959605],
       [0.31906259, 0.00420609, 0.95583643],
       [0.18473381, 0.64127581, 0.41436518],
       [0.11775362, 0.04534339, 0.9865174 ],
       [0.37973611, 0.43703848, 0.29703236],
       [0.81730658, 0.17382415, 0.30462944],
       [0.36768952, 0.5912103 , 0.24113112],
       [0.38095538, 0.19548013, 0.70761849],
       [0.67057994, 0.03154924, 0.8872428 ],
       [0.73027139, 0.79357448, 0.12683997],
       [0.68381953, 0.65772157, 0.62618613],
       [0.76095202, 0.3357927 , 0.61325443],
       [0.19899168, 0.17065659, 0.78927479],
       [0.99768433, 0.70499174, 0.38338826],
       [0.58001591, 0.38110364, 0.69910485],
       [0.05648454, 0.9707813 , 0.17392128],
       [0.1688623 , 0.07165697, 0.69795971],
       [0.35525494, 0.300134  , 0.21976273],
       [0.95765106, 0.77111028, 0.81136534],
       [0.84182072, 0.75058315, 0.40714976],
       [0.60386387, 0.74140982, 0.02947334],
       [0.19194669, 0.92611954, 0.30199611],
       [0.74716251, 0.09640781, 0.53206117],
       [0.67115968, 0.69608084, 0.46364774],
       [0.82537483, 0.77356361, 0.63001829],
       [0.79076305, 0.07342261, 0.73693608],
       [0.71418549, 0.0964451 , 0.30476959],
       [0.413797  , 0.53790549, 0.15998345],
       [0.90306431, 0.98092169, 0.82453767],
       [0.25424381, 0.33220809, 0.88719143],
       [0.71998246, 0.1523081 , 0.85273838],
       [0.27416891, 0.39760219, 0.90303531],
       [0.66783757, 0.39725031, 0.13804654],
       [0.40524154, 0.68554037, 0.41753812],
       [0.33888254, 0.18787745, 0.16188585],
       [0.25162886, 0.80331348, 0.93130941],
       [0.80595322, 0.36018284, 0.66628325],
       [0.74535449, 0.69122971, 0.21893285],
       [0.5887644 , 0.11256121, 0.09279174],
       [0.31885085, 0.86032074, 0.91098728],
       [0.59564168, 0.37228176, 0.11403619],
       [0.69621717, 0.78787949, 0.85134105],
       [0.87679519, 0.62246997, 0.52184652],
       [0.76435071, 0.45934018, 0.81877989],
       [0.49826857, 0.22424163, 0.22362034],
       [0.28662723, 0.57384067, 0.26100101],
       [0.30349831, 0.90984569, 0.03816108],
       [0.59491843, 0.50995824, 0.25161139],
       [0.88888451, 0.66885094, 0.77709325],
       [0.60989787, 0.38828888, 0.30283286],
       [0.98885761, 0.49941407, 0.94952423],
       [0.23800933, 0.48459482, 0.7158806 ],
       [0.6097016 , 0.83357845, 0.29432821],
       [0.53830776, 0.7226755 , 0.27922068],
       [0.4995509 , 0.96097217, 0.47044494],
       [0.50987013, 0.34922725, 0.81196511],
       [0.89362475, 0.72578684, 0.43739454],
       [0.93782081, 0.01567235, 0.62982872],
       [0.50359739, 0.90912144, 0.24054516],
       [0.03582125, 0.76695665, 0.00565738],
       [0.05544791, 0.64495607, 0.12329997],
       [0.10180629, 0.49369839, 0.21091824],
       [0.74091283, 0.95451308, 0.01133794],
       [0.63761547, 0.40690346, 0.71552829],
       [0.1784781 , 0.22525661, 0.74085523],
       [0.89828679, 0.18986398, 0.06730461],
       [0.64269476, 0.64237067, 0.73705147],
       [0.10842249, 0.79340354, 0.58848365],
       [0.35640193, 0.66248687, 0.51883799],
       [0.06048887, 0.2537142 , 0.55316955],
       [0.51274655, 0.37715803, 0.67106144],
       [0.62459888, 0.92989316, 0.15830297],
       [0.90222644, 0.14487514, 0.35566004],
       [0.32455355, 0.61950812, 0.24205932],
       [0.920977  , 0.68900544, 0.92725541],
       [0.61513772, 0.71236849, 0.77770535],
       [0.78288938, 0.7214679 , 0.60561866],
       [0.77520793, 0.49420481, 0.11749744],
       [0.34041787, 0.75730626, 0.53620231],
       [0.96763123, 0.97159507, 0.82259007],
       [0.63821375, 0.60973283, 0.38200946],
       [0.63500038, 0.81846476, 0.07326434],
       [0.48726377, 0.76652505, 0.309746  ],
       [0.50073003, 0.32110164, 0.12670182],
       [0.21947344, 0.97504847, 0.88148391],
       [0.75411004, 0.73788059, 0.9300617 ],
       [0.64892323, 0.12958308, 0.17173424],
       [0.33859882, 0.66854446, 0.83148623],
       [0.28306324, 0.91257903, 0.36770621],
       [0.38422941, 0.90721211, 0.6117125 ],
       [0.68753974, 0.57963546, 0.50169174],
       [0.90073682, 0.70115823, 0.80204007],
       [0.24334086, 0.38113214, 0.10605162],
       [0.43533171, 0.03636817, 0.40066196],
       [0.74131593, 0.8634927 , 0.36563889],
       [0.26324129, 0.80084322, 0.85287621],
       [0.68522762, 0.79848609, 0.77671857],
       [0.93270366, 0.8782484 , 0.94837765],
       [0.07055892, 0.83796397, 0.46763672],
       [0.11223048, 0.4605326 , 0.98612588],
       [0.27303349, 0.72707931, 0.45840285],
       [0.27154618, 0.67664331, 0.44158592],
       [0.20344489, 0.84040102, 0.33927371],
       [0.24553726, 0.49657569, 0.31539379],
       [0.5908409 , 0.73838832, 0.72264223],
       [0.40297701, 0.96952396, 0.03861845],
       [0.58415522, 0.02012987, 0.31324461],
       [0.98757785, 0.29713529, 0.84540123],
       [0.56711905, 0.10616481, 0.29739332],
       [0.24602497, 0.32027211, 0.40914252],
       [0.90230573, 0.28850241, 0.51103226],
       [0.63911728, 0.57455848, 0.24086959],
       [0.66216523, 0.62070217, 0.07180089],
       [0.79721564, 0.74393547, 0.85058883],
       [0.67374348, 0.94244589, 0.53290498],
       [0.41825615, 0.64668048, 0.24158439],
       [0.49633575, 0.81407279, 0.23582085],
       [0.8789555 , 0.6617018 , 0.83041474],
       [0.15878479, 0.89735845, 0.91060048],
       [0.34501964, 0.59579019, 0.19740806],
       [0.11571409, 0.51695672, 0.97098331],
       [0.46178103, 0.95526583, 0.24596372],
       [0.53813404, 0.45610694, 0.02723156],
       [0.5142231 , 0.46034   , 0.40489279],
       [0.38357346, 0.25762709, 0.71684863],
       [0.21039303, 0.82754548, 0.06410292],
       [0.42097683, 0.92740275, 0.73058553],
       [0.11500046, 0.48958619, 0.31650088],
       [0.60010417, 0.37413738, 0.31333421],
       [0.77486003, 0.9368106 , 0.20412423],
       [0.1720781 , 0.71561884, 0.77534505],
       [0.45879478, 0.42959003, 0.93105535],
       [0.91944989, 0.77614003, 0.91192755],
       [0.01332786, 0.40612316, 0.86812029],
       [0.58574283, 0.16976191, 0.10905419],
       [0.13563667, 0.52690934, 0.09120597],
       [0.89861291, 0.59954104, 0.5554541 ],
       [0.68648597, 0.067693  , 0.35205207],
       [0.83494605, 0.18604865, 0.83275862],
       [0.47554611, 0.51598902, 0.95479542],
       [0.39633329, 0.08483671, 0.09488584],
       [0.87391154, 0.36426087, 0.10561177],
       [0.83490431, 0.82598691, 0.6496172 ],
       [0.71343405, 0.44757882, 0.16187324],
       [0.40518309, 0.84305807, 0.91961792],
       [0.74341041, 0.66124442, 0.29088587],
       [0.56903434, 0.07495755, 0.31862572],
       [0.24697999, 0.77821786, 0.26554721],
       [0.31467418, 0.66636769, 0.59011032],
       [0.98472333, 0.52170658, 0.68437164],
       [0.59916969, 0.70709577, 0.80970832],
       [0.68798298, 0.96152288, 0.27929926],
       [0.22440052, 0.41028278, 0.29486569],
       [0.61439375, 0.98208401, 0.70340195],
       [0.73669811, 0.09199729, 0.81714481],
       [0.46781363, 0.96445737, 0.95451124],
       [0.03827495, 0.14198423, 0.7899191 ],
       [0.63787367, 0.64378012, 0.72859644],
       [0.73144602, 0.37827975, 0.0076983 ],
       [0.89280787, 0.37567794, 0.60638177],
       [0.70339329, 0.56034741, 0.40948641],
       [0.87674433, 0.4156258 , 0.28193906],
       [0.47185615, 0.60075009, 0.32590958],
       [0.65344388, 0.38247619, 0.52796854],
       [0.73109247, 0.94747675, 0.60895995],
       [0.83951467, 0.35926619, 0.99502442],
       [0.75248311, 0.31795407, 0.726266  ],
       [0.55914812, 0.63903334, 0.22705865],
       [0.14737871, 0.700341  , 0.12516364],
       [0.79805915, 0.98680182, 0.40608256],
       [0.86475913, 0.31643919, 0.67376452],
       [0.9674887 , 0.30335394, 0.42949699],
       [0.53348043, 0.84103881, 0.99983759],
       [0.54190039, 0.58787597, 0.66569822],
       [0.66276931, 0.85429886, 0.15496808],
       [0.75233854, 0.36603166, 0.94691669],
       [0.32705742, 0.82996836, 0.86363844],
       [0.5197666 , 0.11363674, 0.33062258],
       [0.20668445, 0.54684647, 0.91667351],
       [0.67990305, 0.09775537, 0.39131723],
       [0.45051288, 0.80328985, 0.6536629 ]])


PALLETE_MAP = {k-1:v for k,v in enumerate(list(PALLETE))}


def o3d_viewer(geometries, title="display", world_frame=False, background=[0.75, 0.75, 0.75]):
    if world_frame:
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[-0.0,-0.0,-0.0]))
    
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name=title)
    for geometry in geometries:
        viewer.add_geometry(copy.deepcopy(geometry))
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = False
    opt.background_color = np.asarray(background)
    viewer.run()
    viewer.destroy_window()

# def o3d_viewer(geometries, title="display", world_frame=False, jupyter=False):
#     if world_frame:
#         geometries.append(
#             o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[-0.0,-0.0,-0.0]))
#     if jupyter:
#         draw(geometries, title=title, bg_color=np.asarray([0.75, 0.75, 0.75, 1]))
#     else:
#         viewer = o3d.visualization.Visualizer()
#         viewer.create_window(window_name=title)
#         for geometry in geometries:
#             viewer.add_geometry(copy.deepcopy(geometry))
#         opt = viewer.get_render_option()
#         opt.show_coordinate_frame = False
#         opt.background_color = np.asarray([0.75, 0.75, 0.75])
#         viewer.run()
#         viewer.destroy_window()

def get_wireframe(points, color):  # Added line_width parameter
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 6],
        [1, 7],
        [2, 5],
        [2, 7],
        [3, 5],
        [3, 6],
        [4, 5],
        [4, 6],
        [4, 7]
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # Set the width of the lines
    return line_set


def draw_box_outline(aabbox, color, scale_ranges=(90, 100), rescale=1.0):
    center = aabbox.get_center()
    mesh = get_wireframe(np.array(aabbox.get_box_points()), color)
    for ss in range(*scale_ranges):
        _scale = ss * 0.01
        aabbox_s = copy.deepcopy(aabbox).scale(_scale, center)
        _mesh = get_wireframe(np.array(aabbox_s.get_box_points()), color)
        mesh += _mesh
    mesh = mesh.scale(rescale, mesh.get_center())
    return mesh


def draw_box_full(aabbox, color, alpha=0.5):
    # Create a mesh box (which is by default filled) from the aabbox
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=aabbox.get_extent()[0],
                                                    height=aabbox.get_extent()[1],
                                                    depth=aabbox.get_extent()[2])

    # Translate the mesh box to the center of the bounding box
    mesh_box.translate(aabbox.get_center())

    # mesh_box.compute_triangle_normals()

    # Color the mesh box
    mesh_box.paint_uniform_color(color)  # Red color

    return mesh_box


def paint_meshes_rel(meshes, source, targets):
    source_mesh = meshes[source]
    target_meshes = [meshes[x] for x in targets]
    s = copy.deepcopy(source_mesh).paint_uniform_color([0, 1, 0])
    ts = [copy.deepcopy(t).paint_uniform_color([1, 0, 0]) for t in target_meshes]
    return [s] + ts


def paint_image_rel(image, boxes, source, targets):
    ii = image.copy()
    bbox = boxes[source]
    ii = cv2.rectangle(ii, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
    for t in targets:
        bbox = boxes[t]
        ii = cv2.rectangle(ii, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0, 0), 2)
    return ii


def imshow(im):
    cv2.imshow('display1', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    while True:
        key = cv2.waitKey(1) & 0xff
        if ord('q') == key:
            break
    cv2.destroyAllWindows()


def pcshow(xyz, col, more_meshes=[]):
    pc_o3d = to_o3d(xyz, col)
    return o3d_viewer([pc_o3d] + more_meshes)


def get_coord_frame(origin=[-0.05, -0.05, -0.05], transform=None, scale=1.0):
    if transform is None:
        transform = np.eye(4)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=origin); 
    frame = frame.scale(scale, frame.get_center());
    frame = frame.transform(transform)
    return frame


def draw_2d_grasps_in_image(img , grasp_rectangles):
    tmp = img.copy()
    for entry in grasp_rectangles:
            ptA, ptB, ptC, ptD = [list(map(int, pt.tolist())) for pt in entry]
            tmp = cv2.line(tmp, ptA, ptB, (0,0,0xff), 2)
            tmp = cv2.line(tmp, ptD, ptC, (0,0,0xff), 2)
            tmp = cv2.line(tmp, ptB, ptC, (0xff,0,0), 2)
            tmp = cv2.line(tmp, ptA, ptD, (0xff,0,0), 2)
    return tmp


def visualize_4dof_grasps(
    rgb,
    depth,
    title,
    mask,
    save_path):
    s = self.__getitem__(n)

    rgb = s['img']
    depth = (0xff * s['depth'] / 3).astype(np.uint8)
    ii = self.get_annotated_image(n, text=False)
    sentence = s['sentence']
    msk = s['mask']
    msk_img = (rgb * 0.3).astype(np.uint8).copy()
    msk_img[msk, 0] = 255

    fig = plt.figure(figsize=(25, 10))

    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(rgb)
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 4, 2)
    ax.imshow(depth, cmap='gray')
    ax.set_title('Depth')
    ax.axis('off')

    ax = fig.add_subplot(2, 4, 3)
    ax.imshow(msk_img)
    ax.set_title('Segm Mask')
    ax.axis('off')

    ax = fig.add_subplot(2, 4, 4)
    ax.imshow(ii)
    ax.set_title('Box & Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 4, 5)
    plot = ax.imshow(s['grasp_masks']['pos'], cmap='jet', vmin=0, vmax=1)
    ax.set_title('Position')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 4, 6)
    plot = ax.imshow(s['grasp_masks']['qua'], cmap='jet', vmin=0, vmax=1)
    ax.set_title('Quality')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 4, 7)
    plot = ax.imshow(s['grasp_masks']['ang'], cmap='rainbow', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 4, 8)
    plot = ax.imshow(s['grasp_masks']['wid'], cmap='jet', vmin=0, vmax=1)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)

    plt.suptitle(f"{sentence}", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"sample_{n}.png"))


def viz_multiview_clip_sim(images, sims, text_query, threshold=0.9):
    plt.figure()
    cmap = plt.get_cmap("turbo")
    for idx, (image, sim) in enumerate(zip(images, sims)):
        sim_norm = (sim - sim.min()) / (sim.max() - sim.min())
        
        plt.subplot(2, len(images), idx + 1)
        if threshold is not None:
            # paint red
            sim_thr = (sim_norm > threshold).cpu().numpy()
            image = np.array(image)
            image[sim_thr==True, :] = np.array([255, 0, 0], dtype=np.uint8)
            image = Image.fromarray(image)

        plt.imshow(image)
        
        plt.title(os.path.basename(f'view_{idx}'))
        plt.axis("off")

        plt.subplot(2, len(images), len(images) + idx + 1)
        heatmap = cmap(sim_norm.cpu().numpy())
        plt.imshow(heatmap)
        plt.axis("off")

    plt.tight_layout()
    plt.suptitle(f'Similarity to language query "{text_query}"')

    return plt.show()


def viz_multiview_clip_sim_obj_prior(images, segms, obj_map, sims, text_query):
    plt.figure()
    cmap = plt.get_cmap("jet")
    for idx, (image, seg, sim, objs) in enumerate(zip(images, segms, sims, obj_map)):
        sim = sim.cpu().numpy()
        sim_norm = (sim - sim.min()) / (sim.max() - sim.min())
        
        obj = objs[sim.argmax(0).item()]
        obj_mask = seg == obj
        image = np.array(image)
        image[obj_mask==True, :] = np.array([255, 0, 0], dtype=np.uint8)
        image = Image.fromarray(image)
        plt.subplot(2, len(images), idx + 1)
        plt.imshow(image)
        
        plt.title(os.path.basename(f'view_{idx}'))
        plt.axis("off")

        tmp = np.zeros_like(seg).astype(np.float32)
        for i, obj in enumerate(objs):
            obj_mask = seg == obj
            tmp[obj_mask==True] =  sim_norm[i]

        plt.subplot(2, len(images), len(images) + idx + 1)
        heatmap = cmap(tmp)
        plt.imshow(heatmap)
        plt.axis("off")

    plt.tight_layout()
    plt.suptitle(f'Similarity to language query "{text_query}"" with object prior')

    return plt.show()


def viz_multiview_feat_scene(scene):
    # RGB and labeled pointcloud
    pc = scene['pointcloud']['xyz'][:]
    pc_rgb = scene['pointcloud']['rgb'][:]
    pc_label = scene['pointcloud']['label'][:]
    pc_anno = np.array([PALLETE_MAP[x] for x in pc_label])

    pc_o3d = to_o3d(pc, pc_rgb)
    pc_label_o3d = to_o3d(pc, pc_anno).translate([.75, 0, 0])

    # Patch MV features
    mv_patch = scene['multiview']['patch'][:]
    pca = proj.apply_pca(mv_patch)
    pca_patch_o3d = to_o3d(pc, pca).translate([0, -0.75, 0])

    # Object prior MV features
    obj_ids = scene['multiview']['obj_ids'][:] #(K,)
    per_obj = scene['multiview']['per_obj'][:] #(K, C)

    mv_obj = np.zeros((pc.shape[0], per_obj.shape[-1]), dtype=float)
    for i, obj in enumerate(obj_ids):
        ids = np.argwhere(pc_label == obj)
        mv_obj[ids, :] = per_obj[i]
    pca = proj.apply_pca(mv_obj)
    pca_obj_o3d = to_o3d(pc, pca).translate([.75, -0.75, 0])

    return o3d_viewer([pc_o3d, pc_label_o3d, pca_patch_o3d, pca_obj_o3d])


def viz_feat_scene(xyz, rgb, label, feat, trans_factor=15):
    if not isinstance(xyz, np.ndarray):
        xyz = xyz.numpy()
        rgb = rgb.numpy()
        label = label.numpy()
        feat = feat.numpy()
        
    p_o3d = to_o3d(xyz, rgb)
    py_o3d = to_o3d(xyz,
        np.array([PALLETE_MAP[x] for x in label]))
    feat /= np.linalg.norm(feat, axis=-1)[:,None]
    pca = proj.apply_pca(feat)
    pca_o3d = to_o3d(xyz, pca)
    world_f = get_coord_frame(scale=5)

    return o3d_viewer([p_o3d, world_f, 
        py_o3d.translate([trans_factor, 0, 0]),
        pca_o3d.translate([2 * trans_factor, 0, 0])
    ])


def viz_clip_pred(pc, pred, sims_norm, text_query, background, trans_factor=15):
    cmap = plt.get_cmap("turbo")
    heatmap = cmap(sims_norm)[:,:3].astype(float)
    back = background.copy()
    back[pred==True, :] = np.array([1, 0, 0])
    heatmap_o3d = to_o3d(pc, heatmap)
    thr_o3d = to_o3d(pc, back)
    return o3d_viewer([heatmap_o3d, thr_o3d.translate([trans_factor, 0, 0])], title=text_query)


def viz_clip_pred_gt(pc, pred, gt, sims_norm, text_query, background, trans_factor=15):
    cmap = plt.get_cmap("turbo")
    heatmap = cmap(sims_norm)[:,:3].astype(float)
    back = background.copy()
    back[pred==True, :] = np.array([1, 0, 0])
    heatmap_o3d = to_o3d(pc, heatmap)
    thr_o3d = to_o3d(pc, back)
    gt_o3d = copy.deepcopy(to_o3d(pc, gt[..., None].repeat(3,-1))).translate([trans_factor, 0, 0])
    return o3d_viewer([heatmap_o3d, gt_o3d, thr_o3d.translate([trans_factor*2, 0, 0])], title=text_query)
