import os
import random
import numpy as np
from PIL import Image
from loguru import logger
import sys
import inspect

import cv2
import torch
from torch import nn
import torch.distributed as dist


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    '''poly learning rate policy'''
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


@torch.no_grad()
def trainMetricPC(output, target, threshold=0.35, pr_ious=[0.25, 0.5, 0.75], sigmoid=False):
    assert len(output) == len(target)
    mean_iou = 0.0
    mean_prec = [0.0] * len(pr_ious)
    num_ins = len(output)
    count = 1e-6
    for (pred_pc_m, gt_pc_m) in zip(output, target):
        count += 1
        if sigmoid:
            pred_pc_m = torch.sigmoid(pred_pc_m).squeeze()
        else:
            pred_pc_m = pred_pc_m.squeeze()

        pred_pc_m[pred_pc_m < threshold] = 0.
        pred_pc_m[pred_pc_m >= threshold] = 1.

        # inter & union
        inter = (pred_pc_m.bool() & gt_pc_m.bool()).sum()  # b
        union = (pred_pc_m.bool() | gt_pc_m.bool()).sum()  # b
        iou = inter / (union + 1e-6)  # 0 ~ 1
        mean_iou += iou        
        for j, pr_iou in enumerate(pr_ious):
            prec = (iou > pr_iou).float()
            mean_prec[j] += prec

    mean_iou /= count + 1e-6
    mean_prec = [prec / count for prec in mean_prec]

    return 100. * mean_iou, [100. * x for x in mean_prec]


def init_random_seed(seed=None, device='cuda', rank=0, world_size=1):
    """Initialize random seed."""
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", index=0):
        self.name = name
        self.fmt = fmt
        self.index = index
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.name == "Lr":
            fmtstr = "{name}={val" + self.fmt + "}"
        else:
            fmtstr = "{name}={val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("  ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def trainMetricGPU(output, target, threshold=0.35, pr_iou=0.5, sigmoid=True):
    assert (output.dim() in [2, 3, 4])
    assert output.shape == target.shape
    output = output.flatten(1)
    target = target.flatten(1)
    if sigmoid:
        output = torch.sigmoid(output)
    output[output < threshold] = 0.
    output[output >= threshold] = 1.
    # inter & union
    inter = (output.bool() & target.bool()).sum(dim=1)  # b
    union = (output.bool() | target.bool()).sum(dim=1)  # b
    ious = inter / (union + 1e-6)  # 0 ~ 1
    # iou & pr@5
    iou = ious.mean()
    prec = (ious > pr_iou).float().mean()
    return 100. * iou, 100. * prec


def ValMetricGPU(output, target, threshold=0.35):
    assert output.size(0) == 1
    output = output.flatten(1)
    target = target.flatten(1)
    output = torch.sigmoid(output)
    output[output < threshold] = 0.
    output[output >= threshold] = 1.
    # inter & union
    inter = (output.bool() & target.bool()).sum(dim=1)  # b
    union = (output.bool() | target.bool()).sum(dim=1)  # b
    ious = inter / (union + 1e-6)  # 0 ~ 1
    return ious


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3, 4])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

def group_weight(weight_group, module, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(
        module.parameters())) == len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth.
        Default value: 0.

    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """
    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=loguru_format,
            level="INFO",
            enqueue=True,
        )
        logger.add(save_file)

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO")


def get_seg_image(img: np.array, mask: np.array) -> np.array:
    # My stupid way, don't use it...
    # mask = (1 * np.logical_or.reduce(mask)).astype('uint8')
    mask_inv = cv2.bitwise_not(mask)
    res = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    background = cv2.bitwise_and(gray, gray, mask=mask_inv)
    background = np.stack((background,)*3, axis=-1)
    img_ca = res
    # img_ca = cv2.add(res, background)
    return img_ca


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def calculate_metrics(iou_list, iou3d_list, box_iou_list, subset="", epoch=0, total_epoch=0, prefix=""):
    prec_list = []
    pc_prec_list = []
    box_prec_list = []
    for thres in [0.25, 0.5, 0.75]:
        tmp = (iou_list > thres).float().mean()
        pc_tmp = (iou3d_list > thres).float().mean()
        box_tmp = (box_iou_list > thres).float().mean()
        prec_list.append(tmp)
        pc_prec_list.append(pc_tmp)
        box_prec_list.append(box_tmp)
    
    iou = iou_list.mean()
    iou3d = iou3d_list.mean()
    box_iou = box_iou_list.mean()

    results = {
        f"{prefix}-IoU": iou,
        f"{prefix}-3DIoU": iou3d,
        f"{prefix}-BoxIoU": box_iou,
    }

    temp = f'IoU={100.*iou.item():.2f}  '
    pc_temp = f'3DIoU={100.*iou3d.item():.2f}  '
    box_temp = f'BoxIoU: {100.*box_iou.item():.2f}  '
    for i, thres in enumerate([25, 50, 75]):
        key = 'Pr@{}'.format(thres)
        value = prec_list[i].item()
        pc_value = pc_prec_list[i].item()
        box_value = box_prec_list[i].item()

        results = {
            **results,
            f"{prefix}-Pr@{thres}": value,
            f"{prefix}-3DPr@{thres}": pc_value,
            f"{prefix}-BoxPr@{thres}": box_value
        }
        
        temp += f"Pr@{thres}: {100.*value:.2f}  "
        pc_temp += f"3DPr@{thres}: {100.*pc_value:.2f}  "
        box_temp += f"BoxPr@{thres}: {100.*box_value:.2f}  "
    head = f'Evaluation {subset}: Epoch=[{epoch}/{total_epoch}]  '
    log = head + temp + pc_temp + box_temp

    return log, results



def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    '''Sets the learning rate to the base LR decayed by 10 every step epochs'''
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    '''poly learning rate policy'''
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3, 4])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3, 4])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def export_pointcloud(name, points, colors=None, normals=None):
    if len(points.shape) > 2:
        points = points[0]
        if normals is not None:
            normals = normals[0]
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        if normals is not None:
            normals = normals.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(name, pcd)

def export_mesh(name, v, f, c=None):
    if len(v.shape) > 2:
        v, f = v[0], f[0]
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    if c is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_triangle_mesh(name, mesh)

def visualize_labels(u_index, labels, palette, out_name, loc='lower left', ncol=7):
    patches = []
    for i, index in enumerate(u_index):
        label = labels[index]
        cur_color = [palette[index * 3] / 255.0, palette[index * 3 + 1] / 255.0, palette[index * 3 + 2] / 255.0]
        red_patch = mpatches.Patch(color=cur_color, label=label)
        patches.append(red_patch)
    plt.figure()
    plt.axis('off')
    legend = plt.legend(frameon=False, handles=patches, loc=loc, ncol=ncol, bbox_to_anchor=(0, -0.3), prop={'size': 5}, handlelength=0.7)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array([-5,-5,5,5])))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(out_name, bbox_inches=bbox, dpi=300)
    plt.close()

def get_palette(num_cls=21, colormap='scannet'):
    if colormap == 'scannet':
        scannet_palette = []
        for _, value in SCANNET_COLOR_MAP_20.items():
            scannet_palette.append(np.array(value))
        palette = np.concatenate(scannet_palette)
    elif colormap == 'matterport':
        scannet_palette = []
        for _, value in MATTERPORT_COLOR_MAP_21.items():
            scannet_palette.append(np.array(value))
        palette = np.concatenate(scannet_palette)
    elif colormap == 'matterport_160':
        scannet_palette = []
        for _, value in MATTERPORT_COLOR_MAP_160.items():
            scannet_palette.append(np.array(value))
        palette = np.concatenate(scannet_palette)
    elif colormap == 'nuscenes16':
        nuscenes16_palette = []
        for _, value in NUSCENES16_COLORMAP.items():
            nuscenes16_palette.append(np.array(value))
        palette = np.concatenate(nuscenes16_palette)
    else:
        n = num_cls
        palette = [0]*(n*3)
        for j in range(0,n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while lab > 0:
                palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                i = i + 1
                lab >>= 3
    return palette

def convert_labels_with_palette(input, palette):
    '''Get image color palette for visualizing masks'''

    new_3d = np.zeros((input.shape[0], 3))
    u_index = np.unique(input)
    for index in u_index:
        if index == 255:
            index_ = 20
        else:
            index_ = index

        new_3d[input==index] = np.array(
            [palette[index_ * 3] / 255.0,
             palette[index_ * 3 + 1] / 255.0,
             palette[index_ * 3 + 2] / 255.0])

    return new_3d