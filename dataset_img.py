import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import random


class FrameYUV(object):
    def __init__(self, Y, U, V):
        self._Y = Y
        self._U = U
        self._V = V

    def read_YUV420_specified_frame(fid, width, height, idx):
        # read a frame from a YUV420-formatted sequence
        d00 = height // 2
        d01 = width // 2
        fid.seek(idx * width * height * 3 // 2)
        Y_buf = fid.read(width * height)
        Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [height, width])
        U_buf = fid.read(d01 * d00)
        U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [d00, d01])
        V_buf = fid.read(d01 * d00)
        V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [d00, d01])
        return FrameYUV(Y, U, V)

    def read_YUV420(fid, width, height):
        # read y, u, v from a YUV420 sequence
        d00 = height // 2
        d01 = width // 2
        Y_buf = fid.read(width * height)
        Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [height, width])
        U_buf = fid.read(d01 * d00)
        U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [d00, d01])
        V_buf = fid.read(d01 * d00)
        V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [d00, d01])
        return FrameYUV(Y, U, V)


def write_YUV420_frame(fid, Y_buf, U_buf, V_buf):
    # read a frame from a YUV420-formatted sequence
    Y = Y_buf.astype(np.uint8)
    U = U_buf.astype(np.uint8)
    V = V_buf.astype(np.uint8)
    fid.write(Y)
    fid.write(U)
    fid.write(V)


def write_YUV420(save_path, Y_buf, U_buf, V_buf):
    # read a frame from a YUV420-formatted sequence
    with open(save_path, 'wb') as fp:
        Y = Y_buf.astype(np.uint8)
        U = U_buf.astype(np.uint8)
        V = V_buf.astype(np.uint8)
        fp.write(Y)
        fp.write(U)
        fp.write(V)


def write_YUV420_Y(save_path, Y_buf):
    # read a frame from a YUV420-formatted sequence
    with open(save_path, 'wb') as fp:
        Y = Y_buf.astype(np.uint8)
        fp.write(Y)


def write_ycbcr(y, vid_path):
    with open(vid_path, 'wb') as fp:
        for ite_frm in range(len(y)):
            fp.write(y[ite_frm].reshape(((y[0].shape[0]) * (y[0].shape[1]),)))



def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def get_w_h(filename):
    filename = filename.split('/')[-1]
    w, h = filename.split('x')
    w = w.split('_')[-1]
    h = os.path.splitext(h)[0]
    return [int(w), int(h)]


class BaseDataset_img(Dataset):
    def __init__(self, img_dir: str, img_channel: int, QP: int) -> None:
        super(BaseDataset_img, self).__init__()

        self.qp = QP
        self.channel = img_channel
        self.yuv_raw_name = [os.path.join(img_dir + x) for x in os.listdir(img_dir) if x.endswith('png')]

    def __getitem__(self, index):
        image = self.yuv_raw_name[index]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image = image[:, :, 0]
        # image = image / 255.0
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)  # todo bit depth
        # image = np.expand_dims(image, axis=0)
        # image = image.transpose(2, 0, 1)

        qp_tensor = torch.from_numpy(np.array(self.qp)) / 100.0
        # qp_tensor = torch.from_numpy(np.array(self.qp))
        qp_tensor = qp_tensor.to('cpu').unsqueeze(0)
        image = torch.from_numpy(image.astype(np.float32)) / 255.0
        # image = torch.from_numpy(image.astype(np.float32))
        image = image.to('cpu').unsqueeze(0).unsqueeze(0)

        # image =torch.from_numpy(image)
        return qp_tensor, image

    def __len__(self) -> int:
        return len(self.yuv_raw_name)


class Val_Dataset_img(Dataset):
    def __init__(self, gt_dir: str, lq_dir: str, img_channel: int, QP: int) -> None:
        super(Val_Dataset_img, self).__init__()

        self.qp = QP
        self.channel = img_channel
        self.gt_img = [os.path.join(gt_dir + x) for x in os.listdir(gt_dir) if x.endswith('png')]
        self.lq_img = [os.path.join(lq_dir + x) for x in os.listdir(lq_dir) if x.endswith('png')]

    def __getitem__(self, index):
        gt_img = self.gt_img[index]
        gt_img = cv2.imread(gt_img)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YUV)
        gt_img = gt_img[:, :, 0]

        # gt_img = cv2.resize(gt_img, (224, 224), interpolation=cv2.INTER_AREA)  # [82 202 255]>[51 228 254]

        lq_img = self.lq_img[index]
        lq_img = cv2.imread(lq_img)
        lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2YUV)
        lq_img = lq_img[:, :, 0]

        # lq_img = cv2.resize(lq_img, (224, 224), interpolation=cv2.INTER_AREA)
        gt_img = torch.from_numpy(gt_img.astype(np.float32)) / 255.0  # todo bit depth
        gt_img = gt_img.to('cpu').unsqueeze(0).unsqueeze(0)

        lq_img = torch.from_numpy(lq_img.astype(np.float32)) / 255.0  # todo bit depth
        lq_img = lq_img.to('cpu').unsqueeze(0).unsqueeze(0)

        qp_tensor = torch.from_numpy(np.array(self.qp)) / 100.0
        qp_tensor = qp_tensor.to('cpu').unsqueeze(0)

        # image =torch.from_numpy(image)
        return qp_tensor, lq_img, gt_img

    def __len__(self) -> int:
        return len(self.gt_img)


class Val_Dataset_yuv(Dataset):
    def __init__(self, yuv_dir: str, img_channel: int, image_size: int, mode: str, QP: int) -> None:
        super(Val_Dataset_yuv, self).__init__()
        self.mode = mode
        self.qp = QP
        self.channel = img_channel
        self.patchsize = image_size

        self.yuv_raw_name = [os.path.join(yuv_dir + '/GT/', x) for x in os.listdir(yuv_dir + '/GT/')
                             if x.endswith('yuv')]
        self.yuv_cur_qp_name = [os.path.join(yuv_dir + '/QP{}/'.format(QP), x)
                                for x in os.listdir(yuv_dir + '/QP{}/'.format(QP)) if x.endswith('yuv')]

    def __getitem__(self, index_frame):
        raw_name = self.yuv_raw_name[index_frame]
        wh = get_w_h(raw_name)
        # qp_idx = random.randint(0, 3)
        fid_yuv_raw = open(self.yuv_raw_name[index_frame], 'rb')
        # cmp_list = [self.yuv_qp22_name[0], self.yuv_qp27_name[0], self.yuv_qp32_name[0], self.yuv_qp37_name[0]]
        # fid_yuv_cmp = open(cmp_list[qp_idx], 'rb')
        fid_yuv_cmp = open(self.yuv_cur_qp_name[index_frame], 'rb')

        frame_YUV_raw = FrameYUV.read_YUV420(fid_yuv_raw, wh[0], wh[1])    # 根据idx获取对应的帧
        frame_YUV_cmp = FrameYUV.read_YUV420(fid_yuv_cmp, wh[0], wh[1])

        input = frame_YUV_cmp._Y
        target = frame_YUV_raw._Y

        H = target.shape[0]
        W = target.shape[1]
        if self.channel == 1:
            img_in = np.expand_dims(input, 2)
            img_gt = np.expand_dims(target, 2)

        if self.mode == "train":
            # randomly crop
            rnd_h = random.randint(0, max(0, H - self.patchsize))
            rnd_w = random.randint(0, max(0, W - self.patchsize))
            img_in = img_in[rnd_h:rnd_h + self.patchsize, rnd_w:rnd_w + self.patchsize, :]
            img_gt = img_gt[rnd_h:rnd_h + self.patchsize, rnd_w:rnd_w + self.patchsize, :]

            img_in, img_gt = augment([img_in, img_gt])
        else:
            boundary_left = (W // 2 - self.patchsize) if (W // 2 - self.patchsize) > 0 else 0
            boundary_right = (W // 2 + self.patchsize) if (W // 2 + self.patchsize) < W else W

            boundary_top = (H // 2 - self.patchsize) if (H // 2 - self.patchsize) > 0 else 0
            boundary_bottom = (H // 2 + self.patchsize) if (H // 2 + self.patchsize) < H else H
            img_in = img_in[boundary_top:boundary_bottom, boundary_left:boundary_right, :]
            img_gt = img_gt[boundary_top:boundary_bottom, boundary_left:boundary_right, :]

        img_in = np.array(img_in).astype(np.float32)
        img_gt = np.array(img_gt).astype(np.float32)

        img_in = np.transpose(img_in, axes=[2, 0, 1])  # TODO 为什么要转换
        # img_in4 = np.expand_dims(img_in4, axis=0)
        img_gt = np.transpose(img_gt, axes=[2, 0, 1])

        # normalization
        img_in /= 255.0
        # img_in4 /= 255.0
        img_gt /= 255.0
        qp = self.qp/100.0
        qp = np.array(qp).astype(np.float32)
        return qp, img_in, img_gt

    def __len__(self) -> int:
        return len(self.yuv_raw_name)
