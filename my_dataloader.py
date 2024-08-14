import os
from utils.general import LOGGER, NUM_THREADS, segments2boxes, TQDM_BAR_FORMAT
from utils.torch_utils import torch_distributed_zero_first
from torch.utils.data import Dataset, distributed, DataLoader, dataloader
import cv2
from pathlib import Path
import glob
from tqdm import tqdm
from multiprocessing.pool import Pool
import numpy as np
from utils.augmentations import letterbox
from utils.general import xywhn2xyxy, xyxy2xywhn
import random
import psutil
from itertools import repeat
from PIL import Image, ImageOps
import torch
import pickle

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders


def img2label_paths(img_path):
    img_path = list(img_path)
    last_part = os.path.basename(os.path.dirname(os.path.dirname(img_path[0])))
    labels_dir_path = r'/mnt/block0/optics_data/datax/labels'
    # labels_dir_path = r'G:\pycharmProject\yolov9_transformer\datasets\little_test\labels'
    labels_train_path = os.path.join(labels_dir_path, last_part)
    labels = []
    if not os.path.exists(labels_train_path):
        os.makedirs(labels_train_path)
    for image in img_path:
        labels_path = image.replace("images", "labels")
        labels_path = labels_path.replace("/cam0", "")
        labels_path = os.path.splitext(labels_path)[0] + '.txt'
        labels.append(labels_path)
    return labels


'''
def img2label_paths(img_path):
    img_path = list(img_path)
    last_part = os.path.basename(os.path.dirname(os.path.dirname(img_path[0])))
    path_parts = img_path[0].split('\\')
    target_folder_name = 'labels'
    for i in range(len(path_parts) - 2, -1, -1):
        if path_parts[i] == 'images':
            # 找到 'images' 目录，删除它及其后面的所有部分
            path_parts = path_parts[:i+1]
            break
    path_parts[path_parts.index('images')] = target_folder_name
    labels_dir_path  = '\\'.join(path_parts)
    labels_train_path = os.path.join(labels_dir_path, last_part)
    labels = []
    if not os.path.exists(labels_train_path):
        os.makedirs(labels_train_path)
    for image in img_path :
        labels_path = image.replace("images", "labels")
        labels_path = labels_path.replace(r"\\cam0", "")
        labels_path =  os.path.splitext(labels_path)[0] + '.txt'
        labels.append(labels_path)
    return labels
'''


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = im.size  # image size
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


class LoadImagesAndLabels(Dataset):
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def __init__(self, path, img_size=640, batch_size=16, hyp=None,
                 rect=False, image_weights=False, cache_images=False,
                 single_cls=False, stride=32, pad=0.0, prefix=''):
        self.im_files = {}
        self.npy_files = {}
        self.path = path
        self.img_size = img_size
        self.stride = stride
        self.hyp = hyp
        self.rect = False if image_weights else rect
        self.image_weights = image_weights

        for cam in os.listdir(self.path):
            Cam_pwd_path = []
            cam_pwd_path = os.path.join(self.path, cam)
            for img_file in os.listdir(cam_pwd_path):
                if img_file.endswith(".jpg"):
                    Cam_pwd_path.append(os.path.join(cam_pwd_path, img_file))
            self.im_files[cam] = Cam_pwd_path

        for i in range(len(self.im_files['cam8'])):
            cam8 = self.im_files['cam8'][i].split('/')[-1]
            cam8 = cam8.split('.')[0]

            cam0 = self.im_files['cam0'][i].split('/')[-1]
            cam0 = cam0.split('.')[0]
            if cam0 != cam8:
                self.im_files['cam0'] = [pp.replace('cam8', 'cam0') for pp in self.im_files['cam8']]
                break

        # Check cache   （.cache 加载进 创建好的label_files  .txt中）
        self.label_files = img2label_paths(self.im_files['cam0'])  # labels
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0, f'{prefix}No labels found in {cache_path}, can not start training.'

        # Read cache
        cache.pop('msgs')  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0, f'{prefix}All labels empty in {cache_path}, can not start training.'
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.label_files = img2label_paths(cache.keys())  # update

        # Create indices
        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Rectangular Training
        '''
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]
        '''

        # Cache images into RAM/disk for faster training
        if cache_images == 'ram' and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        cam_keys = self.im_files.keys()
        self.npy_files = {}
        for cam_key in cam_keys:
            self.npy_files[cam_key] = [Path(f).with_suffix('.npy') for f in self.im_files[cam_key]]
        '''
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({b / gb:.1f}GB {cache_images})'
            pbar.close()
        '''

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning {path.parent / path.stem}..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files['cam0'], self.label_files,
                                                          repeat(prefix))), desc=desc,
                        total=len(self.im_files),
                        bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING ⚠️ No labels found in {path}.')
        x['results'] = nf, nm, ne, nc, len(self.im_files['cam0'])
        x['msgs'] = msgs  # warnings
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def check_cache_ram(self, safety_margin=0.1, prefix=''):
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.n, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.img_size / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.n / n  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required * (1 + safety_margin) < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(f"{prefix}{mem_required / gb:.1f}GB RAM required, "
                        f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, "
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files['cam0'][i]))

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims[i], self.im_files['cam0'][i], self.npy_files[i],
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def load_image9(self, i):
        cam_keys = self.im_files.keys()
        im_9 = []
        h0_9 = []
        w0_9 = []
        h_9 = []
        w_9 = []
        imgs = []
        imgs_index = [None] * 9

        for cam in cam_keys:
            index_key = int(cam[-1])
            imgs_index[index_key] = self.im_files[cam][i]
            imgs.append(self.im_files[cam][i])
        im_9_path = imgs_index

        for path in im_9_path:
            try:
                image = cv2.imread(path)
            except:
                print(path)
            image_array = np.array(image)  # 转换为NumPy..数组

            h0, w0 = image_array.shape[:2]
            r = self.img_size / max(h0, w0)
            h = int(h0 * r)
            w = int(w0 * r)
            image_array = cv2.resize(image_array, (w, h))
            im_9.append(image_array)
            h0_9.append(h0)
            w0_9.append(w0)
            h_9.append(h)
            w_9.append(w)
        return im_9, h0_9, w0_9, h_9, w_9

    def __len__(self):
        return len(self.im_files['cam0'])

    def __getitem__(self, index):  # 返回索引处的9通道  img 和  labels
        
        if index >= len(self.indices):
            '''
            print('\n')
            print(index)
            print(len(self.indices))
            print('+++++++++++++++++')
            '''
            
            index = index - 60
        '''
        if index < 60:
            print(index)
        '''
        index = self.indices[index]  # linear, shuffled, or image_weights
        hyp = self.hyp
        # Load image
        img_9, h0_9, w0_9, h_9, w_9 = self.load_image9(index)

        # Letterbox
        shape = self.img_size
        torch_img_9 = []
        labels_out_9 = []
        shapes_9 = []
        im_files = []
        imgs_index = [None] * 9
        cam_keys = self.im_files.keys()

        for cam in cam_keys:
            index_key = int(cam[-1])
            imgs_index[index_key] = self.im_files[cam][index]
        im_files = imgs_index

        
        '''
        for cam_key in cam_keys:  # 加载 cam0 -- cam8 的第 index 张图片的路径
            im_files.append(self.im_files[cam_key][index])
        '''

        for i in range(9):
            img, ratio, pad = letterbox(img_9[i], shape, auto=False, scaleup=False)  # resize图像并padding
            shapes = (h0_9[i], w0_9[i]), ((h_9[i] / h0_9[i], w_9[i] / w0_9[i]), pad)  # for COCO mAP rescaling
            shapes_9.append(shapes)

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w_9[i], ratio[1] * h_9[i], padw=pad[0],
                                           padh=pad[1])
            nl = len(labels)  # number of labels
            if nl:
                labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)
            labels_out = torch.zeros((nl, 6))
            if nl:
                labels_out[:, 1:] = torch.from_numpy(labels)  # 将 numpy 转换为 torch
            labels_out_9.append(labels_out)

            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            torch_img_9.append(torch.from_numpy(img))

        return torch_img_9, labels_out_9, im_files, shapes_9

    @staticmethod
    def collate_fn(batch):
        im_9, label_9, path_9, shapes_9 = zip(*batch)
        tuples = len(im_9)
        batch_label_9 = []
        batch_im_9 = []
        for i in range(9):
            lbs = []
            for j in range(tuples):
                label = label_9[j][i]
                if label.shape[0] == 0:
                    label = torch.zeros((1, 6))
                lbs.append(label)
            lbs = tuple(lbs)

            lbs_new = torch.stack(lbs, dim=0)
            batch_label_9.append(lbs_new)

        for i in range(9):
            ims = []
            for j in range(tuples):
                img = im_9[j][i][0]
                ims.append(img)
            ims = tuple(ims)
            ims_new = torch.stack(ims, dim=0)
            batch_im_9.append(ims_new)

        return batch_im_9, batch_label_9, path_9, shapes_9


def seed_worker(worker_id):
    # Set dataloader worker seed
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False,
                      hyp=None, cache=False, rect=False, rank=-1, workers=8,
                      image_weights=False, min_items=0, prefix=''):
    if rect:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path=path,
                                      img_size=imgsz, batch_size=batch_size, hyp=hyp,
                                      rect=rect, image_weights=image_weights, cache_images=cache,
                                      single_cls=single_cls, stride=stride, prefix=prefix)
        batch_size = min(batch_size, len(dataset))
        nd = torch.cuda.device_count()  # number of CUDA devices
        nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
        sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=False)
        # loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
        loader = DataLoader
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)
        return loader(dataset,
                      batch_size=batch_size,
                      shuffle=True and sampler is None,
                      num_workers=nw,
                      sampler=sampler,
                      pin_memory=PIN_MEMORY,
                      collate_fn=LoadImagesAndLabels.collate_fn,
                      worker_init_fn=seed_worker,
                      generator=generator, 
                      drop_last=True), dataset

