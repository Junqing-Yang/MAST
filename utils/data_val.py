import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import torch
from torch.utils.data.distributed import DistributedSampler


# several data augumentation strategies
def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


class VideoDataset(data.Dataset):
    def __init__(self, video_dataset, trainsize, time_interval=2, video_time_clips=1):
        super(VideoDataset, self).__init__()
        self.time_clips = video_time_clips
        self.video_train_list = []
        self.trainsize = trainsize

        video_root = video_dataset
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')

        cls_list = os.listdir(img_root)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []
            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)

            tmp_list = os.listdir(cls_img_path)

            tmp_list.sort(key=lambda name: (name.split('.jpg')[0]))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png"))
                ))
        # ensemble - change to continuous frames 2023.9.3
        for cls in cls_list:
            li = self.video_filelist[cls]
            for begin in range(1, len(li) - self.time_clips * time_interval):
                batch_clips = []
                batch_clips.append(li[begin-1])
                for t in range(1,self.time_clips+1):
                    batch_clips.append(li[begin + time_interval * t])
                self.video_train_list.append(batch_clips)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        for idx, (img_path, label_path) in enumerate(img_label_li):
            img = self.rgb_loader(img_path)
            label = self.binary_loader(label_path)
            img, label = cv_random_flip(img, label)
            img, label = randomCrop(img, label)
            img, label = randomRotation(img, label)
            img = colorEnhance(img)
            label = randomPeper(label)
            img_li.append(self.img_transform(img))
            label_li.append(self.gt_transform(label))

        for idx, (img, label) in enumerate(zip(img_li, label_li)):
            # print("idx: ", idx, "img: ", img_label_li[idx])
            if idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(label_li), *(label.shape))
            IMG[idx, :, :, :] = img
            LABEL[idx, :, :, :] = label
        IMG_PRE = IMG[:1, :, :, :]
        IMG_POST = IMG[1:, :, :, :]
        LABEL_PRE = LABEL[:1, :, :, :]
        LABEL_POST = LABEL[1:, :, :, :]
        # print("IMG_PRE: ", IMG_PRE.shape, "IMG_POST: ", IMG_POST.shape, "LABEL_PRE: ", LABEL_PRE.shape, "LABEL_POST: ", LABEL_POST.shape)
        return IMG_PRE, LABEL_PRE, IMG_POST, LABEL_POST
        # return IMG, LABEL

    def __len__(self):
        return len(self.video_train_list)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

def get_loader(image_root, batchsize, trainsize, shuffle=False, num_workers=2, pin_memory=True):
    dataset = VideoDataset(image_root, trainsize, time_interval=2)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers,
                                  pin_memory=pin_memory,sampler=DistributedSampler(dataset))
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images, self.gts = [], []
        for i in os.listdir(image_root):
            self.images += [os.path.join(image_root , i , k) for k in os.listdir(os.path.join(image_root , i))]
            self.gts += [os.path.join(gt_root , i , k) for k in os.listdir(os.path.join(gt_root ,i))]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-2:]

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name[-1].endswith('.jpg'):
            name[-1] = name[-1].split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        if path.endswith('.jpg') or path.endswith('.png'):
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        else:
            pass

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
