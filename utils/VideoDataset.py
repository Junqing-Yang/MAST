class VideoDataset(data.Dataset):
    def __init__(self, video_dataset, trainsize, time_interval=1, video_time_clips=5):
        super(VideoDataset, self).__init__()
        self.time_clips = video_time_clips
        self.video_train_list = []
        self.trainsize = trainsize

        video_root = video_dataset
        img_root = os.path.join(video_root, 'Imgs')
        gt_root = os.path.join(video_root, 'GT')

        cls_list = os.listdir(img_root)

        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)

            tmp_list = os.listdir(cls_img_path)

            tmp_list.sort(key=lambda name: (
                int(name.split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png"))
                ))
        # ensemble
        for cls in cls_list:
            li = self.video_filelist[cls]
            for begin in range(1, len(li) - (self.time_clips - 1) * time_interval - 1):
                batch_clips = []
                batch_clips.append(li[0])
                for t in range(self.time_clips):
                    batch_clips.append(li[begin + time_interval * t])
                self.video_train_list.append(batch_clips)

        # self.img_label_transform = transform

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
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            img = self.img_transform(img)
            label = self.gt_transform(label)
            print(img.size())
            img_li.append(img)
            label_li.append(label)

        for idx, (img, label) in enumerate(zip(img_li, label_li)):
            if idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li) - 1, *(label.shape))

                IMG[idx, :, :, :] = img
            else:
                IMG[idx, :, :, :] = img
                LABEL[idx - 1, :, :, :] = label


        return IMG, LABEL

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


# dataloader for training
def get_loader(image_root, batchsize, trainsize, shuffle=True, num_workers=2, pin_memory=True):

    dataset = VideoDataset(image_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader