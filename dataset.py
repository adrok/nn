import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import cv2


class FramesDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.frames = []

        for dirname in os.listdir(root_dir):
            d = os.path.join(root_dir, dirname)
            if not os.path.isdir(d):
                continue

            for filename in os.listdir(d):
                f = os.path.join(d, filename)
                if not os.path.isfile(f):
                    continue

                name, ext = os.path.splitext(os.path.basename(f))

                path = os.path.join(d, f'{int(name)}{ext}')
                original = cv2.imread(path)

                t = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ])

                t_low = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                ])

                self.frames.append(
                    [t(original), t_low(original)]
                )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.frames[idx]
