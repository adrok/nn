import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import torchvision


def load_img(path):
    img = torchvision.io.read_image(path)

    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((152*10, 152*15)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
    ])

    return t(img)


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

                self.frames.append(path)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.frames[idx]

        original = torchvision.io.read_image(path)

        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((152, 152)),
            transforms.ToTensor(),
        ])

        t_low = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((152, 152)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ])

        return t(original), t_low(original)
