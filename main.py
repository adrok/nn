import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from torch import nn
import torch.optim as optim
import cv2
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FramesDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.frames = []

        for dirname in os.listdir(root_dir):
            d = os.path.join(root_dir, dirname)
            if os.path.isdir(d):
                for filename in os.listdir(d):
                    f = os.path.join(d, filename)
                    if os.path.isfile(f):
                        name, ext = os.path.splitext(os.path.basename(f))

                        # current_frame = f
                        path = os.path.join(d, f'{int(name)}{ext}')
                        frame = cv2.imread(path)

                        t = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                        ])

                        blured = cv2.blur(frame, (50, 50))

                        self.frames.append(
                            [t(frame), t(blured)]
                        )
                        # next_frame = os.path.join(d, f'{int(name) + 1}{ext}')
                        #
                        # if os.path.exists(next_frame):
                        #     self.frames.append([current_frame, next_frame])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame = self.frames[idx][0]
        frame_blur = self.frames[idx][1]

        return frame_blur, frame


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 6, padding=2),
            nn.ReLU(True)
        )
        # self.conv2 = nn.Conv2d(64, 32, 1, padding=2)
        self.conv3 = nn.Conv2d(64, 3, 4, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(self.conv2(x))
        x = self.conv3(x)

        return x


dataset = FramesDataset(root_dir='./data/frame/')

train_len = len(dataset) - 40
val_len = len(dataset) - train_len

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0)

net = Net()
net.to(device)

print(net)
total_params = sum(p.numel() for p in net.parameters())
print(f"Number of parameters: {total_params}")

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=5,
    factor=0.5,
    verbose=True
)

for epoch in range(50):
    train_loss = 0.0
    ts = time.time()


    for i, data in enumerate(train_loader, 0):
        (frames_blured, frames_original) = data

        frames_blured = frames_blured.to(device)
        frames_original = frames_original.to(device)

        optimizer.zero_grad()

        outputs = net(frames_blured)

        loss = criterion(outputs, frames_original)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        if i % 100 == 99:
            print(f'train: [{epoch + 1:2d}, {i + 1:3d}] loss: {train_loss / 100:.5f} took {(time.time() - ts):.2f}s')
            train_loss = 0.0

        if i == len(train_loader) - 1:
            img = frames_blured.cpu().data
            torchvision.utils.save_image(img, f"./outputs/{epoch}_input.jpg")

            img = outputs.cpu().data
            torchvision.utils.save_image(img, f"./outputs/{epoch}_output.png")

            img = frames_original.cpu().data
            torchvision.utils.save_image(img, f"./outputs/{epoch}_original.jpg")


            # scheduler.step(train_loss)

    # with torch.no_grad():
    #     val_loss = 0.0
    #     for i, data in enumerate(val_loader, 0):
    #         (frames_blured, frames_original) = data
    #
    #         outputs = net(frames_blured)
    #
    #         loss = criterion(outputs, frames_original)
    #
    #         val_loss += loss.item()
    #         if i % 10 == 9:
    #             print(f'validation: [{epoch + 1}, {i + 1:3d}] loss: {val_loss / 10:.5f}')
    #             val_loss = 0.0

print('Finished Training')
