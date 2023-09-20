import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import time

from dataset import FramesDataset
from model import Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ds = FramesDataset(root_dir='./data/frame/')

train_len = len(ds) - 10
val_len = len(ds) - train_len

train_dataset, val_dataset = torch.utils.data.random_split(ds, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

model = Model()
model.to(device)

print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
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
        (high_resolution, low_resolution) = data

        high_resolution = high_resolution.to(device)
        low_resolution = low_resolution.to(device)

        optimizer.zero_grad()

        outputs = model(low_resolution)

        loss = criterion(outputs, high_resolution)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        if i % 10 == 9:
            print(f'train: [{epoch + 1:2d}, {i + 1:3d}] loss: {train_loss / 10:.5f} took {(time.time() - ts):.2f}s')
            train_loss = 0.0

        if i == len(train_loader) - 1:
            img = low_resolution.cpu().data
            torchvision.utils.save_image(img, f"./outputs/{epoch}_input.jpg")

            img = outputs.cpu().data
            torchvision.utils.save_image(img, f"./outputs/{epoch}_output.png")

            img = high_resolution.cpu().data
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
