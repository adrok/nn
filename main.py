import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm

from dataset import FramesDataset, load_img
from model import Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ds = FramesDataset(root_dir='./data/frame/')

train_len = len(ds) - 100
val_len = len(ds) - train_len

train_dataset, val_dataset = torch.utils.data.random_split(ds, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=0)

model = Model()
model.to(device)

print(model)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {total_params}")

summary(model, input_size=(3, 1024, 1024))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=5,
    factor=0.5,
    verbose=True
)

train_losses = []
for epoch in range(50):
    train_loss = 0.0

    for data in tqdm(train_loader):
        original, inputs = data

        original = original.to(device)
        inputs = inputs.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, original)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss = train_loss / len(train_loader)
    train_losses.append(train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # save
    with torch.no_grad():
        model.eval()
        for _, data in list(enumerate(val_loader))[:10]:
            original, inputs = data

            inputs = inputs.to(device)
            original = original.to(device)

            outputs = model(inputs)

            img = torch.cat((inputs, outputs, original), dim=0).cpu().data
            torchvision.utils.save_image(img, f"./outputs/{epoch}.jpg", nrow=10)

        img = load_img("./data/img.png")
        img = img.to(device)
        img = img.unsqueeze(0)
        outputs = model(img)
        to_save = torch.cat((img, outputs), dim=0).cpu().data
        torchvision.utils.save_image(to_save, f"./outputs/{epoch}_x.jpg")

    # img = outputs.cpu().data
    # torchvision.utils.save_image(img, f"./outputs/{epoch}_output.png")
    #
    # img = high_resolution.cpu().data
    # torchvision.utils.save_image(img, f"./outputs/{epoch}_original.jpg")

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
