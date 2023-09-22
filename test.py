from dataset import load_img
import torch
import torchvision
from model import Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Model()
model.load_state_dict(torch.load('./data/model.pth'))
model.eval()

for i in ['tree', 'room']:
    img = load_img(f'data/{i}.png')
    img = img.to(device)
    img = img.unsqueeze(0)
    outputs = model(img)
    to_save = torch.cat((img, outputs), dim=0).cpu().data
    torchvision.utils.save_image(to_save, f"./outputs/test_{i}.png")
