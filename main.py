import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


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

                        current_frame = f
                        next_frame = os.path.join(d, f'{int(name) + 1}{ext}')

                        if os.path.exists(next_frame):
                            self.frames.append([current_frame, next_frame])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame = Image.open(self.frames[idx][0])
        next_frame = Image.open(self.frames[idx][1])

        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(256)
        ])

        frame = transform(frame).type(torch.float)
        next_frame = transform(next_frame).type(torch.float)

        sample = {'frame': frame, 'next_frame': next_frame}

        return sample


frame_dataset = FramesDataset(root_dir='./data/frame/')

train_loader = DataLoader(frame_dataset, batch_size=4, shuffle=True, num_workers=0)


# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch)


# fig = plt.figure()
#
# for i, sample in enumerate(frame_dataset):
#     print(i, sample['frame'].shape, sample['next_frame'].shape)
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample['frame'])
#
#     if i == 3:
#         plt.show()
#         break

#
# ########################################################################
# # The output of torchvision datasets are PILImage images of range [0, 1].
# # We transform them to Tensors of normalized range [-1, 1].
#
# ########################################################################
# # .. note::
# #     If running on Windows and you get a BrokenPipeError, try setting
# #     the num_worker of torch.utils.data.DataLoader() to 0.
#
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# batch_size = 4
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
#
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# ########################################################################
# # Let us show some of the training images, for fun.
#
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# # functions to show an image
#
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# # get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)
#
# # # show images
# # imshow(torchvision.utils.make_grid(images))
# # # print labels
# # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
#
#
# ########################################################################
# # 2. Define a Convolutional Neural Network
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# # Copy the neural network from the Neural Networks section before and modify it to
# # take 3-channel images (instead of 1-channel images as it was defined).
#
# import torch.nn as nn
# import torch.nn.functional as F
#
#
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 32, 5, padding=2)

        self.conv3 = nn.Conv2d(32, 3, 5, padding=2)
        # self.fc1 = nn.Linear(492032, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        # x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        # x = self.pool(x)

        # x = torch.flatten(x, 1)
        #
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # x = F.relu(x)
        return x


net = Net()

print(net)
total_params = sum(p.numel() for p in net.parameters())
print(f"Number of parameters: {total_params}")

criterion = nn.MSELoss(reduction='sum')
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
# ########################################################################
# # 4. Train the network
# # ^^^^^^^^^^^^^^^^^^^^
# #
# # This is when things start to get interesting.
# # We simply have to loop over our data iterator, and feed the inputs to the
# # network and optimize.
#
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        frames = data['frame']
        next_frames = data['next_frame']

        optimizer.zero_grad()

        outputs = net(frames)
        loss = criterion(frames, next_frames)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
#
# ########################################################################
# # Let's quickly save our trained model:
#
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)
#
# ########################################################################
# # See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_
# # for more details on saving PyTorch models.
# #
# # 5. Test the network on the test data
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# #
# # We have trained the network for 2 passes over the training dataset.
# # But we need to check if the network has learnt anything at all.
# #
# # We will check this by predicting the class label that the neural network
# # outputs, and checking it against the ground-truth. If the prediction is
# # correct, we add the sample to the list of correct predictions.
# #
# # Okay, first step. Let us display an image from the test set to get familiar.
#
# dataiter = iter(testloader)
# images, labels = next(dataiter)
#
# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
#
# ########################################################################
# # Next, let's load back in our saved model (note: saving and re-loading the model
# # wasn't necessary here, we only did it to illustrate how to do so):
#
# net = Net()
# net.load_state_dict(torch.load(PATH))
#
# ########################################################################
# # Okay, now let us see what the neural network thinks these examples above are:
#
# outputs = net(images)
#
# ########################################################################
# # The outputs are energies for the 10 classes.
# # The higher the energy for a class, the more the network
# # thinks that the image is of the particular class.
# # So, let's get the index of the highest energy:
# _, predicted = torch.max(outputs, 1)
#
# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
#                               for j in range(4)))
#
# ########################################################################
# # The results seem pretty good.
# #
# # Let us look at how the network performs on the whole dataset.
#
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
#
# ########################################################################
# # That looks way better than chance, which is 10% accuracy (randomly picking
# # a class out of 10 classes).
# # Seems like the network learnt something.
# #
# # Hmmm, what are the classes that performed well, and the classes that did
# # not perform well:
#
# # prepare to count predictions for each class
# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}
#
# # again no gradients needed
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predictions = torch.max(outputs, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1
#
# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
#
# ########################################################################
# # Okay, so what next?
# #
# # How do we run these neural networks on the GPU?
# #
# # Training on GPU
# # ----------------
# # Just like how you transfer a Tensor onto the GPU, you transfer the neural
# # net onto the GPU.
# #
# # Let's first define our device as the first visible cuda device if we have
# # CUDA available:
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# # Assuming that we are on a CUDA machine, this should print a CUDA device:
#
# print(device)
