import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import cv2
import os
import matplotlib.pyplot as plt

def tostring(n):
    if n < 10:
        return "00"+str(n)
    if n < 100:
        return "0"+str(n)
    return str(n)

def masking(img):
    # im_gr = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img = np.array(img, np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipse = cv2.fitEllipse(contours[0])
    return cv2.ellipse(img, ellipse, (255,255,255), -1)

class CustomDataset(Dataset):
    def __init__(self, link, arg):
        self.arg = arg
        self.link = link

    def __getitem__(self, idx):
        if self.arg == 'valid':
            idx += 200
        img = cv2.imread(self.link+tostring(idx)+"_HC.png", cv2.IMREAD_GRAYSCALE)
        ano = cv2.imread(self.link+tostring(idx)+"_HC_Annotation.png", cv2.IMREAD_GRAYSCALE)
        if img.shape[0] > 540:
            img = img[:540, :]
            ano = ano[:540, :]
        if img.shape[1] > 800:
            img = img[:, :800]
            ano = ano[:, :800]
        if img.shape[0] < 540 or img.shape[1] < 800:
            new_img = np.zeros((540, 800))
            new_ano = np.zeros((540, 800))
            new_img[:img.shape[0], :img.shape[1]] = img
            new_ano[:img.shape[0], :img.shape[1]] = ano
            img = new_img
            ano = new_ano
        img = torch.FloatTensor(img).unsqueeze(dim=0)
        mask = np.zeros((548, 804))
        mask[4:-4, 2:-2] = masking(ano)
        mask = (mask/255).astype(int)
        mask = torch.FloatTensor(mask).unsqueeze(dim=0)

        return img, mask

    def __len__(self):
        if self.arg == 'valid':
            return 200
        return 606

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inblock1 = nn.Sequential(
            nn.Conv2d(1, 64, 3,padding=(96, 94)),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU()
        )
        self.inblock2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU()
        )
        self.inblock3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
        )
        self.inblock4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 2, 2)
        )

        self.outblock4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2)
        )
        self.outblock3 = nn.Sequential(
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2)
        )
        self.outblock2 = nn.Sequential(
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2)
        )
        self.outblock1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1),
            nn.Conv2d(2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        in1 = self.inblock1(data)
        in2 = self.inblock2(in1)
        in3 = self.inblock3(in2)
        in4 = self.inblock4(in3)

        out4 = self.bottleneck(in4)
        x4, y4 = out4.shape[-2], out4.shape[-1]
        out4 = torch.cat([in4[:,:,in4.shape[-2]//2-x4//2:in4.shape[-2]//2-x4//2+x4, in4.shape[-1]//2-y4//2:in4.shape[-1]//2-y4//2+y4], out4], dim=1)

        out3 = self.outblock4(out4)
        x3, y3 = out3.shape[-2], out3.shape[-1]
        out3 = torch.cat([in3[:,:,in3.shape[-2]//2-x3//2:in3.shape[-2]//2-x3//2+x3, in3.shape[-1]//2-y3//2:in3.shape[-1]//2-y3//2+y3], out3], dim=1)

        out2 = self.outblock3(out3)
        x2, y2 = out2.shape[-2], out2.shape[-1]
        out2 = torch.cat([in2[:,:,in2.shape[-2]//2-x2//2:in2.shape[-2]//2-x2//2+x2, in2.shape[-1]//2-y2//2:in2.shape[-1]//2-y2//2+y2], out2], dim=1)

        out1 = self.outblock2(out2)
        x1, y1 = out1.shape[-2], out1.shape[-1]
        out1 = torch.cat([in1[:,:,in1.shape[-2]//2-x1//2:in1.shape[-2]//2-x1//2+x1, in1.shape[-1]//2-y1//2:in1.shape[-1]//2-y1//2+y1], out1], dim=1)

        output = self.outblock1(out1)
        return output

trainset = CustomDataset("training_set/", "train")
validset = CustomDataset("training_set/", "valid")
train_loader = DataLoader(trainset, batch_size=4, shuffle=True)
valid_loader = DataLoader(validset, batch_size=4, shuffle=True)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
model = UNet().to(device)
# model.load_state_dict(torch.load("state/model_e90.pth"))

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.BCELoss()

E = 101
# train_loss_hist = []
# valid_loss_hist = []
# train_loss_hist = np.load("train_loss_hist.npy")[:91].tolist()
for e in range(E):
    train_loss = 0
    # train_acc = 0
    model.train()
    for x, y  in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output,y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # train_acc += (np.sum(np.array(output.argmax(dim=1).tolist()) == np.array(y.tolist())))/23000
    train_loss /= len(train_loader)
    train_loss_hist.append(train_loss)
    if e%10==0:
        torch.save(model.state_dict(), "state/model_e"+str(e)+".pth")
#     print(e, ": ", total_loss) 
    


    valid_loss = 0 
    # valid_acc = 0
    with torch.inference_mode():
        for x,y in tqdm(valid_loader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            valid_loss += loss.item()
    # 		valid_acc += (np.sum(np.array(target.argmax(dim=1).tolist()) == np.array(y.tolist())))/2000
    valid_loss /= len(valid_loader)
    valid_loss_hist.append(valid_loss)

    print(e, ": " , train_loss, ", ", valid_loss)
    
    trainhist = np.array(train_loss_hist)
    validhist = np.array(valid_loss_hist)
    np.save("train_loss_hist.npy", trainhist)
    np.save("valid_loss_hist.npy", validhist)
    # print(e , ": " , total_loss, ", ", train_acc,", ",valid_loss,", ", valid_acc)
image, ano = trainset[10]
imagein = image.unsqueeze(dim=1).to(device)
model = UNet().to('cuda')
fig, ax = plt.subplots(1,3)
ax[0].imshow(image[0], 'gray')
ax[0].set_title("Input Image")
ax[1].imshow(ano[0], 'gray')
ax[0].axis('off')
ax[1].axis('off')
ax[1].set_title("Anotation")
for i in range(1):
    model.load_state_dict(torch.load('state/model_e'+str(i*10)+'.pth'))
    ax[i+2].imshow(model(imagein).detach().cpu()[0][0], 'gray')
    ax[i+2].axis('off')
    ax[i+2].set_title("Output Epoch "+str(i*10))
plt.show()

fig, ax = plt.subplots(1,3)
for i in range(1, 4):
    model.load_state_dict(torch.load('state/model_e'+str(i*10)+'.pth'))
    ax[i-1].imshow(model(imagein).detach().cpu()[0][0], 'gray')
    ax[i-1].axis('off')
    ax[i-1].set_title("Output Epoch "+str(i*10))
plt.show()

fig, ax = plt.subplots(1,3)
for i in range(4, 7):
    model.load_state_dict(torch.load('state/model_e'+str(i*10)+'.pth'))
    ax[i-4].imshow(model(imagein).detach().cpu()[0][0], 'gray')
    ax[i-4].axis('off')
    ax[i-4].set_title("Output Epoch "+str(i*10))
plt.show()

fig, ax = plt.subplots(1,3)
for i in range(7, 10):
    model.load_state_dict(torch.load('state/model_e'+str(i*10)+'.pth'))
    ax[i-7].imshow(model(imagein).detach().cpu()[0][0], 'gray')
    ax[i-7].axis('off')
    ax[i-7].set_title("Output Epoch "+str(i*10))
plt.show()

model.load_state_dict(torch.load('state/model_e100.pth'))
plt.imshow(model(imagein).detach().cpu()[0][0],'gray')
plt.axis('off')
plt.title("Output Epoch 100")
plt.show()
