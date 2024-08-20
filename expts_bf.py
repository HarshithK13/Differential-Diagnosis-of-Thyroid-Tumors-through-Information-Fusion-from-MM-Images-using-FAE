import numpy as np
import pandas as pd
import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import cv2

from PIL import Image
import torchvision.transforms as T
import sklearn.metrics as metrics

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from os.path import exists




file_exists = exists('models/model_autoencoder_AEN.pt')
# print("file_exists:", file_exists)

res=[]
root_dir_fa = '/home/reddy16/a_data/FA/'
root_dir_ftc = '/home/reddy16/a_data/FTC/'

roi_dir_fa = os.listdir(root_dir_fa)
# print(roi_dir_fa)
roi_dir_fa_len = len(roi_dir_fa)
# print(roi_dir_fa_len)

roi_dir_ftc = os.listdir(root_dir_ftc)
# print(roi_dir_ftc)
roi_dir_ftc_len = len(roi_dir_ftc)
# print(roi_dir_ftc_len)

patient_fa = []
patient_ftc = []
patient_dir = []
FA = []
FTC = []

for dir in roi_dir_fa:
    patient_fa.append(dir.split("-")[0])

for dir in roi_dir_ftc:
    patient_ftc.append(dir.split("-")[0])

for item in patient_fa: 
    if item not in patient_dir: 
        patient_dir.append(item)

for item in patient_ftc: 
    if item not in patient_dir: 
        patient_dir.append(item)
# print(patient_dir)

import random
data = []
# train_data = random.sample(patient_dir, 20)

# train_data = ['541952','568151']


# train_data = ['541952', '541951', '542858', '568147', '494917', '693346', '540886', '494919', '494922',
#                 '568146', '693348', '568145', '540889', '494920', '568148','568149', '494921', '541938', '540888']


train_data = ['542871','542858','541938', '541937', '693345','568151','568149',
                '568148','568147','568146','568145','540889','540887','540886','494922',
                '494921','494920']

for x in patient_dir:
    if x not in train_data:
        data.append(x)

# valid_data = random.sample(data, 3)
# valid_data = ['541952', '541951', '494919', '693347', '693346']
# valid_data = ['494918','568151','541925']
test_data = ['541952', '541951', '494919', '693347', '693346', '540888']

# for x in patient_dir:
#     if (x not in train_data) and (x not in valid_data):
#         test_data.append(x)
    
# test_data = ['541930','541925','693348','494918','494917']
# test_data = ['693345', '540887', '693347','541930','541937','542871']
valid_data = ['541930','541925','693348','494918','494917']

# print("length of train_data",len(train_data))
# print("length of valid_data",len(valid_data))
# print("length of test_data",len(test_data))


# print("train_data",train_data)
# print("valid_data",valid_data)
# print("test_data",test_data)

import glob

train_data_roi = []
test_data_roi = []
valid_data_roi = []
FA=[]
FTC=[]

for file_list in train_data:
    dirfiles = glob.glob(f'/home/reddy16/a_data/FA/{str(file_list)}*')
    flag = 0
    if len(dirfiles)==0:
        flag=1        
        dirfiles = glob.glob(f'/home/reddy16/a_data/FTC/{str(file_list)}*')
    for k in dirfiles:
        
        file = k.rsplit('/', 1)[1]
        if flag==0:
            FA.append(file)
        else:
            FTC.append(file)
        train_data_roi.append(file)


for file_list in valid_data:
    dirfiles = glob.glob(f'//home/reddy16/a_data/FA/{str(file_list)}*')
    flag = 0

    if len(dirfiles)==0:
        flag=1
        dirfiles = glob.glob(f'/home/reddy16/a_data/FTC/{str(file_list)}*')
    for k in dirfiles:
        file = k.rsplit('/', 1)[1]
        if flag==0:
            FA.append(file)
        else:
            FTC.append(file)
        valid_data_roi.append(file)


for file_list in test_data:
    dirfiles = glob.glob(f'/home/reddy16/a_data/FA/{str(file_list)}*')
    flag=0

    if len(dirfiles)==0:
        flag=1
        dirfiles = glob.glob(f'/home/reddy16/a_data/FTC/{str(file_list)}*')
    for k in dirfiles:
        file = k.rsplit('/', 1)[1]
        if flag==0:
            FA.append(file)
        else:
            FTC.append(file)
        test_data_roi.append(file)

print(len(train_data_roi))
print(train_data_roi)

print(len(valid_data_roi))
print(valid_data_roi)

print(len(test_data_roi))
print(test_data_roi)

len(FA)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

image_degree=[20,40,60,80,100,120,140,160]
import glob

root_dir = ['FA','FTC']
bshg_images_train=[]
fshg_images_train=[]
# tpef_images_train=[]
class_labels_train = []

bshg_images_val=[]
fshg_images_val=[]
# tpef_images_val=[]
class_labels_val = []

bshg_images_test=[]
fshg_images_test=[]
# tpef_images_test=[]
class_labels_test = []




from PIL import Image



for folder in train_data_roi:
    # print("For ROI:", folder)
    if folder in FA:
        path = root_dir_fa
        lbl = 'FA'
    else:
        path = root_dir_ftc
        lbl = 'FTC'

    # bshg1 = Image.open(f'{path}{folder}/BSHG/BSHG_pol_lin_0.tif')
    # bshg1 = np.asarray(bshg1, dtype=np.float32)
    # print(bshg1.shape)

    if folder != '494920-3$':
        bshg1 = Image.open(f'{path}{folder}/BSHG/BSHG_pol_lin_0.tif')
        fshg1 = Image.open(f'{path}{folder}/FSHG/FSHG_pol_lin_0.tif')
        # tpef = Image.open(f'{path}{folder}/TPEF.tif')
        

        bshg1 = np.asarray(bshg1, dtype=np.float32)
        fshg1 = np.asarray(fshg1, dtype=np.float32)
        # tpef = np.asarray(tpef, dtype=np.float32)
        
        #normalization
#         print("max value before normalization: ", bshg1.max())
        # bshg1 = (bshg1 - bshg1.min()) * 255// (bshg1.max() - bshg1.min())
        # fshg1 = (fshg1 - fshg1.min()) * 255// (fshg1.max() - fshg1.min())
        # tpef = (tpef - tpef.min()) * 255// (tpef.max() - tpef.min())
        
        
        # bshg_list_train.append(bshg1)
        # fshg_list_train.append(fshg1)
        
        bshg=bshg1
        fshg = fshg1

        for degree in image_degree:
            bshg1 = Image.open(f'{path}{folder}/BSHG/BSHG_pol_lin_{degree}.tif')
            fshg1 = Image.open(f'{path}{folder}/FSHG/FSHG_pol_lin_{degree}.tif')
            bshg1 = np.asarray(bshg1, dtype=np.float32)
            fshg1 = np.asarray(fshg1, dtype=np.float32)
            
#             print("max value before normalization: ", bshg1.max())
            # bshg1 = (bshg1 - bshg1.min()) * 255// (bshg1.max() - bshg1.min())
            # fshg1 = (fshg1 - fshg1.min()) * 255// (fshg1.max() - fshg1.min())
            
            
            # bshg1 = torch.tensor(bshg1)
            # fshg1 = torch.tensor(fshg1)
            bshg = bshg + bshg1
            fshg = fshg + fshg1

            # bshg_list_train.append(bshg1)
            # fshg_list_train.append(fshg1)


        bshg = bshg/9
        fshg = fshg/9
#         print("\n Size of BSHG and FSHG later", bshg.shape,fshg.shape)

        # batch_bshg = np.stack(bshg_list_train)
        # bshg = np.mean(batch_bshg)
        
        # batch_fshg = np.stack(fshg_list_train)
        # fshg = np.mean(batch_fshg)

        bshg_images_train.append(bshg)
        fshg_images_train.append(fshg)
        # tpef_images_train.append(tpef)

        class_labels_train.append(lbl)

# print("Validation\n")
########## validation ###########


for folder in valid_data_roi:
    if folder in FA:
        path = root_dir_fa
        lbl = 'FA'
    else:
        path = root_dir_ftc
        lbl = 'FTC'
        
    if folder != '494920-3$':

        bshg1 = Image.open(f'{path}{folder}/BSHG/BSHG_pol_lin_0.tif')
        fshg1 = Image.open(f'{path}{folder}/FSHG/FSHG_pol_lin_0.tif')
        # tpef = Image.open(f'{path}{folder}/TPEF.tif')

        bshg1 = np.asarray(bshg1, dtype=np.float32)
        fshg1 = np.asarray(fshg1, dtype=np.float32)
        # tpef = np.asarray(tpef, dtype=np.float32)
        
        # bshg1 = (bshg1 - bshg1.min()) * 255// (bshg1.max() - bshg1.min())
        # fshg1 = (fshg1 - fshg1.min()) * 255// (fshg1.max() - fshg1.min())
        # tpef = (tpef - tpef.min()) * 255// (tpef.max() - tpef.min())
        
        
        
        # bshg1 = torch.tensor(bshg1)
        # fshg1 = torch.tensor(fshg1)
        
        # bshg_list_val.append(bshg1)
        # fshg_list_val.append(fshg1)
        
        bshg = bshg1
        fshg = fshg1

        # print("\n Size of BSHG and FSHG initially", bshg.size(),fshg.size())

        for degree in image_degree:
            bshg1 = Image.open(f'{path}{folder}/BSHG/BSHG_pol_lin_{degree}.tif')
#             print("bshg1 data",bshg1)
            fshg1 = Image.open(f'{path}{folder}/FSHG/FSHG_pol_lin_{degree}.tif')
            bshg1 = np.asarray(bshg1, dtype=np.float32)
            fshg1 = np.asarray(fshg1, dtype=np.float32)
            
            # bshg1 = (bshg1 - bshg1.min()) * 255// (bshg1.max() - bshg1.min())
            # fshg1 = (fshg1 - fshg1.min()) * 255// (fshg1.max() - fshg1.min())
            
            
            
            # bshg1 = torch.tensor(bshg1)
            # fshg1 = torch.tensor(fshg1)
            bshg = bshg + bshg1
            fshg = fshg + fshg1
            # bshg_list_val.append(bshg1)
            # fshg_list_val.append(fshg1)

        bshg = bshg/9
        fshg = fshg/9

        # batch_bshg = np.stack(bshg_list_val)
        # bshg = np.mean(batch_bshg)
        
        # batch_fshg = np.stack(fshg_list_val)
        # fshg = np.mean(batch_fshg)

        bshg_images_val.append(bshg)
        fshg_images_val.append(fshg)
        # tpef_images_val.append(tpef)

        class_labels_val.append(lbl)


######## Test Data ###############

for folder in test_data_roi:
    # print("For ROI:", folder)
    if folder in FA:
        path = root_dir_fa
        lbl = 'FA'
    else:
        path = root_dir_ftc
        lbl = 'FTC'

    if folder != '494920-3$':
        bshg1 = Image.open(f'{path}{folder}/BSHG/BSHG_pol_lin_0.tif')
        fshg1 = Image.open(f'{path}{folder}/FSHG/FSHG_pol_lin_0.tif')
        # tpef = Image.open(f'{path}{folder}/TPEF.tif')

        bshg1 = np.asarray(bshg1, dtype=np.float32)
        fshg1 = np.asarray(fshg1, dtype=np.float32)
        # tpef = np.asarray(tpef, dtype=np.float32)
        
        # bshg1 = (bshg1 - bshg1.min()) * 255// (bshg1.max() - bshg1.min())
        # fshg1 = (fshg1 - fshg1.min()) * 255// (fshg1.max() - fshg1.min())
        # tpef = (tpef - tpef.min()) * 255// (tpef.max() - tpef.min())
        
        
        
        # bshg1 = torch.tensor(bshg1)
        # fshg1 = torch.tensor(fshg1)
        
        # bshg_list_test.append(bshg1)
        # fshg_list_test.append(fshg1)
        
        bshg = bshg1
        fshg = fshg1

        # print("\n Size of BSHG and FSHG initially", bshg.size(),fshg.size())

        for degree in image_degree:
            bshg1 = Image.open(f'{path}{folder}/BSHG/BSHG_pol_lin_{degree}.tif')
            fshg1 = Image.open(f'{path}{folder}/FSHG/FSHG_pol_lin_{degree}.tif')
            bshg1 = np.asarray(bshg1, dtype=np.float32)
            fshg1 = np.asarray(fshg1, dtype=np.float32)
            
            # bshg1 = (bshg1 - bshg1.min()) * 255// (bshg1.max() - bshg1.min())
            # fshg1 = (fshg1 - fshg1.min()) * 255// (fshg1.max() - fshg1.min())
            
            
        
            # bshg1 = torch.tensor(bshg1)
            # fshg1 = torch.tensor(fshg1)
            bshg = bshg + bshg1
            fshg = fshg + fshg1
            
            # bshg_list_test.append(bshg1)
            # fshg_list_test.append(fshg1)
            

        bshg = bshg/9
        fshg = fshg/9

        # batch_bshg = np.stack(bshg_list_test)
        # bshg = np.mean(batch_bshg)
        
        # batch_fshg = np.stack(fshg_list_test)
        # fshg = np.mean(batch_fshg)

        bshg_images_test.append(bshg)
        fshg_images_test.append(fshg)
        # tpef_images_test.append(tpef)

        class_labels_test.append(lbl)

print("\nAUTOENCODER START")





import torch.nn as nn
import torch.nn.functional as F

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 1, 3)
        
        self.convD = nn.Conv2d(1, 4, 3, dilation=4)
        self.convA = nn.Conv2d(1, 2, 3)
        self.convB = nn.Conv2d(2, 3, 3)
        self.convC = nn.Conv2d(3, 4, 3)
        
        self.conv1 = nn.Conv2d(8, 14, 3)
        self.conv2 = nn.Conv2d(14, 16, 3)
        self.conv3 = nn.Conv2d(16, 20, 3)
        self.conv4 = nn.Conv2d(20, 20, 3)
        self.conv5 = nn.Conv2d(20, 32, 3)        
        self.conv6 = nn.Conv2d(32, 64, 3)        
        self.conv7 = nn.Conv2d(64, 128, 3)        
        self.conv8 = nn.Conv2d(128, 128, 3)
        # self.conv_new2 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv9 = nn.Conv2d(128, 256, 3)
        self.conv10 = nn.Conv2d(256, 512, 3)
        self.conv11 = nn.Conv2d(512, 512, 3)
        # self.conv_new = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv_latent = nn.Conv2d(512, 3, 3)

        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn3 = nn.BatchNorm2d(3)
        self.bnt0 = nn.BatchNorm2d(256)
        self.bnt1 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn_new = nn.BatchNorm2d(20)
        self.bn_new2 = nn.BatchNorm2d(14)
        self.bn_new3 = nn.BatchNorm2d(16)
        self.bn_new4 = nn.BatchNorm2d(20)
        self.bn_new5 = nn.BatchNorm2d(64)


        self.pool = nn.MaxPool2d(2, 2)
        self.t_conv1 = nn.ConvTranspose2d(512, 256, 3)         # added layer 2
        self.t_conv2 = nn.ConvTranspose2d(256, 128, 3)
        self.t_conv3 = nn.ConvTranspose2d(128, 64, 3)          
        self.t_conv4 = nn.ConvTranspose2d(64, 32, 3)
        self.t_conv5 = nn.ConvTranspose2d(32, 24, 3)
        self.t_conv6 = nn.ConvTranspose2d(24, 12, 3)
        self.t_conv7 = nn.ConvTranspose2d(6, 4, 3)
        self.t_conv8 = nn.ConvTranspose2d(4, 1, 3)
        self.t_conv9 = nn.ConvTranspose2d(1, 1, 3)
        # self.t_conv_new = nn.ConvTranspose2d(1, 1, 3, padding = 1)
        self.t_conv10 = nn.ConvTranspose2d(1, 1, 3, stride=2)
        self.t_conv11 = nn.ConvTranspose2d(1, 1, 3, stride = 2, padding=2, output_padding = 1)
        self.t_conv12 = nn.ConvTranspose2d(1, 1, 3, padding=1)

    def forward(self, b,f):
        
        stack_img = np.zeros(shape=(1,8,502,502), dtype=np.float32)
        stack_img = torch.tensor(stack_img).to(device)
        

        b1 = F.gelu(self.bn0(self.conv0(b)))
        f1 = F.gelu(self.bn0(self.conv0(f)))
        # t1 = F.gelu(self.conv0(t))         
        
        b1 = F.gelu(self.convD(b1))
        f1 = F.gelu(self.convD(f1))
        # t1 = F.gelu(self.conv0(t1))
        
        # t1 = F.gelu(self.convA(t1))
        # t1 = F.gelu(self.convB(t1))
        # t1 = F.gelu(self.convC(t1))
        
        
        stack_img[:,0:4,:,:] = b1
        stack_img[:,4:8,:,:] = f1
        # stack_img[:,8:12,:,:] = t1
        
        x = stack_img
        
        x = F.gelu(self.conv1(x))
        x = self.pool(x)    
        x = F.gelu(self.conv2(x))              # added layer
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))             # added layer 2
        
        x = F.gelu(self.conv6(x))
        x = self.pool(x)
        x = F.gelu(self.conv7(x))
        x = F.gelu(self.conv8(x))    # added layer
        # x = F.gelu(self.conv_new2(x))
        x = F.gelu(self.conv9(x))
        x = F.gelu(self.conv10(x))
        x = F.gelu(self.conv11(x))
        # x = F.gelu(self.conv_new(x))

        y = F.gelu(self.conv_latent(x))

        x = F.gelu(self.t_conv1(x))
        x = F.gelu(self.t_conv2(x))           # added layer
        x = F.gelu(self.t_conv3(x))
        x = F.gelu(self.t_conv4(x))
        x = F.gelu(self.t_conv5(x))
        x = F.gelu(self.t_conv6(x))        # added layer

        x1, x2= x[:,0:6,:,:], x[:,6:12,:,:]

        x1 = F.gelu(self.t_conv7(x1))        
        x2 = F.gelu(self.t_conv7(x2))
        # x3 = F.gelu(self.t_conv7(x3))

        x1 = F.gelu(self.t_conv8(x1))
        x2 = F.gelu(self.t_conv8(x2))
        # x3 = F.gelu(self.t_conv8(x3))

        x1 = F.gelu(self.t_conv9(x1))
        x2 = F.gelu(self.t_conv9(x2))
        # x3 = F.gelu(self.t_conv9(x3))
        
        # x1 = F.gelu(self.t_conv_new(x1))
        # x2 = F.gelu(self.t_conv_new(x2))
        # x3 = F.gelu(self.t_conv_new(x3))

        x1 = F.gelu(self.t_conv10(x1))
        x2 = F.gelu(self.t_conv10(x2))
        # x3 = F.gelu(self.t_conv10(x3))

        x1 = F.gelu(self.t_conv11(x1))
        x2 = F.gelu(self.t_conv11(x2))
        # x3 = F.gelu(self.t_conv11(x3))

        # x3 = F.gelu(self.t_conv12(x3))
        # x3 = F.gelu(self.t_conv12(x3))
        # x3 = F.gelu(self.t_conv12(x3))

        # print("shape of x3:", x3.shape)

        return x1, x2, y

model = AE()


loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# model.apply(weights_init_uniform)
model=model.to(device)


epochs = 30
output_loss_train = []
train_loss_bshg = []
train_loss_fshg = []
# train_loss_tpef = []

losses = []
output_loss_val = []
val_loss_bshg = []
val_loss_fshg = []
# val_loss_tpef = []
transform = T.ToPILImage()


best_val_loss = float('inf')
best_model_weights = None



if file_exists==False:
    for epoch in range(epochs):
        running_loss = 0.0
        loss_bshg, loss_fshg= 0.0,0.0
        for bshg,fshg,img_folder, class_label in zip(bshg_images_train, fshg_images_train,train_data_roi, class_labels_train):
        
            # bshg = bshg//255.0
            bshg = torch.from_numpy(bshg)
            bshg = bshg.to(device)
            bshg = bshg.unsqueeze(0)
            bshg = bshg.unsqueeze(1)
            
            
            # fshg = fshg//255.0
            fshg = torch.from_numpy(fshg)
            fshg = fshg.to(device)
            fshg = fshg.unsqueeze(0)
            fshg = fshg.unsqueeze(1)
            
            # tp = tp//255.0
            # tp = torch.from_numpy(tp)
            # tp = tp.to(device)
            # tp = tp.unsqueeze(0)
            # tp = tp.unsqueeze(1)
            
#             print("shape1 of bshg:",bshg.shape)
#             print("shape1 of fshg:",fshg.shape)
#             print("shape1 of tpef:",tp.shape)
            
            re_bshg,re_fshg,latent_image = model(bshg,fshg)

            # print("max values of bshg and bshg_re: ", bshg.max(), re_bshg.max())

#             print("shape1 of re_bshg:",re_bshg.shape)
#             print("shape1 of re_fshg:",re_fshg.shape)
#             print("shape1 of re_tpef:",re_tp.shape)

            loss1 = loss_function(re_bshg, bshg)
            loss2 = loss_function(re_fshg, fshg)
            # loss3 = loss_function(re_tp, tp)

            loss = loss1 + 0.5*loss2 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loss_bshg += loss1.item() 
            loss_fshg += loss2.item() 
            # loss_tpef += loss3.item()

            if epoch==(epochs-1):
                I1 = latent_image
                I1 = torch.squeeze(I1,0)
                I1 = transform(I1)
                I1= np.asarray(I1)
                result=cv2.imwrite(f'/home/reddy16/datafolder_AEN/resultant_images/trainData/{class_label}/{img_folder}_img.png', I1)
                img = Image.open(f'/home/reddy16/datafolder_AEN/resultant_images/trainData/{class_label}/{img_folder}_img.png')
                resized_img = img.resize((126, 126))
                resized_img.save(f'/home/reddy16/datafolder_AEN/resultant_images/trainData/{class_label}/{img_folder}_img.png')

                # I1 = re_bshg
                # I1 = torch.squeeze(I1,0)
                # I1 = transform(I1)
                # I1= np.asarray(I1)
                # result=cv2.imwrite(f'/home/reddy16/datafolder_AEN/BSHG_images/trainData/{class_label}/{img_folder}_img.jpg', I1)
                # img = Image.open(f'/home/reddy16/datafolder_AEN/BSHG_images/trainData/{class_label}/{img_folder}_img.jpg')
                # resized_img = img.resize((224, 224))
                # resized_img.save(f'/home/reddy16/datafolder_AEN/BSHG_images/trainData/{class_label}/{img_folder}_img.jpg')

                # I1 = re_fshg
                # I1 = torch.squeeze(I1,0)
                # I1 = transform(I1)
                # I1= np.asarray(I1)
                # result=cv2.imwrite(f'/home/reddy16/datafolder_AEN/FSHG_images/trainData/{class_label}/{img_folder}_img.jpg', I1)
                # img = Image.open(f'/home/reddy16/datafolder_AEN/FSHG_images/trainData/{class_label}/{img_folder}_img.jpg')
                # resized_img = img.resize((224, 224))
                # resized_img.save(f'/home/reddy16/datafolder_AEN/FSHG_images/trainData/{class_label}/{img_folder}_img.jpg')

                # I1 = re_tp
                # I1 = torch.squeeze(I1,0)
                # I1 = transform(I1)
                # I1= np.asarray(I1)
                # result=cv2.imwrite(f'/home/reddy16/datafolder_AEN/TPEF_images/trainData/{class_label}/{img_folder}_img.jpg', I1)
                # img = Image.open(f'/home/reddy16/datafolder_AEN/TPEF_images/trainData/{class_label}/{img_folder}_img.jpg')
                # resized_img = img.resize((224, 224))
                # resized_img.save(f'/home/reddy16/datafolder_AEN/TPEF_images/trainData/{class_label}/{img_folder}_img.jpg')
            

        loss = running_loss / len(train_data_roi)
        loss_b = loss_bshg / len(train_data_roi)
        loss_f = loss_fshg / len(train_data_roi)
        # loss_t = loss_tpef / len(train_data_roi)

        train_loss_bshg.append(loss_b)
        train_loss_fshg.append(loss_f)
        # train_loss_tpef.append(loss_t)

        output_loss_train.append(loss)
        print('\n\nEpoch {} of {}, Train Loss: {:.3f}'.format(epoch+1, epochs, loss))

        print("\n\nValidation loss:\n")



        
        losses = []

        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            loss_bshg, loss_fshg = 0.0,0.0
            for bshg, fshg, img_folder, class_label in zip(bshg_images_val, fshg_images_val,valid_data_roi, class_labels_val):
                
                # bshg = bshg//255.0
                bshg = torch.from_numpy(bshg)
                bshg = bshg.to(device)
                bshg = bshg.unsqueeze(0)
                bshg = bshg.unsqueeze(1)


                # fshg = fshg//255.0
                fshg = torch.from_numpy(fshg)
                fshg = fshg.to(device)
                fshg = fshg.unsqueeze(0)
                fshg = fshg.unsqueeze(1)

                # tp = tp//255.0
                # tp = torch.from_numpy(tp)
                # tp = tp.to(device)
                # tp = tp.unsqueeze(0)
                # tp = tp.unsqueeze(1)

#                 print("shape1 of bshg:",bshg.shape)
#                 print("shape1 of fshg:",fshg.shape)
#                 print("shape1 of tpef:",tp.shape)

                re_bshg,re_fshg,latent_image = model(bshg,fshg)

#                 print("shape1 of re_bshg:",re_bshg.shape)
#                 print("shape1 of re_fshg:",re_fshg.shape)
#                 print("shape1 of re_tpef:",re_tp.shape)

                loss1 = loss_function(re_bshg, bshg)
                loss2 = loss_function(re_fshg, fshg)
                # loss3 = loss_function(re_tp, tp)

                loss = loss1 + 0.5*loss2 
                
                running_loss += loss.item()
                loss_bshg += loss1.item() 
                loss_fshg += loss2.item() 
                # loss_tpef += loss3.item()

                if epoch==(epochs-1):
                    I1 = latent_image
                    I1 = torch.squeeze(I1,0)
                    I1 = transform(I1)
                    I1= np.asarray(I1)
                    result=cv2.imwrite(f'/home/reddy16/datafolder_AEN/resultant_images/valData/{class_label}/{img_folder}_img.png', I1)
                    img = Image.open(f'/home/reddy16/datafolder_AEN/resultant_images/valData/{class_label}/{img_folder}_img.png')
                    resized_img = img.resize((126, 126))
                    resized_img.save(f'/home/reddy16/datafolder_AEN/resultant_images/valData/{class_label}/{img_folder}_img.png')

                    # I1 = re_bshg
                    # I1 = torch.squeeze(I1,0)
                    # I1 = transform(I1)
                    # I1= np.asarray(I1)
                    # result=cv2.imwrite(f'/home/reddy16/datafolder_AEN/BSHG_images/valData/{class_label}/{img_folder}_img.jpg', I1)
                    # img = Image.open(f'/home/reddy16/datafolder_AEN/BSHG_images/valData/{class_label}/{img_folder}_img.jpg')
                    # resized_img = img.resize((224, 224))
                    # resized_img.save(f'/home/reddy16/datafolder_AEN/BSHG_images/valData/{class_label}/{img_folder}_img.jpg')

                    # I1 = re_fshg
                    # I1 = torch.squeeze(I1,0)
                    # I1 = transform(I1)
                    # I1= np.asarray(I1)
                    # result=cv2.imwrite(f'/home/reddy16/datafolder_AEN/FSHG_images/valData/{class_label}/{img_folder}_img.jpg', I1)
                    # img = Image.open(f'/home/reddy16/datafolder_AEN/FSHG_images/valData/{class_label}/{img_folder}_img.jpg')
                    # resized_img = img.resize((224, 224))
                    # resized_img.save(f'/home/reddy16/datafolder_AEN/FSHG_images/valData/{class_label}/{img_folder}_img.jpg')

                    # I1 = re_tp
                    # I1 = torch.squeeze(I1,0)
                    # I1 = transform(I1)
                    # I1= np.asarray(I1)
                    # result=cv2.imwrite(f'/home/reddy16/datafolder_AEN/TPEF_images/valData/{class_label}/{img_folder}_img.jpg', I1)
                    # img = Image.open(f'/home/reddy16/datafolder_AEN/TPEF_images/valData/{class_label}/{img_folder}_img.jpg')
                    # resized_img = img.resize((224, 224))
                    # resized_img.save(f'/home/reddy16/datafolder_AEN/TPEF_images/valData/{class_label}/{img_folder}_img.jpg')



            loss = running_loss / len(valid_data_roi)
            loss_b = loss_bshg / len(valid_data_roi)
            loss_f = loss_fshg / len(valid_data_roi)
            # loss_t = loss_tpef / len(valid_data_roi)

            val_loss_bshg.append(loss_b)
            val_loss_fshg.append(loss_f)
            # val_loss_tpef.append(loss_t)

            avg_val_loss = loss
            output_loss_val.append(loss)
            print('Epoch {} of {}, Val Loss: {:.3f}'.format(epoch+1, epochs, loss))
            print("bshg val loss:", loss_b)
            print("fshg val loss:", loss_f)
            # print("tpef val loss:", loss_t)


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict()
            best_model_optimizer = optimizer.state_dict()
            # Save the model checkpoint
            torch.save({'model_state_dict':best_model_weights,'optimizer_state_dict':best_model_optimizer}, '/home/reddy16/models/model_autoencoder_AEN.pt')
            print("Model saved successfully")

checkpoint = torch.load('/home/reddy16/models/model_autoencoder_AEN.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def draw_result_AE(epochs, lst_loss_train, lst_loss_val):
    plt.plot(epochs, lst_loss_train, '-b', label='train_loss')
    plt.plot(epochs, lst_loss_val, '-r', label='val_loss')

    plt.xlabel("Loss values")
    plt.legend(loc='upper left')
    title = "loss_values_AE"
    # save image
    plt.savefig(title+".png")  # should before show method

    # show
    plt.show()
    
epochs =[]   
for i in range(30):
    epochs.append(i)
# print(epochs)
# print(len(output_loss_train), len(output_loss_val))
# draw_result_AE(epochs, output_loss_train, output_loss_val)

model.eval()
### testing data ###

output_loss_test = []
with torch.no_grad():
    for bshg, fshg, img_folder, class_label in zip(bshg_images_test, fshg_images_test, test_data_roi, class_labels_test):
        
        # bshg = bshg//255.0
        bshg = torch.from_numpy(bshg)
        bshg = bshg.to(device)
        bshg = bshg.unsqueeze(0)
        bshg = bshg.unsqueeze(1)


        # fshg = fshg//255.0
        fshg = torch.from_numpy(fshg)
        fshg = fshg.to(device)
        fshg = fshg.unsqueeze(0)
        fshg = fshg.unsqueeze(1)

        # tp = tp//255.0
        # tp = torch.from_numpy(tp)
        # tp = tp.to(device)
        # tp = tp.unsqueeze(0)
        # tp = tp.unsqueeze(1)


        re_bshg,re_fshg,latent_image = model(bshg,fshg)




        I1 = latent_image
        I1 = torch.squeeze(I1,0)

        I1 = transform(I1)
        I1= np.asarray(I1)
        
        result=cv2.imwrite(f'/home/reddy16/datafolder_AEN/resultant_images/testData/{class_label}/{img_folder}_img.png', I1)
        img = Image.open(f'/home/reddy16/datafolder_AEN/resultant_images/testData/{class_label}/{img_folder}_img.png')
        resized_img = img.resize((256, 256))
        resized_img.save(f'/home/reddy16/datafolder_AEN/resultant_images/testData/{class_label}/{img_folder}_img.png')

        I1 = re_bshg
        I1 = torch.squeeze(I1,0)
        I1 = transform(I1)
        I1= np.asarray(I1)
        result=cv2.imwrite(f'/home/reddy16/datafolder_AEN/BSHG_images/testData/{class_label}/{img_folder}_img.jpg', I1)

        I1 = re_fshg
        I1 = torch.squeeze(I1,0)
        I1 = transform(I1)
        I1= np.asarray(I1)
        result=cv2.imwrite(f'/home/reddy16/datafolder_AEN/FSHG_images/testData/{class_label}/{img_folder}_img.jpg', I1)

        
        # I1 = re_tp
        # I1 = torch.squeeze(I1,0)
        # I1 = transform(I1)
        # I1= np.asarray(I1)
        # result=cv2.imwrite(f'/home/reddy16/datafolder_AEN/TPEF_images/testData/{class_label}/{img_folder}_img.jpg', I1)

print("\nAUTOENCODER END")

# #def draw_result_AE(epochs, lst_loss_train, lst_loss_val):
#     plt.plot(epochs, lst_loss_train, '-b', label='train_loss')
#     plt.plot(epochs, lst_loss_val, '-r', label='val_loss')

#     plt.xlabel("Loss values")
#     plt.legend(loc='upper left')
#     title = "loss_values_AE"
#     # save image
#     plt.savefig(title+".png")  # should before show method

#     # show
#     plt.show()
    
# epochs =[]   
# for i in range(2):
#     epochs.append(i)
# print(epochs)
print(len(output_loss_train), len(output_loss_val))
#draw_result_AE(epochs, output_loss_train, output_loss_val)

transforms_train = transforms.Compose([  
    transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transforms_test = transforms.Compose([
    transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transforms_val = transforms.Compose([
    
    transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_dir = "/home/reddy16/datafolder_AEN/resultant_images/trainData/"
val_dir = "/home/reddy16/datafolder_AEN/resultant_images/valData/"
test_dir = "/home/reddy16/datafolder_AEN/resultant_images/testData/"


train_classa_dir = "/home/reddy16/datafolder_AEN/resultant_images/trainData/FA/"
train_classb_dir = "/home/reddy16/datafolder_AEN/resultant_images/trainData/FTC/"

test_classa_dir = "/home/reddy16/datafolder_AEN/resultant_images/testData/FA/"
test_classb_dir = "/home/reddy16/datafolder_AEN/resultant_images/testData/FTC/"

val_classa_dir = "/home/reddy16/datafolder_AEN/resultant_images/valData/FA/"
val_classb_dir = "/home/reddy16/datafolder_AEN/resultant_images/valData/FTC/"




from torchvision import datasets

train_dataset = datasets.ImageFolder(train_dir, transforms_train)
print(train_dataset)
test_dataset = datasets.ImageFolder(test_dir, transforms_test)
val_dataset = datasets.ImageFolder(val_dir, transforms_val)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))
class_names = train_dataset.classes
print('Class names:', class_names)

class_names_test = test_dataset.classes
print('Class names Test:', class_names_test)


train_features, train_labels = next(iter(train_dataloader))
print("\n Train class labels:", train_labels)

count_0 = 0
count_1 = 0

for i in train_labels:
    if i==0:
        count_0 = count_0 + 1
    else:
        count_1 = count_1 + 1

print("\n Count of FA: ", count_0)
print("\n Count of FTC: ", count_1)


test_features, test_labels = next(iter(test_dataloader))
print("\n Test class labels:", test_labels)

count_0 = 0
count_1 = 0

for i in test_labels:
    if i==0:
        count_0 = count_0 + 1
    else:
        count_1 = count_1 + 1

print("\n Count of FA: ", count_0)
print("\n Count of FTC: ", count_1)

# !pip install --upgrade efficientnet-pytorch

from torchvision import models
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

target = [0,1]


def multiclass_roc_auc_score(y_test, y_pred, average=None):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def weights_init_uniform(m):
    classname = m.__class__.__name__
    class_list = ['Conv2d','Linear']
    if classname in class_list:
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='sigmoid')
        m.weight.data.uniform_(0.0001, 0.001)

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
# model = AE()
for param in model.parameters():
    param.requires_grad = True


num_features=model._fc.in_features
model._fc = nn.Sequential(
    nn.Linear(num_features, 2),nn.Sigmoid())

model.apply(weights_init_uniform)
model = model.to(device) 
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

import time
import tqdm
from tqdm import tqdm
num_epochs = 40

best_val_loss = float('inf')
best_model_weights = None

for epoch in range(num_epochs):
    total_correct = 0.0
    running_loss = 0.0

    for inputs, vals in train_dataloader:
        
        ground_truth = np.zeros((inputs.size(0),2))
        for i in range(len(vals)):
            ground_truth[i,vals[i].item()]=1


        # inputs= inputs.unsqueeze(0)
        inputs = inputs.float().to(device)
        vals = vals.to(device)
        ground_truth = torch.from_numpy(ground_truth)
        ground_truth = ground_truth.float().to(device)
        # vals = vals.softmax(dim=0)
        outputs = model(inputs)
        optimizer.zero_grad()
        
        loss = criterion(outputs, ground_truth)
        loss.backward()
        optimizer.step()

        # _, output_idx = torch.max(outputs, 1)
        output_idx = torch.argmax(outputs, dim=1)
        # print("vals:",vals)
        # print("output_idx:", output_idx)
        total_correct += (vals == output_idx).sum().item()
        running_loss += loss.item() * inputs.size(0)

    print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% '.format(epoch, running_loss/len(train_dataset), (total_correct/len(train_dataset))*100))
    
    preds, total_vals = [],[]
    diff = []
    label_val = []
    model.eval()
    with torch.no_grad():
        
        total_loss = 0.0
        total_correct = 0.0
        for inputs, vals in val_dataloader:

            ground_truth = np.zeros((inputs.size(0),2))
            for i in range(len(vals)):
                ground_truth[i,vals[i].item()]=1

            inputs = inputs.float().to(device)
            vals = vals.to(device)
            ground_truth = torch.from_numpy(ground_truth)
            ground_truth = ground_truth.float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, ground_truth)
            total_loss += loss.item() * inputs.size(0)

            # _, output_idx = torch.max(outputs, 1)
            output_idx = torch.argmax(outputs, dim=1)
            total_correct += sum(vals==output_idx)
            preds.append(output_idx.tolist())
            total_vals.append(vals.tolist())

            avg_val_loss = total_loss/len(val_dataset)
        
        print('[Validation #] Loss: {:.4f} Acc: {:.4f}% '.format(total_loss/len(val_dataset), (total_correct/len(val_dataset))*100))
        

        preds, total_vals = preds[0], total_vals[0]
        print("\n Preds: ", preds)
        print("\n Len preds: ", len(preds))
        print("\n Ground truth: ", total_vals)
        print("\n Len ground truth: ", len(total_vals))

        class_report = classification_report(total_vals, preds, labels=[0,1])
        print("\n Class_report: \n", class_report)
        f1_value = f1_score(total_vals, preds, average=None, labels=[0,1])
        precision = precision_score(total_vals, preds, average=None, labels=[0,1])
        recall = recall_score(total_vals, preds, average=None, labels=[0,1])

        conf_matrix = metrics.confusion_matrix(total_vals, preds)
        print("\n Val Precision: {}\n Val Recall: {}\n Val F1_score: {}\n ".format(precision,recall,f1_value))
        print("Confusion Matrix:", conf_matrix)

        for j in range(len(preds)):
            if preds[j]!=total_vals[j]: 
                diff.append(j+1)
                label_val.append(total_vals[j])

        print("\n Difference is there at the following indices:", diff)
        print("\n No. of mismatches: ", len(diff))
        print("\n Corresponding mismatch ground truth: ", label_val)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = model.state_dict()
        # Save the model checkpoint
        torch.save(best_model_weights, 'models/model_AEN.pt')
        print("Model saved successfully")


    # scheduler.step()

print("\n Training completed")

model.load_state_dict(torch.load('/home/reddy16/models/model_AEN.pt'))

model.eval()

preds, total_vals = [],[]
diff = []
label_val = []
#####   TEST THE MODEL  ##########

with torch.no_grad():
    total_loss = 0.0
    total_correct = 0.0
    for inputs, vals in test_dataloader:
        inputs = inputs.to(device)
        vals = vals.to(device)
        inputs = inputs.float()
        outputs = model(inputs)
        # print("output of test data:", outputs)

        # _, output_idx = torch.max(outputs, 1)
        output_idx = torch.argmax(outputs, dim=1)
        # print("output_idx of test data:", output_idx)
        total_correct += sum(vals==output_idx)
        preds.append(output_idx.tolist())
        total_vals.append(vals.tolist())
    
    print('[Test #] Loss: {:.4f} Acc: {:.4f}% '.format(total_loss/len(test_dataset), (total_correct/len(test_dataset))*100))
    

    preds, total_vals = preds[0], total_vals[0]
    # print("\n Preds: ", preds)
    # print("\n Len preds: ", len(preds))
    # print("\n Ground truth: ", total_vals)
    # print("\n Len ground truth: ", len(total_vals))

    class_report = classification_report(total_vals, preds, labels=[0,1], zero_division=0)
    print("\n Class_report: \n", class_report)
    f1_value = f1_score(total_vals, preds, average=None, labels=[0,1], zero_division=0)
    precision = precision_score(total_vals, preds, average=None, labels=[0,1], zero_division=0)
    recall = recall_score(total_vals, preds, average=None, labels=[0,1], zero_division=0)

    print('ROC AUC score:', multiclass_roc_auc_score(total_vals, preds,average=None))
    roc_auc_score = metrics.roc_auc_score(total_vals, preds,average=None)
    print("\n Precision: {}\n Recall: {}\n F1_score: {}\n AUC_ROC_Score: {}".format(precision,recall,f1_value,roc_auc_score))


    for j in range(len(preds)):
        if preds[j]!=total_vals[j]:
            diff.append(j+1)
            label_val.append(total_vals[j])

    print("\n Difference is there at the following indices:", diff)
    print("\n No. of mismatches: ", len(diff))
    print("\n Corresponding mismatch ground truth: ", label_val)
