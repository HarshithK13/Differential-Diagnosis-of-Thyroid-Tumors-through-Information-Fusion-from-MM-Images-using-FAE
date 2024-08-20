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
                '568148','568147','568146','568145','540889','540888','540887','540886','494922',
                '494921','494920']

for x in patient_dir:
    if x not in train_data:
        data.append(x)

# valid_data = random.sample(data, 3)
# valid_data = ['541952', '541951', '494919', '693347', '693346']
# valid_data = ['494918','568151','541925']
valid_data = ['541952', '541951', '494919', '693347', '693346']

# for x in patient_dir:
#     if (x not in train_data) and (x not in valid_data):
#         test_data.append(x)
    
# test_data = ['541930','541925','693348','494918','494917']
# test_data = ['693345', '540887', '693347','541930','541937','542871']
test_data = ['541930','541925','693348','494918','494917']

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
tpef_images_train=[]
class_labels_train = []

bshg_images_val=[]
fshg_images_val=[]
tpef_images_val=[]
class_labels_val = []

bshg_images_test=[]
fshg_images_test=[]
tpef_images_test=[]
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
        tpef = Image.open(f'{path}{folder}/TPEF.tif')

        bshg1 = np.asarray(bshg1, dtype=np.float32)[:, :, np.newaxis]
        fshg1 = np.asarray(fshg1, dtype=np.float32)[:, :, np.newaxis]
        tpef = np.asarray(tpef, dtype=np.float32)[:, :, np.newaxis]
        

        bshg1 = np.asarray(bshg1, dtype=np.float32)
        fshg1 = np.asarray(fshg1, dtype=np.float32)
        tpef = np.asarray(tpef, dtype=np.float32)
        
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

            bshg1 = np.asarray(bshg1, dtype=np.float32)[:, :, np.newaxis]
            fshg1 = np.asarray(fshg1, dtype=np.float32)[:, :, np.newaxis]
            tpef = np.asarray(tpef, dtype=np.float32)[:, :, np.newaxis]
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
        # print(bshg.shape)
        bshg_images_train.append(bshg)
        fshg_images_train.append(fshg)
        tpef_images_train.append(tpef)

        class_labels_train.append(lbl)

# print(class_labels_train)

parent_directory = '/home/reddy16/FSHG_EB4'
for fshg_image, label in zip(fshg_images_train, class_labels_train):
    # Convert the BSHG image to uint8 format and adjust the shape
    fshg_image = (fshg_image * 255).astype(np.uint8)
    fshg_image = np.reshape(fshg_image, (512, 512))
    # bshg_image = np.squeeze(bshg_image, axis=2)  # Remove the single-channel dimension
    
    # Define the destination directory based on the label
    destination_directory = os.path.join(parent_directory, label)
    
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)
    
    # Generate a unique filename for each BSHG image
    unique_filename = f"{label}_{len(os.listdir(destination_directory)) + 1}.png"
    
    # Save the BSHG image in the destination directory
    Image.fromarray(fshg_image).save(os.path.join(destination_directory, unique_filename))


# print("Validation\n")
########## validation ###########
# print(train_data_roi)

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
        tpef = Image.open(f'{path}{folder}/TPEF.tif')

        bshg1 = np.asarray(bshg1, dtype=np.float32)[:, :, np.newaxis]
        fshg1 = np.asarray(fshg1, dtype=np.float32)[:, :, np.newaxis]
        tpef = np.asarray(tpef, dtype=np.float32)[:, :, np.newaxis]

        bshg1 = np.asarray(bshg1, dtype=np.float32)
        fshg1 = np.asarray(fshg1, dtype=np.float32)
        tpef = np.asarray(tpef, dtype=np.float32)
        
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
            bshg1 = np.asarray(bshg1, dtype=np.float32)[:, :, np.newaxis]
            fshg1 = np.asarray(fshg1, dtype=np.float32)[:, :, np.newaxis]

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
        tpef_images_val.append(tpef)

        class_labels_val.append(lbl)

parent_directory2 = '/home/reddy16/FSHG_EB4_valid'
for fshg_image, label in zip(fshg_images_val, class_labels_val):
    # Convert the BSHG image to uint8 format and adjust the shape
    fshg_image = (fshg_image * 255).astype(np.uint8)
    fshg_image = np.reshape(fshg_image, (512, 512))
    # bshg_image = np.squeeze(bshg_image, axis=2)  # Remove the single-channel dimension
    
    # Define the destination directory based on the label
    destination_directory = os.path.join(parent_directory2, label)
    
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)
    
    # Generate a unique filename for each BSHG image
    unique_filename = f"{label}_{len(os.listdir(destination_directory)) + 1}.png"
    
    # Save the BSHG image in the destination directory
    Image.fromarray(fshg_image).save(os.path.join(destination_directory, unique_filename))

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
        tpef = Image.open(f'{path}{folder}/TPEF.tif')

        bshg1 = np.asarray(bshg1, dtype=np.float32)
        fshg1 = np.asarray(fshg1, dtype=np.float32)
        tpef = np.asarray(tpef, dtype=np.float32)
        
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
        tpef_images_test.append(tpef)

        class_labels_test.append(lbl)

parent_directory3 = '/home/reddy16/FSHG_EB4_test'
for fshg_image, label in zip(fshg_images_test, class_labels_test):
    # Convert the BSHG image to uint8 format and adjust the shape
    fshg_image = (fshg_image * 255).astype(np.uint8)
    fshg_image = np.reshape(fshg_image, (512, 512))
    # bshg_image = np.squeeze(bshg_image, axis=2)  # Remove the single-channel dimension
    
    # Define the destination directory based on the label
    destination_directory = os.path.join(parent_directory3, label)
    
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)
    
    # Generate a unique filename for each BSHG image
    unique_filename = f"{label}_{len(os.listdir(destination_directory)) + 1}.png"
    
    # Save the BSHG image in the destination directory
    Image.fromarray(fshg_image).save(os.path.join(destination_directory, unique_filename))

print("\nEFFICIENT NET B4 START")


transforms_train = transforms.Compose([  
    transforms.Resize(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transforms_test = transforms.Compose([
    transforms.Resize(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transforms_val = transforms.Compose([
    
    transforms.Resize(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


# !pip install --upgrade efficientnet-pytorch

from torchvision import models
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

target = [0,1]
# for bshg,img_folder, class_label in zip(bshg_images_train, train_data_roi, class_labels_train):



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

# from torchsummary import summary
# print(summary(model, (1, 512, 512))) 

num_features=model._fc.in_features
model._fc = nn.Sequential(
    nn.Linear(num_features, 1),nn.Sigmoid())
# model.conv_stem = nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1, bias=False)


model.apply(weights_init_uniform)
model = model.to(device) 
criterion = nn.BCELoss()
# from torchsummary import summary
# print(summary(model, (1, 512, 512)))
optimizer = optim.Adam(model.parameters(), lr=0.0001)

import time
import tqdm
from tqdm import tqdm

best_val_loss = float('inf')
best_model_weights = None

train_losses = []
valid_losses = []
train_accs = []
valid_accs = []

from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# Define a transform to resize the images and convert to grayscale
transform = transforms.Compose([
    transforms.ToPILImage(),                 # Convert to PIL Image
    transforms.Resize((224, 224)),          # Resize to 224x224
    transforms.Grayscale(num_output_channels=1),  # Convert to single-channel grayscale
    transforms.ToTensor()                    # Convert to a PyTorch tensor
])

# Convert your data into PyTorch tensors and create DataLoader for training, validation, and test sets
train_dir = "/home/reddy16/FSHG_EB4"
val_dir = "/home/reddy16/FSHG_EB4_valid"
test_dir = "/home/reddy16/FSHG_EB4_test"
train_dataset = datasets.ImageFolder(train_dir, transforms_train)
# print(train_dataset)
test_dataset = datasets.ImageFolder(test_dir, transforms_test)
val_dataset = datasets.ImageFolder(val_dir, transforms_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

for images, labels in train_loader:
    print(labels)
    break

# Define your training loop
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            # print(inputs[0].shape)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1).float())  # Assuming binary classification
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1).float())
                val_loss += loss.item()
                preds = (outputs >= 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(valid_loader)

        # Calculate evaluation metrics
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)

        print(f'Epoch [{epoch + 1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f} '
              f'Val Loss: {val_loss:.4f} ')

# Train the model
num_epochs = 40
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Test the model
model.eval()
test_loss = 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1).float())
        test_loss += loss.item()
        preds = (outputs >= 0.5).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader)

# Calculate evaluation metrics for the test set
f1 = f1_score(all_labels, all_preds, average=None, labels=[0,1], zero_division=0)
precision = precision_score(all_labels, all_preds, average=None, labels=[0,1], zero_division=0)
recall = recall_score(all_labels, all_preds, average=None, labels=[0,1], zero_division=0)
roc_auc = multiclass_roc_auc_score(all_labels, all_preds, average=None)

print(f'Test Loss: {test_loss:.4f}')
print("\n Precision: {}\n Recall: {}\n F1_score: {}\n AUC_ROC_Score: {}".format(precision,recall,f1,roc_auc))
diff = []
label_val = []

for j in range(len(all_preds)):
    if all_preds[j]!=all_labels[j]:
        diff.append(j+1)
        label_val.append(all_labels[j])

print(f'Number of mismatches: {len(diff)}')
