from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.svm import LinearSVC

plt.ion()   # interactive mode


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#data_dir = '/home/caglar/Dataset_Small'
data_dir = "C:\\Users\\sevda\\Desktop\\BBM406 Machine Learning\\Term Project\\Dataset_Small"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validation', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'validation', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using ", device )

print(class_names)
print(dataset_sizes)

loss_list = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    loss_list = []
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss_list.append(loss)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # torch.save(model , "final_model.pt")
    return model



def eval_model(model, criterion):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    
    test_batches = len(dataloaders["test"])
    print("Evaluating model")
    print('-' * 10)
    
    for i, data in enumerate(dataloaders["test"]):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        model.train(False)
        model.eval()
        
        inputs, labels = data
        
        if (device == torch.device('cuda:0')):
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        outputs = model(inputs)
       
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss_test += loss.item()
        acc_test += torch.sum(preds == labels.data).item()
        #print(i, " acc: ", torch.sum(preds == labels.data).item())

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache
        
    avg_loss = loss_test / dataset_sizes["test"]
    avg_acc = acc_test / dataset_sizes["test"]
    
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


criterion = nn.CrossEntropyLoss()

#------------------------------------------------------------------

def calculate_svm(model):
  
    model.eval()
    model.to(device)
    
    feat_train = []
    feat_classes_train = []
    feat_test = []
    feat_classes_test = []
    
    with torch.no_grad():
        for i, (inputs,classes) in enumerate(dataloaders['train']):
            inputs = inputs.to(device)
            outputs = model(inputs)
            feat_train.extend(outputs.cpu().numpy())
            feat_classes_train.extend(classes.cpu().numpy())
            
    with torch.no_grad():
        for i, (inputs,classes) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            outputs = model(inputs)
            feat_test.extend(outputs.cpu().numpy())
            feat_classes_test.extend(classes.cpu().numpy())
            
    clf_linear = LinearSVC(random_state=0, max_iter=1000)
    classifier = clf_linear.fit(feat_train, feat_classes_train)

    
    print('Accuracy : {:.2f}'.format(100*clf_linear.score(feat_test,feat_classes_test)))
    return (feat_train, feat_classes_train, feat_test, feat_classes_test, clf_linear)


def calculate_classbased_accuracies(feat_train, feat_classes_train, feat_test, feat_classes_test, clf):
    predictions = clf.predict(feat_test)
    test_number = dataset_sizes['test']/len(image_datasets["test"].classes)
    
    
    true_pred = np.zeros(len(image_datasets["test"].classes))
    for i in range(len(predictions)):
      if(predictions[i] == feat_classes_test[i]):
        true_pred[predictions[i]] = true_pred[predictions[i]] + 1
    for i in range(len(true_pred)):
      true_pred[i] = 100* true_pred[i]/test_number
      print("Class based accuracy for {} = {:.1f}%".format(class_names[i], true_pred[i]))
    


#------------------------------------------------------------------
print("Training VGG16 FROM SCRATCH....")

vgg16_model = models.vgg16(pretrained=False)  

for param in vgg16_model.features.parameters():
    param.requires_grad = False

num_ftrs = vgg16_model.classifier[6].in_features
vgg16_model.classifier[6] = nn.Linear(num_ftrs ,len(class_names))
    
#print(vgg16_model)
vgg16_model  =vgg16_model.to(device)

optimizer = torch.optim.Adam(vgg16_model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_conv = train_model(vgg16_model, criterion, optimizer,exp_lr_scheduler, num_epochs=20) 
torch.save(model_conv, "vggmodel.pt")

with open('losslist_vgg_scratch.txt', 'w') as f:
    for item in loss_list:
        f.write("%s\n" % item)

f.close()


#------------------------------------------------------------------

print("Training VGG16 TRANSFER LEARNING....")

vgg16_model_pretrained = models.vgg16(pretrained=True)  

for param in vgg16_model_pretrained.features.parameters():
    param.requires_grad = False

num_ftrs = vgg16_model_pretrained.classifier[6].in_features
vgg16_model_pretrained.classifier[6] = nn.Linear(num_ftrs ,len(class_names))
    
#print(vgg16_model_pretrained)
vgg16_model_pretrained  =vgg16_model_pretrained.to(device)

optimizer = torch.optim.Adam(vgg16_model_pretrained.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_conv_pretrained_vgg = train_model(vgg16_model_pretrained, criterion, optimizer,exp_lr_scheduler, num_epochs=20) 
torch.save(model_conv_pretrained_vgg, "vggmodel_pretrained.pt")

with open('losslist_vgg_pretrained.txt', 'w') as f:
    for item in loss_list:
        f.write("%s\n" % item)
f.close()
loss_list = []

print("Evaluating VGG16 TRANSFER LEARNING....")
eval_model(model_conv_pretrained_vgg, criterion)



#------------------------------------------------------------------

# RESNET

print("Training RESNET50 FROM SCRATCH....")
resnet_model = models.resnet50(pretrained=False)

num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, len(class_names))
resnet_model  = resnet_model.to(device)
# print(resnet_model)

optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_conv_resnet = train_model(resnet_model, criterion, optimizer,exp_lr_scheduler, num_epochs=15) 
torch.save(model_conv_resnet , "resnetmodel.pt")
with open('losslist_resnet_scratch.txt', 'w') as f:
    for item in loss_list:
        f.write("%s\n" % item)
f.close()


print("Evaluating RESNET50 FROM SCRATCH....")
eval_model(model_conv_resnet, criterion)



#------------------------------------------------------------------
print("Training RESNET50 TRANSFER LEARNING....")

resnet_model_pretrained = models.resnet50(pretrained=True)

num_ftrs = resnet_model_pretrained.fc.in_features
resnet_model_pretrained.fc = nn.Linear(num_ftrs, len(class_names))
resnet_model_pretrained  = resnet_model_pretrained.to(device)
# print(resnet_model_pretrained)

optimizer = torch.optim.Adam(resnet_model_pretrained.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_conv_resnet_pretrained = train_model(resnet_model_pretrained, criterion, optimizer,exp_lr_scheduler, num_epochs=15) 
torch.save(model_conv_resnet_pretrained , "resnetmodel_pretrained.pt")
with open('losslist_resnet_pretrained.txt', 'w') as f:
    for item in loss_list:
        f.write("%s\n" % item)

f.close()


print("Evaluating RESNET50 TRANSFER LEARNING....")

eval_model(model_conv_resnet_pretrained, criterion)



print('Class Based Accuracies :\n')

result_arrays = calculate_svm(model_conv_resnet_pretrained)
calculate_classbased_accuracies(result_arrays[0],result_arrays[1], result_arrays[2], result_arrays[3],result_arrays[4])