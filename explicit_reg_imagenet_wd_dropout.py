import os
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
  
def train_dataset(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    
    train_transforms = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        train_transforms
    )
    
    return train_dataset
  
def val_dataset(data_dir):
    val_dir = os.path.join(data_dir, 'val')
    
    val_transforms = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transforms
    )
    
    return val_dataset
  
def data_loader(data_dir, batch_size=32, workers=4, pin_memory=True):
    train_ds = train_dataset(data_dir)
    val_ds = val_dataset(data_dir)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    ) 
    return train_loader, val_loader

train_loader, val_loader = data_loader('/ssd_scratch/cvit/madhav/Imagenet/Imagenet-orig/')


def get_device():
  if torch.cuda.is_available():
      return torch.device('cuda')
  else:
      return torch.device('cpu')
device = get_device()

def eval_model(model,trainloader,testloader):
  model.eval()
  correct = 0
  total = 0
  for i, data in enumerate(testloader, 0):
    images, labels = data[0].to(device), data[1].to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, dim=1)
    total += labels.size(0)
    correct += (preds == labels).sum().item()
  acc_v = (correct / total)

  correct = 0
  total = 0            
  for i, data in enumerate(trainloader, 0):
    images, labels = data[0].to(device), data[1].to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, dim=1)
    total += labels.size(0)
    correct += (preds == labels).sum().item()
  acc_t = (correct / total)
  return acc_t,acc_v

def fit(epochs, model, trainloader, testloader, optimizer,scheduler,st_epoch=0,step_count=0):
  history_t = []
  history_v = []
  criterion = nn.CrossEntropyLoss().to(device)
  model.train()
  for epoch in range(st_epoch,epochs):  
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)
      optimizer.zero_grad()
      outputs, aux_outputs = model(inputs)
      loss1 = criterion(outputs, labels)
      loss2 = criterion(aux_outputs, labels)
      loss = loss1 + 0.4*loss2
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      step_count += 1
      if step_count%10000==0:
        acc_t,acc_v = eval_model(model,trainloader,testloader)
        model.train()
        history_t.append(acc_t)
        history_v.append(acc_v)
        log_ = str(step_count)+","+str(acc_t)+","+str(acc_v)+"\n"
        with open("model_1_new.log", "a") as f:
          f.write(log_)
        print("Epoch: {} | Step: {} | loss: {:.4f} | Train acc: {:.4f} | Val acc: {:.4f}".format(epoch+1,step_count, running_loss,acc_t, acc_v))
    scheduler.step()
    filename = '/ssd_scratch/cvit/madhav/check_point1/'+str(epoch+1)+'_checkpoint.pth.tar'
    torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step_count': step_count
            },filename)
    print("----Checkpoint Saved----",epoch+1) 
  return model,history_t,history_v

if not os.path.exists('/ssd_scratch/cvit/madhav/check_point1/'):
  os.makedirs('/ssd_scratch/cvit/madhav/check_point1/')

net = models.inception_v3(pretrained=False).to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9,,weight_decay=0.95)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
if resume_training == True:
	checkpoint =  torch.load('/ssd_scratch/cvit/madhav/check_point1/15_checkpoint.pth.tar')
	net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	st_epoch = checkpoint['epoch']
	step_count = checkpoint['step_count']
else:
	st_epoch = 0
	step_count = 0
trained_model,history_t,history_v = fit(100, net, train_loader, val_loader, optimizer,scheduler,st_epoch=st_epoch,step_count=step_count)
