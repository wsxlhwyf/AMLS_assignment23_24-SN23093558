import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch



device = "cuda" if torch.cuda.is_available() else "cpu"

# Build dataset. the rule of using pytorch, needs to packet data according to the type of the dataset.
class mydata(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x).view(-1, 1, 28, 28) # Transfer the data shape into [batch, channel，length， width]
        self.y = torch.FloatTensor(y).view(-1, 1)   # Transfer the shape of y into [batch, 1]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)
    
class CNN(torch.nn.Module):
    # init. Define the module which will be used in building the neural network.
    def __init__(self):
        super(CNN, self).__init__()
        # in_channels is the input of the channel number；out_channels is the number of features；kernel size is the size of filter；stride is the step length
        self.convd1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3), stride=1, padding='valid')
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=1)
        self.convd2 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3,3), stride=1, padding='valid')
        self.pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=1)
        self.mlp1 = torch.nn.Linear(400, 1) # Fully connected layer

    # forward. Build the network with the modules in "init" 
    def forward(self, input):
        output = self.convd1(input)  # convolutional layer
        output = torch.relu(output) # relu activate function
        output = self.pool1(output)  # pooling layer

        output = self.convd2(output)  # convolutional layer
        output = torch.relu(output) # relu activate function
        output = self.pool2(output)  # pooling layer
        
        output = output.view(output.size(0), -1)  # flatten
        output = self.mlp1(output)  # Fully connected layer
        return output 
    
def train_and_eval(model, criterion, opt, epochs, train_dataloader, val_dataloader, test_dataloader):
    train_acc, val_acc, test_acc = [], [], []
    # Hole data needs to be trained "epoch" times
    for epoch in range(epochs):
        model.train()
        # train each batch 
        train_pre, train_label = [], []
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            # forward propogation
            pre = model.forward(x)
            # calculate loss
            loss = criterion(pre, y)
            # gradient-zeroing
            opt.zero_grad()
            # backward propogation
            loss.backward()
            # gradient upgrading
            opt.step()
            if batch_idx % 100 == 0: # output loss every 100 steps
                print('step:' + str(batch_idx) + ' loss:' + str(loss.item()))
            train_pre += pre.cpu().detach().numpy().flatten().tolist()
            train_label += y.cpu().detach().numpy().flatten().tolist()
        train_pre = [1 if i>=0.5 else 0 for i in train_pre]
        acc = accuracy_score(train_label, train_pre)
        train_acc.append(acc)
        print('epoch: ' + str(epoch) +'---' + 'train_acc: ' + str(acc))
        
        model.eval()
        # calculate the result of validation dataset
        val_pre, val_label = [], []
        for _, (x, y) in enumerate(val_dataloader):  
            x = x.to(device)
            y = y.to(device)
            pre = model.forward(x)  # output the prediction of the batch in the validation dataset
            val_pre += pre.cpu().detach().numpy().flatten().tolist()
            val_label += y.cpu().detach().numpy().flatten().tolist()
        val_pre = [1 if i>=0.5 else 0 for i in val_pre]
        acc = accuracy_score(val_label, val_pre)
        val_acc.append(acc)
        print('epoch: ' + str(epoch) +'---' + 'val_acc: ' + str(acc))
        
        # calculate the result of test dataset
        test_pre, test_label = [], []
        for _, (x, y) in enumerate(test_dataloader):  
            x = x.to(device)
            y = y.to(device)
            pre = model.forward(x)  # output the prediction of the batch in the test dataset
            test_pre += pre.cpu().detach().numpy().flatten().tolist()
            test_label += y.cpu().detach().numpy().flatten().tolist()
        test_pre = [1 if i>=0.5 else 0 for i in test_pre]
        acc = accuracy_score(test_label, test_pre)
        test_acc.append(acc)
        print('epoch: ' + str(epoch) +'---' + 'test_acc: ' + str(acc))
    return train_acc, val_acc, test_acc
        

def eval_and_predict(model, test_dataloader):
    test_result = []
    test_label = []
    for _, (x, y) in enumerate(test_dataloader):
        test_pre = model.forward(x)
        test_result += test_pre.detach().numpy().flatten().tolist()
        test_label += y.numpy().flatten().tolist()
    return test_result


# Dataprocessing
print('start...')
pndata = np.load('./Datasets/pneumoniamnist.npz')
train_data, train_label = pndata['train_images']/255, pndata['train_labels']
val_data, val_label = pndata['val_images']/255, pndata['val_labels']
test_data, test_label = pndata['test_images']/255, pndata['test_labels']
train_dataset = mydata(train_data, train_label)
val_dataset = mydata(val_data, val_label)
test_dataset = mydata(test_data, test_label)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
model = CNN()
model.to(device)
# loss function
criterion = torch.nn.BCEWithLogitsLoss()
# Adam optimizer 
opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
epochs = 50

train_acc, val_acc, test_acc = train_and_eval(model, criterion, opt, epochs, train_dataloader, val_dataloader, test_dataloader)

# graph for the trainning accuracy
epochs = [i for i in range(len(train_acc))]
plt.plot(epochs, train_acc)
plt.title('train acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()

# graph for the validation accuracy
epochs = [i for i in range(len(val_acc))]
plt.plot(epochs, val_acc)
plt.title('val acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()

# graph for the test accuracy
epochs = [i for i in range(len(test_acc))]
plt.plot(epochs, test_acc)
plt.title('test acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()

# graph for the overall accuracy
epochs = [i for i in range(len(test_acc))]
plt.plot(epochs, train_acc, label='train')
plt.plot(epochs, val_acc, label='val')
plt.plot(epochs, test_acc, label='test')
plt.legend()
plt.title('accurate')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()

