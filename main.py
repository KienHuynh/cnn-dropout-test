import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import pickle

import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_name = 'dropout0.5'
dropout_p = 0.2 # Dropout rate = 0%
save_path = '../data/trained/cnn-dropout-test/'

# Prepare the data
def data_prepare(data_path='../data/cifar-10-batches-py/'):
    train_X = np.zeros((50000, 32*32*3), dtype=np.float32)
    train_Y = np.zeros((50000, ), dtype=np.int8)
    test_X = np.zeros((10000, 32*32*3), dtype=np.float32)
    test_Y = np.zeros((10000, ), dtype=np.int8)
    for i in range(1,6):
        file_name = data_path + ('data_batch_%d' % i)
        with open(file_name, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            train_X[(i-1)*10000:i*10000, :] = batch['data']
            train_Y[(i-1)*10000:i*10000] = batch['labels']

    file_name = data_path + ('test_batch')
    with open(file_name, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        test_X = batch['data']
        test_Y[0:10000] = batch['labels']
    
    # Reshape data 
    train_X = np.reshape(train_X, (50000, 3, 32, 32))
    #train_Y = create_one_hot(train_Y, 10)
    test_X = np.reshape(test_X, (10000, 3, 32, 32))
    #test_Y = create_one_hot(test_Y, 10)

    train_mean = np.mean(train_X, axis=(0,2,3), keepdims=True)
    train_std = np.std(train_X, axis=(0,2,3), keepdims=True)
    train_X = (train_X-train_mean)/train_std
    test_X = (test_X-train_mean)/train_std
    train_Y = train_Y*10 + 10
    test_Y = test_Y*10 + 10
    #plt.imshow(np.transpose(train_X[0,:,:,:], (1,2,0)).astype(np.uint8)) 
    return (train_X, train_Y.astype(np.float32), test_X, test_Y.astype(np.float32))

class CNN(nn.Module):
    def __init__(self, dropout_p=0):
        super(CNN, self).__init__()
        self.conv11 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv12 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv21 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv22 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv31 = nn.Conv2d(128, 256, 5, padding=2)
        self.conv32 = nn.Conv2d(256, 256, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout2d(dropout_p)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool(self.dropout(x))
        
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.pool(self.dropout(x))

        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = self.pool(x)

        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test(train=True, test_times=0):
    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available() 
    net = CNN(dropout_p)
     
    if use_cuda:
        net.cuda()
    
    net.load_state_dict(torch.load(save_path + ('%s.dat' % model_name))) 
    if(train):
        net.train()
    else:
        net.eval()
    train_X, train_Y, test_X, test_Y = data_prepare()
    criterion = nn.MSELoss()
    
    batch_size = 100
    num_train = 50000
    num_test = 10000
    num_ite_per_e = int(np.ceil(float(num_test)/float(batch_size)))
    full_ind = np.arange(num_test)
    all_loss = [] 
    all_pred = []
    running_loss = 0.0
    for t in range(test_times):
        for i in range(num_ite_per_e):
            if (i+1)*batch_size <= num_test:
                batch_range = range(i*batch_size, (i+1)*batch_size)
            else:
                batch_range = range(i*batch_size, num_test)
            batch_range = full_ind[batch_range]
            batch_X = np_to_var(test_X[batch_range], use_cuda) 
            batch_Y = np_to_var(test_Y[batch_range], use_cuda)
            outputs = net(batch_X)
            loss = (outputs.view(-1) - batch_Y)**2 
            all_pred += list(var_to_np(outputs.view(-1), use_cuda))
            all_loss += list(var_to_np(loss, use_cuda))
            if i % 10 == 9 and not train:
                print("Testing iteration %d out of %d" % (i, num_ite_per_e))            
        if (train):
            print("Tested test %d out of %d" % (t, test_times))

    print('Prediction mean: %.3f' % np.mean(all_pred)) 
    print('Test result: %.3f' % np.mean(all_loss))
    pdb.set_trace()

def train():
    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available() 
    net = CNN(dropout_p)
    
    if use_cuda:
        net.cuda()
   
    net.train() 
    train_X, train_Y, test_X, test_Y = data_prepare()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    
    batch_size = 100
    num_train = 50000
    num_test = 10000
    num_ite_per_e = int(np.ceil(float(num_train)/float(batch_size)))
    full_ind = np.arange(num_train)
    rng = np.random.RandomState(1311) 
    all_loss = [] 
    for e in range(200):
        running_loss = 0.0
        for i in range(num_ite_per_e):
            rng.shuffle(full_ind)
            optimizer.zero_grad()
            
            if (i+1)*batch_size <= num_train:
                batch_range = range(i*batch_size, (i+1)*batch_size)
            else:
                batch_range = range(i*batch_size, num_train)
            batch_range = full_ind[batch_range]
            batch_X = np_to_var(train_X[batch_range], use_cuda) 
            batch_Y = np_to_var(train_Y[batch_range], use_cuda)
            outputs = net(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0]
            all_loss.append(loss.data[0])
            if i % 100 == 99:
                print('[%d, %d] loss: %.3f' % (e+1, e*num_ite_per_e+i+1, running_loss/100))
                running_loss = 0.0
            if i % 500 == 499:
                plt.clf()
                plt.plot(all_loss)
                plt.show()
                plt.savefig('../vis/Image2Latex/nodropout.pdf')
    
    torch.save(net.state_dict(), save_path + ('%s.dat' % model_name))



def np_to_var(tensor, use_cuda):
    if use_cuda:
        return Variable(torch.from_numpy(tensor)).cuda()
    else:
        return Variable(torch.from_numpy(tensor))


def var_to_np(tensor, use_cuda):
    if use_cuda:
        return tensor.data.cpu().numpy()
    else:
        return tensor.data.numpy() 


if __name__ == '__main__':
    torch.manual_seed(1311)
    np.random.seed(1311)
    train()
    test(train=True, test_times=100)
    test(False, test_times=1)
