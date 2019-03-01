from data_loader import *
from model import *
from model_bn import *

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np 

from fire import Fire
import matplotlib.pyplot as plt
import pickle

def plot_results(model_name, losses): 
    train, val = zip(*losses) 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1 ,1)
    ax.plot(range(len(train)), train, 'b--', label='training loss')
    ax.plot(range(len(val)), val, 'g--', label='validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Root mean squared error')
    ax.legend()
    ax.set_title(model_name)
    plt.show()

    

def load_model(input_size, output_size, new=True, filename=None,model_name='model adam', batch_norm=False):
    if not new:
        f = open(filename, 'rb')
        d = pickle.load(f)
    
        return d[model_name][0]

    if batch_norm:
        return Model_BN(input_size, output_size)
    else:
        return Model(input_size, output_size)

def main(batch_size=64, num_epochs=100, 
            num_workers=8, algorithm='adam',
             maxsize=50000, new=True, dataset='data/train_dataset.h5', 
             batch_norm=False, filename='data/results.pkl',lr=0.001):
    
    indices = np.random.RandomState(seed=42).permutation(maxsize)

    val_indices, train_indices = indices[:maxsize//10], indices[maxsize//10:]

    train_dl = DataLoader(DatasetFromHdf5(dataset, maxsize=len(train_indices)), 
                        batch_size=batch_size,
                        sampler=SubsetRandomSampler(train_indices),
                         num_workers=num_workers)

    val_dl = DataLoader(DatasetFromHdf5(dataset, maxsize=len(val_indices)), 
                        batch_size=batch_size, 
                        sampler=SubsetRandomSampler(val_indices),
                         num_workers=num_workers)
    
    input_size = train_dl.dataset.programs.shape[1] + 2*train_dl.dataset.schedules.shape[1] 
    output_size = 1

    model_name = "model " + algorithm
    if batch_norm:
        model_name += " batch_norm"

    model = load_model(input_size, output_size, model_name=model_name,filename=filename, batch_norm=batch_norm)

    criterion = nn.MSELoss()
    optimizer= optim.Adam(model.parameters(), lr=0.01)

    if algorithm != 'adam':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    dl = {'train':train_dl, 'val': val_dl}
 
    model, losses = train_model(model, criterion, optimizer, dl, num_epochs)

    #pickle results
    save_results(model_name, model, losses)
    
    #plot_results(losses)

def save_results(model_name, model, losses, filename="data/results.pkl"):
    f = open(filename, 'rb')
    d = pickle.load(f)
    f.close()

    d[model_name] = (model, losses)

    f = open("results.pkl", "wb")
    pickle.dump(d, f)
    f.close()

def show_results(model_name, filename="data/results.pkl"):
    f = open(filename, 'rb')
    models = pickle.load(f)
    f.close()
    model, losses = models[model_name]
    
    plot_results(model_name, losses)


if __name__ == '__main__':
    Fire()







