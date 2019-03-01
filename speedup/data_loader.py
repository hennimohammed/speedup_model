import torch.utils.data as data 
import torch
import h5py 
from bisect import bisect_right
import numpy as np

class DatasetFromHdf5(data.Dataset):

    def __init__(self, filename, normalized=True, maxsize=30000):
        super().__init__()

        self.maxsize = maxsize
        self.f = h5py.File(filename, mode='r', swmr=True)
        
        self.schedules = self.f.get('schedules')
        self.programs = self.f.get('programs')
        self.speedups = self.f.get('times')

        self.X = np.concatenate((np.array(self.programs), np.array(self.schedules)), axis=1).astype('float32')
        self.Y = np.array(self.speedups, dtype='float32').reshape(-1, 1)
        
        
    def __len__(self):
        if self.maxsize is None:
            return len(self.Y)

        return self.maxsize
    

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def normalize_min_max(self, data):
        data = np.array(data)

        denominator = data.max(axis=0) - data.min(axis=0) 
        denominator[denominator == 0] = 1

        data = (data - data.min(axis=0))/denominator

        return data

    def normalize_dataset(self):
        #reopen file in write mode
        filename = self.f.filename
        self.f.close()
        self.f = h5py.File(filename, mode='a')

        self.programs = self.f.get('programs')
        self.schedules = self.f.get('schedules')
        #normalize programs 
        normalized_progs = self.normalize_min_max(self.programs)
        self.f.create_dataset('normalized_programs', data=normalized_progs, dtype="float32")
        #normalize schedules
        normalized_scheds = self.normalize_min_max(self.schedules)
        self.f.create_dataset('normalized_schedules', data=normalized_scheds, dtype="float32")

        #go back to read mode
        self.f.close()
        self.__init__(filename)

    
       
        
