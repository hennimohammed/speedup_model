import torch.utils.data as data 
import torch
import h5py 
from bisect import bisect_right
import numpy as np

class DatasetFromHdf5(data.Dataset):

    def __init__(self, filename, normalized=True, maxsize=30000):
        super(DatasetFromHdf5, self).__init__()

        self.maxsize = maxsize
        self.f = h5py.File(filename, mode='r', swmr=True)
        self.schedules_offset, self.permutations_offset  = zip(*np.array(self.f.get('indexes')))
        self.times = np.array(self.f.get('times'))
        self.programs = np.array(self.f.get('normalized_programs'), dtype='float32')
        self.schedules = np.array(self.f.get('normalized_schedules'), dtype='float32')

        if not normalized:
            self.programs = np.array(self.f.get('programs'), dtype='float32')
            self.schedules = np.array(self.f.get('schedules'), dtype='float32')
        

    def __len__(self):
        if self.maxsize is None:
            return self.permutations_offset[-1]

        return self.maxsize
    

    def __getitem__(self, index):
       
        prog_index = self.get_program_index_pair(index)

        schedule_i, schedule_j = self.get_schedule_pair_index(prog_index, index)

        #print(schedule_i," ", schedule_j)
        #concat program, schedule1, schedule2
        X = np.concatenate((self.programs[prog_index], self.schedules[schedule_i],
                         self.schedules[schedule_j]))
        #speedup of i over j
        Y = self.times[schedule_j] / self.times[schedule_i]

        Y = torch.tensor([Y], dtype=torch.float32)

        return X, Y.numpy()



    
    def get_schedule_pair_index(self, program_index, index):
        schedule_offset = 0 
        permutation_offset = index 
       
        if program_index > 0:
            schedule_offset = self.schedules_offset[program_index-1]
            permutation_offset -= self.permutations_offset[program_index - 1]
            
        n_schedules = self.schedules_offset[program_index] - schedule_offset 
        

        i = permutation_offset // (n_schedules -1)
        j = permutation_offset % (n_schedules -1)

        if j >= i :
            j += 1 

        return i,j
        
    def get_program_index_pair(self, index):
       
        return bisect_right(self.permutations_offset, index)

    def get_program_index(self, index):
        return bisect_right(self.schedules_offset, index)

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

    
       
        
