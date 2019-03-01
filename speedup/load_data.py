from stats import Stats
import numpy as np 
from itertools import permutations
from tqdm import tqdm 
import h5py


def data_to_h5(programs, schedules, exec_times, filename="speedup_dataset.h5"):
    #create file
    f = h5py.File(filename, mode="w")

    #get dimensions of data
    n_cols_progs = len(programs[0].__array__())
    n_cols_scheds = len(schedules[0][0].__array__())

    #get data
    programs_array, schedules_array, times_array = get_speedup_data(programs, schedules, exec_times)

    assert programs_array.shape[0] ==  schedules_array.shape[0]
    assert schedules_array.shape[0] == times_array.shape[0]
    assert schedules_array.shape[1] == n_cols_scheds
    assert programs_array.shape[1] == n_cols_progs 

    #create datasets 
    f.create_dataset('programs', data=programs_array, dtype="int32")
    f.create_dataset('schedules', data=schedules_array, dtype="int16") 
    f.create_dataset('times', data=times_array) 

    f.close()


def get_speedup_data(programs, schedules, exec_times):

    assert len(programs) == len(schedules) 
    assert len(schedules) == len(exec_times)

    programs_array = np.array([np.array(program) for program in programs])

    schedules_array = []
    times_array = []
    duplicated_programs = []

    for i in range(len(programs_array)):
        assert "no_schedule" in schedules[i][0].name 

        for j in range(len(schedules[i])):
            duplicated_programs.append(programs_array[i])
            schedules_array.append(np.array(schedules[i][j]))

            speedup = exec_times[i][0] / exec_times[i][j] 
            times_array.append(speedup)



    schedules_array = np.array(schedules_array)
    times_array = np.array(times_array)
    duplicated_programs = np.array(duplicated_programs)


    return (duplicated_programs, schedules_array, times_array)


def get_data(programs, schedules, exec_times):
    
    assert len(programs) == len(schedules) 
    assert len(schedules) == len(exec_times)

   
    programs_array = np.array([np.array(program) for program in programs])

    schedules_array = []
    times_array = []

    for program_schedules in schedules:
        program_schedules = [np.array(schedule) for schedule in program_schedules]
        schedules_array.extend(program_schedules)

    for program_times in exec_times:
        times_array.extend(program_times)

    schedules_array = np.array(schedules_array)
    times_array = np.array(times_array)

    indexes_array = []
    schedule_offset = 0
    permutation_offset = 0 

    for i in range(len(programs_array)):
        num_schedules = len(schedules[i]) #number of schedules for prog i

        schedule_offset += num_schedules 
        permutation_offset += num_schedules*(num_schedules - 1)

        indexes_array.append([schedule_offset, permutation_offset])

        
        
    indexes_array = np.array(indexes_array)


    return (programs_array, indexes_array, schedules_array, times_array)


if __name__=='__main__':
    st = Stats('../data/')

    print("loading data")
    programs, schedules, exec_times = st.load_data()
    print("data loaded")
    print("calculating model input")
    data_to_h5(programs, schedules, exec_times)
    print("done")
    