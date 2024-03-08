import os
import pickle
import pandas as pd
import numpy as np
import math
import torch

from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


import numpy as np

def adjust_vector_size(matrix):
    """
    Adjusts the dimension of a matrix based on the specified number of rows,
    while keeping the same number of columns.
    
    Parameters:
        matrix (array-like): The input matrix.  
        num_rows (int): The desired number of rows of the matrix.
        
    Returns:
        numpy.ndarray: The adjusted matrix.
    """
    matrix = np.array(matrix, dtype=float)
    #print("original shape, ", matrix.shape)
    num_rows=750
    current_rows, _ = matrix.shape
    #print(current_rows)
    if current_rows < num_rows and current_rows>0:
        # Pad the matrix with zeros along rows
        padding_rows = num_rows - current_rows
        padded_matrix = np.pad(matrix, ((0, padding_rows), (0,0)), mode='constant', constant_values=(matrix[-1][-1]))
        #print(padded_matrix.shape)
        return padded_matrix
    elif current_rows > num_rows and current_rows>0:
        # Perform uniform sampling to reduce the number of rows
        indices = np.round(np.linspace(0, current_rows - 1, num_rows)).astype(int)
        sampled_matrix = matrix[indices, :]
        print(indices)
        return sampled_matrix
    else:
        # Matrix has the desired number of rows
        return matrix


def lowpass_filter(data, cutoff_freq, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def preprocessing(emg_data, duration,min,max):

    sampling_rate=len(emg_data)/duration
    # Apply low-pass filter
    filtered_emg=lowpass_filter(emg_data, cutoff_frequency, sampling_rate)

            
    # Jointly normalize and shift to the range [-1, 1]
    min_value = np.amin(filtered_emg,axis=0, keepdims=True)
    max_value = np.amax(filtered_emg,axis=0, keepdims=True)
    normalized_and_shifted_data = 2 * (filtered_emg - min_value) / (max_value - min_value) - 1
    
    #normalized_and_shifted_data = 2 * (filtered_emg - min) / (max - min) - 1
    
    #print(normalized_and_shifted_data)
    return normalized_and_shifted_data

def create_subactions(action_data, segment_duration=5, overlap=1):
    """
    Create subactions from the given action interval and return information for each subaction.

    Parameters:
    - action_data: Dictionary containing information about the action instance.
    - segment_duration: Duration of each subaction in seconds.
    - overlap: Overlapping duration between consecutive subactions in seconds.
    - num_subactions: Number of subactions to create.

    Returns:
    - List of dictionaries, each containing information about a subaction.
    """

    start_time_s = action_data['start_time_s']
    end_time_s = action_data['end_time_s']
    duration_s = action_data['duration_s']

    # Calculate the number of subactions
    num_subactions = math.ceil(duration_s/segment_duration)

    subactions_info = []

    data_left=preprocessing(action_data['emg_data_left'], duration_s,global_min_l, global_max_l)
    data_right=preprocessing(action_data['emg_data_right'],duration_s,global_min_r, global_max_r)
    
    # print("Original LEFT: ", action_data["emg_data_left"].shape[0])
    # print("Original RIGHT: ", action_data["emg_data_right"].shape[0])

    row_matrix = max(data_left.shape[0],data_right.shape[0]) // num_subactions

    for i in range(num_subactions):

        subaction_start_time = start_time_s + i* (segment_duration)
        subaction_end_time = subaction_start_time + segment_duration


        start_index = i * row_matrix
        end_index = (i + 1) * row_matrix
        emg_data_subsample_l = data_left[start_index:end_index, :]
   
        emg_data_subsample_r = data_right[start_index:end_index, :]
        
        emg_data_subsample_len = min(emg_data_subsample_l.shape[0], emg_data_subsample_r.shape[0])

        # Concatenazione lungo l'asse 1 (orizzontale)
        emg_data_final = np.concatenate((emg_data_subsample_l[0:emg_data_subsample_len, :], emg_data_subsample_r[0:emg_data_subsample_len, :]), axis=1)

        emg_data_final= adjust_vector_size(emg_data_final)
        
   
        if emg_data_final.shape[0] > 0:
   
            subaction_data = {
                'label': action_data['label'],
                'index': action_data['index'],
                'start_time_s': subaction_start_time,
                'end_time_s': subaction_end_time,
                'duration_s': segment_duration,
                'emg_data': emg_data_final,
            }
            subactions_info.append(subaction_data)
            
    return subactions_info

cutoff_frequency = 5  # Cutoff frequency in Hz


# Percorso della cartella di cui vogliamo ottenere i nomi dei file
cartella = 'C:/Users/Laura/Desktop/Universita/Polito/Advanced Machine Learning/cartella_condivisa_git/aml23-ego/EMG_data/'

# Ottieni i nomi dei file nella cartella
nomi_file = os.listdir(cartella)
videos_name=[]
split="train"

# Ottieni i nomi dei file nella cartella
train_data_preprocessed={}
test_data_preprocessed={}

nomi_file = os.listdir(cartella)
global_min_l=[math.inf]*8
global_max_l=[float('-inf')]*8
global_min_r=[math.inf]*8
global_max_r=[float('-inf')]*8

analyzed_subject=[]
for file in nomi_file:
    print(file)
    if file.startswith("emg-data-S"):
        print("Elaborating  file ", file)
        is_right_split=file.split("-")[3].split(".")[0]==split

        subject=file.split('-')[0]+'-'+file.split('-')[1]+'-'+file.split('-')[2]
        
        if subject not in analyzed_subject:
            analyzed_subject.append(subject)

            for split in ['train', 'test']:
                
                with open(cartella+subject+'-'+split+'.pkl', 'rb') as f_pickle:
                    dati=pickle.load(f_pickle)
                    df=pd.DataFrame(dati)

                    if split=='train':
                        for d in dati:
                            #print(d)
                            # Apply low-pass filter
                            sampling_rate=len(d['emg_data_left'])/d['duration_s']
        
                            filtered_emg=lowpass_filter(d['emg_data_left'], cutoff_frequency, sampling_rate)

                                    
                            # Jointly normalize and shift to the range [-1, 1]
                            min_value = np.amin(filtered_emg,axis=0, keepdims=True)[0]
                            
                            global_min_l = np.minimum(min_value, global_min_l)

                            max_value = np.amax(filtered_emg,axis=0, keepdims=True)[0]
                            global_max_l = np.maximum(max_value, global_max_l)
                            
                            
                            # Apply low-pass filter
                            sampling_rate=len(d['emg_data_right'])/d['duration_s']
        
                            filtered_emg=lowpass_filter(d['emg_data_right'], cutoff_frequency, sampling_rate)

                                    
                            # Jointly normalize and shift to the range [-1, 1]
                            min_value = np.amin(filtered_emg,axis=0, keepdims=True)[0]
                            global_min_r = np.minimum(min_value, global_min_r)
                            
                            max_value = np.amax(filtered_emg,axis=0, keepdims=True)[0]
                            global_max_r = np.maximum(max_value, global_max_r)
                            
                    for d in dati:

                        subactions=create_subactions(d)

                        for s in subactions:
                            if split=='train':
                                    
                                next_key = len(train_data_preprocessed) 
                                #print(subject)
                                train_data_preprocessed[next_key] = {"subject":subject,"emg_data":s['emg_data'],"label":s['label'], "start_timestamp": s['start_time_s'],"end_timestamp": s['end_time_s']}

                            else:       
                                
                                next_key = len(test_data_preprocessed) 
                                #print(subject)
                                test_data_preprocessed[next_key] = {"subject":subject,"emg_data":s['emg_data'],"label":s['label'],"start_timestamp": s['start_time_s'],"end_timestamp": s['end_time_s']}

print(len(train_data_preprocessed))
with open(cartella + 'emg_data_preprocessed_train.pkl', 'wb') as f_pickle:
    pickle.dump(train_data_preprocessed, f_pickle)


print(len(test_data_preprocessed))
with open(cartella + 'emg_data_preprocessed_test.pkl', 'wb') as f_pickle:
    pickle.dump(test_data_preprocessed, f_pickle)