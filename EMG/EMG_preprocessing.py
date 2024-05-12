import numpy as np
import pickle
import pandas as pd
import os
from scipy import interpolate # for resampling
from scipy.signal import butter, lfilter # for filtering

EMG_data_path = 'C:/Users/Laura/Desktop/Universita/Polito/Advanced Machine Learning/Progetto/aml23-ego/EMG_data/Provided'
annotations_path = 'C:/Users/Laura/Desktop/Universita/Polito/Advanced Machine Learning/Progetto/aml23-ego/action-net'

subjects = ('S00_2.pkl', 'S01_1.pkl', 'S02_2.pkl' , 'S02_3.pkl','S02_4.pkl', 'S03_1.pkl' ,'S03_2.pkl','S04_1.pkl','S05_2.pkl','S06_1.pkl','S06_2.pkl','S07_1.pkl', 'S08_1.pkl', 'S09_2.pkl')


# Define segmentation parameters.
resampled_Fs = 80 # define a resampling rate for all sensors to interpolate
num_segments_per_subject = 20

segment_duration_s = 10
segment_length = int(round(resampled_Fs*segment_duration_s))
buffer_startActivity_s = 2
buffer_endActivity_s = 2

# Define filtering parameters.
filter_cutoff_emg_Hz = 5

# Will filter each column of the data.
# taken from Del Preto

def lowpass_filter(data, cutoff, Fs, order=5):
  nyq = 0.5 * Fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = lfilter(b, a, data.T).T
  return y

def load_emg_data():
    
    files_names = os.listdir(EMG_data_path)
    
    all_data=pd.DataFrame()
    
    for file_name in files_names:
        if file_name.startswith("S0"):
    
            with open(EMG_data_path+'/'+file_name, 'rb') as f:
                content = pd.read_pickle(f)
                content['file'] = file_name
                content['index']  = np.arange(content.shape[0])
                all_data = pd.concat([all_data, content], ignore_index=True)

    #read annotation files
    annotations_train = pd.read_pickle(os.path.join(annotations_path, 'ActionNet_train.pkl'))
    annotations_test = pd.read_pickle(os.path.join(annotations_path, 'ActionNet_test.pkl'))

    #Inner join by indexes and file
    train_data = annotations_train.merge(all_data, on=['index','file'], how='inner')
    test_data = annotations_test.merge(all_data, on=['index', 'file'], how='inner')
    
    #drop additional "description_y" column after the join and rename "description_x" into simple "description"
    train_data.drop('description_y', axis=1, inplace=True)
    train_data.rename(columns={'description_x': 'description'}, inplace=True)
    
    test_data.drop('description_y', axis=1, inplace=True)
    test_data.rename(columns={'description_x': 'description'}, inplace=True)
    return train_data, test_data

def filter_data(data):

    filtered_data=[]
    for file_name in os.listdir(EMG_data_path):
        #print('Filtering data for subject %s' % file_name)
        file_data = [action for action in data if action.get('file') == file_name]
        
        if len(file_data)  == 0: #in the test split there are no samples for every subject
            continue
        #internally sort actions of this subject because of timestamps when computing Fs
        file_data = sorted(file_data, key= lambda action: action['start'])
        
        for action in file_data:
            for myo_key in ['myo_left', 'myo_right']:
                t = action[myo_key+'_timestamps']
                Fs = (t.size - 1) / (t[-1] - t[0])
                #print(' Filtering %s with Fs %0.1f Hz to cutoff %f' % (myo_key, Fs, filter_cutoff_emg_Hz))
                data_stream = action[myo_key+'_readings'][:, :]
                y = np.abs(data_stream)
                y = lowpass_filter(y, filter_cutoff_emg_Hz, Fs)
                
                action[myo_key+'_readings']= y
            filtered_data.append(action)
    print(len(filtered_data))
    return filtered_data
def normalize_data(data):

    normalized_data=[]
    for file_name in os.listdir(EMG_data_path):
        file_data = [action for action in data if action.get('file') == file_name]
        
        if len(file_data)  == 0: #in the test split there are no samples for every subject
            continue
        
        for action in file_data:
            for myo_key in ['myo_left', 'myo_right']:    
                data_stream = action[myo_key+'_readings']
                y = data_stream
                #print(' Normalizing %s with min/max [%0.1f, %0.1f]' % (myo_key, np.amin(y), np.amax(y)))
                # Normalize them jointly.
                y = y / ((np.amax(y) - np.amin(y))/2)
                # Jointly shift the baseline to -1 instead of 0.
                y = y - np.amin(y) - 1
                action[myo_key+'_readings'] = y
            normalized_data.append(action)
    return normalized_data

def resample_data(data):

    resampled_data=[]
   
    for file_name in os.listdir(EMG_data_path):
        file_data = [action for action in data if action.get('file') == file_name]
        
        
        if len(file_data)  == 0: #in the test split there are no samples for every subject
            continue

        file_data=sorted(file_data, key= lambda action: action['start'])
                
        for action in file_data:
            for myo_key in ['myo_left', 'myo_right']:    
                readings =  np.array(action[myo_key +'_readings'])
                time_s = action[myo_key +'_timestamps']
                
                target_time_s = np.linspace(time_s[0], time_s[-1],
                                            num=int(round(1+resampled_Fs*(time_s[-1] - time_s[0]))),
                                            endpoint=True)
                fn_interpolate = interpolate.interp1d(
                    time_s, # x values
                    readings,   # y values
                    axis=0,              # axis of the data along which to interpolate
                    kind='linear',       # interpolation method, such as 'linear', 'zero', 'nearest', 'quadratic', 'cubic', etc.
                    fill_value='extrapolate' # how to handle x values outside the original range
                )
                data_resampled = fn_interpolate(target_time_s)
                if np.any(np.isnan(data_resampled)):
                    print('\n'*5)
                    print('='*50)
                    print('='*50)
                    print('FOUND NAN')
                    timesteps_have_nan = np.any(np.isnan(data_resampled), axis=tuple(np.arange(1,np.ndim(data_resampled))))
                    print('Timestep indexes with NaN:', np.where(timesteps_have_nan)[0])
                    data_resampled[np.isnan(data_resampled)] = 0
                action[myo_key + '_readings'] = data_resampled
                action[myo_key + '_timestamps'] = target_time_s

            resampled_data.append(action)
        
            
    return resampled_data

def augment_data_old(data):
    #schema: ['index', 'file', 'description', 'labels', 'start','stop', 'myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings']
    augmented_data = []
    
    for action in data:   
        # Compute the start and stop timesteps for each interval of this action
        start_ts = action['start'] 
        stop_ts = action['stop'] 
        duration_s = stop_ts - start_ts
        
        if action['file']=='S04_1.pkl':
            print(action['index'], duration_s)
        """  if duration_s < 5.0:
            continue """

        segment_start_times_s = np.linspace(start_ts, stop_ts - segment_duration_s,
                                            num = num_segments_per_subject,
                                            endpoint=True)
        
        keep_action = True
        for j, segment_start_time_s in enumerate(segment_start_times_s):
            
            segment_end_time_s = segment_start_time_s + segment_duration_s
            
            combined_readings = np.empty(shape=(resampled_Fs * segment_duration_s, 0))
            
            for key in ['myo_right', 'myo_left']:
                
                filtered_myo_indices = np.where((segment_start_time_s <= action[key + '_timestamps']) & (action[key + '_timestamps'] < segment_end_time_s))[0]
                
                filtered_myo_indices = list(filtered_myo_indices)
                #PAD
                while len(filtered_myo_indices) < segment_duration_s*resampled_Fs:
                    if filtered_myo_indices[0] > 0: # != 0
                        filtered_myo_indices = [filtered_myo_indices[0]-1] + filtered_myo_indices
                    elif filtered_myo_indices[-1] < len(action[key + '_timestamps'])-1:
                        filtered_myo_indices.append(filtered_myo_indices[-1]+1)
                    else: #if cannot be extended from beginning nor from end, drop action
                        if action['file']=='S04_1.pkl':
                            print("dropping action", action['index'])
                        keep_action = False
                        break
                    
                #CUT    
                while len(filtered_myo_indices) > segment_duration_s*resampled_Fs:
                    filtered_myo_indices.pop()
                    
                filtered_myo_indices = np.array(filtered_myo_indices)
    
                if keep_action:            
                    #take data
                    filtered_myo_key_readings = np.array([action[key + '_readings'][i] for i in filtered_myo_indices]) 

                    combined_readings = np.concatenate((combined_readings, filtered_myo_key_readings), axis=1)
            
            if keep_action:          
                #! Create new action
                new_action = {'index': action['index'],
                                'file': action['file'],
                                'description': action['description'],
                                'labels': action['labels'],
                                'start': segment_start_time_s,
                                'stop': segment_end_time_s,
                                'emg_data': combined_readings,
                                }

                keep_action = True
                augmented_data.append(new_action)
       
    return augmented_data

def augment_data(data):
    augmented_data = []
    
    for action in data:
        timestamps_left = action['myo_left_timestamps']
        readings_left = action['myo_left_readings']
        timestamps_right = action['myo_right_timestamps']
        readings_right = action['myo_right_readings']
        
        num_readings_per_segment = int(segment_duration_s * resampled_Fs)
        
        # Calculate the number of subactions based on the provided value or the duration of the action
        if len(timestamps_left) < segment_duration_s * resampled_Fs:
            num_subactions = 1
        else:
            num_subactions =20
        # Pad if the action is shorter than 10 seconds
        if len(timestamps_left) < segment_duration_s * resampled_Fs:
            padded_readings_left = np.pad(readings_left, ((0, num_readings_per_segment - len(timestamps_left)), (0, 0)), mode='constant')
            padded_readings_right = np.pad(readings_right, ((0, num_readings_per_segment - len(timestamps_right)), (0, 0)), mode='constant')
            combined_readings = np.concatenate((padded_readings_left, padded_readings_right), axis=1)
            new_action = {
                'index': action['index'],
                'file': action['file'],
                'description': action['description'],
                'labels': action['labels'],
                'start': timestamps_left[0],
                'stop': timestamps_left[-1],
                'emg_data': combined_readings,
            }
            augmented_data.append(new_action)
        else:
            segment_start_times_s = np.linspace(timestamps_left[0], timestamps_left[-1] - segment_duration_s, num=num_subactions)
            
            for start_time in segment_start_times_s:
                start_index = np.where(timestamps_left >= start_time)[0][0]
                end_index = start_index + num_readings_per_segment
                
                # Control end timestamp
                if end_index >= len(timestamps_left):
                    break
                
                # Left arm readings
                left_segment_readings = readings_left[start_index:end_index]
                # Pad if necessary
                if len(left_segment_readings) < num_readings_per_segment:
                    left_segment_readings = np.pad(left_segment_readings, ((0, num_readings_per_segment - len(left_segment_readings)), (0, 0)), mode='constant')

                # Right arm readings
                right_segment_readings = readings_right[start_index:end_index]
                # Pad if necessary
                if len(right_segment_readings) < num_readings_per_segment:
                    right_segment_readings = np.pad(right_segment_readings, ((0, num_readings_per_segment - len(right_segment_readings)), (0, 0)), mode='constant')

                # Concatenate readings from both arms along the y-axis
                combined_segment_readings = np.concatenate((left_segment_readings, right_segment_readings), axis=1)

                # Create new_action structure
                new_action = {
                    'index': action['index'],
                    'file': action['file'],
                    'description': action['description'],
                    'labels': action['labels'],
                    'start': timestamps_left[start_index],
                    'stop': timestamps_left[end_index-1],
                    'emg_data': combined_segment_readings,
                }
                augmented_data.append(new_action)  # Append new_action to augmented_data list
        
    return augmented_data

if __name__=='__main__':
    #Load all EMG data and split dataset into train and test splits according to annotations files
    train_data_df, test_data_df = load_emg_data() #returns pd dataframe
    
    #Convert the datasets to dictionaries with schema: ['index', 'file', 'description', 'labels', 'start', 'stop','myo_left_timestamps', 'myo_left_readings','myo_right_timestamps', 'myo_right_readings']
    train_data = train_data_df.to_dict('records') 
    test_data = test_data_df.copy().to_dict('records')
   
    
    train_data=filter_data(train_data)
    train_data=normalize_data(train_data)
    train_data= resample_data(train_data)
    train_data= augment_data(train_data)
    
   
    test_data=filter_data(test_data)
    test_data=normalize_data(test_data)
    test_data= resample_data(test_data)
    test_data= augment_data(test_data)
   
    #Convert back to pd dataframes
    train_final_df = pd.DataFrame(train_data, columns=['index', 'file', 'description', 'labels', 'start','stop','emg_data'])
    test_final_df = pd.DataFrame(test_data, columns=['index', 'file', 'description', 'labels', 'start','stop', 'emg_data'])

    #There are some activities with slightly different names that I want to merge 
    activities_renamed = {
        'Open/close a jar of almond butter': ['Open a jar of almond butter'],
        'Get/replace items from refrigerator/cabinets/drawers': ['Get items from refrigerator/cabinets/drawers'],
    }
    
    train_final_df.loc[train_final_df['description'] == 'Open/close a jar of almond butter', 'description'] = 'Open a jar of almond butter'
    test_final_df.loc[test_final_df['description'] == 'Open/close a jar of almond butter', 'description'] = 'Open a jar of almond butter'
    train_final_df.loc[train_final_df['description'] == 'Get/replace items from refrigerator/cabinets/drawers', 'description'] = 'Get items from refrigerator/cabinets/drawers'
    test_final_df.loc[test_final_df['description'] == 'Get/replace items from refrigerator/cabinets/drawers', 'description'] = 'Get items from refrigerator/cabinets/drawers'
    
    # #add class column based on different instances of "description"
    unique_values = train_final_df['description'].unique()
    value_to_int = {value: idx for idx, value in enumerate(unique_values)}
    train_final_df['description_class'] = train_final_df['description'].map(value_to_int)
    test_final_df['description_class'] = test_final_df['description'].map(value_to_int)
    
    #add unique index column identifying each action, because "index" column has the same value for augmented actions
    train_final_df['uid'] = range(len(train_final_df))
    test_final_df['uid'] = range(len(test_final_df))
    train_final_df = train_final_df.sample(frac=1).reset_index(drop=True)
    test_final_df = test_final_df.sample(frac=1).reset_index(drop=True)
    
    output_filepath = 'C:/Users/Laura/Desktop/Universita/Polito/Advanced Machine Learning/Progetto/aml23-ego/new_emg_data_80fs_train.pkl'
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(train_final_df, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    output_filepath = 'C:/Users/Laura/Desktop/Universita/Polito/Advanced Machine Learning/Progetto/aml23-ego/new_emg_data_80fs_test.pkl'
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(test_final_df, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
