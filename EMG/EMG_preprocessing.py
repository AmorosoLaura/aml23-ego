import numpy as np
import pickle
import pandas as pd
import os
from scipy import interpolate # for resampling
from scipy.signal import butter, lfilter # for filtering

activities_to_classify = {
        'Get items from refrigerator/cabinets/drawers': 0,
        'Peel a cucumber': 1,
        'Clear cutting board': 2,
        'Slice a cucumber': 3,
        'Peel a potato': 4,
        'Slice a potato': 5,
        'Slice bread': 6,
        'Spread almond butter on a bread slice': 7,
        'Spread jelly on a bread slice': 8,
        'Open a jar of almond butter': 9,
        'Pour water from a pitcher into a glass': 10,
        'Clean a plate with a sponge': 11,
        'Clean a plate with a towel': 12,
        'Clean a pan with a sponge': 13,
        'Clean a pan with a towel': 14,
        'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 15,
        'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 16,
        'Stack on table: 3 each large/small plates, bowls': 17,
        'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 18,
        'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 19,
}
# Define segmentation parameters.
resampled_Fs = 10 # define a resampling rate for all sensors to interpolate
num_segments_per_subject = 20

segment_duration_s = 10
segment_length = int(round(resampled_Fs*segment_duration_s))
# Define filtering parameters.
filter_cutoff_emg_Hz = 5

def load_emg_data(emg_data_path, annotations_path):
    
    files_names = os.listdir(emg_data_path)
    
    all_data=pd.DataFrame()
    
    for file_name in files_names:
        if file_name.startswith("S0"):
            with open(emg_data_path+'/'+file_name, 'rb') as f:
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
    
    
    train_data['description']=train_data['description_x']
    test_data['description']=test_data['description_x']

    train_data.drop(['description_x','description_y'], axis=1, inplace=True)
    test_data.drop(['description_x','description_y'], axis=1, inplace=True)

    print(test_data.columns)
    return train_data, test_data


# Will filter each column of the data.

# taken from Del Preto
#https://github.com/delpreto/ActionNet/blob/master/parsing_data/example_activity_classification/01_create_examples.py

def lowpass_filter(data, cutoff, Fs, order=5):
  nyq = 0.5 * Fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = lfilter(b, a, data.T).T
  return y

def filter_data(data, emg_data_path):

    filtered_data=[]
    for file_name in os.listdir(emg_data_path):
       
        file_data = [action for action in data if action.get('file') == file_name]
        
        if len(file_data)  == 0: #in the test split there are no samples for every subject
            continue
        
        file_data = sorted(file_data, key= lambda action: action['start'])
        
        # taken from Del Preto
        #https://github.com/delpreto/ActionNet/blob/master/parsing_data/example_activity_classification/01_create_examples.py

        for action in file_data:
            for myo_key in ['myo_left', 'myo_right']:
                t = action[myo_key+'_timestamps']
                Fs = (t.size - 1) / (t[-1] - t[0])
                data_stream = action[myo_key+'_readings'][:, :]
                y = np.abs(data_stream)
                y = lowpass_filter(y, filter_cutoff_emg_Hz, Fs)
                
                action[myo_key+'_readings']= y
            filtered_data.append(action)
    return filtered_data

def normalize_data(data, emg_data_path):

    normalized_data=[]
    for file_name in os.listdir(emg_data_path):
        file_data = [action for action in data if action.get('file') == file_name]
        
        if len(file_data)  == 0: #in the test split there are no samples for every subject
            continue
        
        # taken from Del Preto
        #https://github.com/delpreto/ActionNet/blob/master/parsing_data/example_activity_classification/01_create_examples.py

        for action in file_data:
            for myo_key in ['myo_left', 'myo_right']:    
                data_stream = action[myo_key+'_readings']
                y = data_stream
                # Normalize them jointly.
                y = y / ((np.amax(y) - np.amin(y))/2)
                # Jointly shift the baseline to -1 instead of 0.
                y = y - np.amin(y) - 1
                action[myo_key+'_readings'] = y
            normalized_data.append(action)
    return normalized_data

def resample_data(data, emg_data_path):

    resampled_data=[]
   
    for file_name in os.listdir(emg_data_path):
        
        file_data = [action for action in data if action.get('file') == file_name]

        if len(file_data)  == 0: #in the test split there are no samples for every subject
            continue

        file_data=sorted(file_data, key= lambda action: action['start'])
                
        # taken from Del Preto
        #https://github.com/delpreto/ActionNet/blob/master/parsing_data/example_activity_classification/01_create_examples.py

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
            num_subactions = num_segments_per_subject
        # Pad if the action is shorter 
        if len(timestamps_left) < segment_duration_s * resampled_Fs:
            padded_readings_left = np.pad(readings_left, ((0, num_readings_per_segment - len(timestamps_left)), (0, 0)), mode='constant')
            padded_readings_right = np.pad(readings_right, ((0, num_readings_per_segment - len(timestamps_right)), (0, 0)), mode='constant')
            combined_readings = np.concatenate((padded_readings_left, padded_readings_right), axis=1)
            new_action = {
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

    
    EMG_data_path = './EMG_data/Provided'
    
    annotations_path = '../action-net'
    #Load all EMG data and split dataset into train and test splits according to annotations files
    train_data_df, test_data_df = load_emg_data(EMG_data_path , annotations_path) #returns pd dataframe
    
    train_data = train_data_df.to_dict('records') 
    test_data = test_data_df.to_dict('records')
   
    
    train_data=filter_data(train_data,EMG_data_path)
    train_data=normalize_data(train_data,EMG_data_path)
    train_data= resample_data(train_data,EMG_data_path)
    train_data= augment_data(train_data)
    
   
    test_data=filter_data(test_data,EMG_data_path)
    test_data=normalize_data(test_data,EMG_data_path)
    test_data= resample_data(test_data,EMG_data_path)
    test_data= augment_data(test_data)
   
    train_final_df = pd.DataFrame(train_data)
    test_final_df = pd.DataFrame(test_data)
   
    train_final_df['uid']=train_final_df.index
    test_final_df['uid']=test_final_df.index 
    
    activities_renamed = {
    'Open/close a jar of almond butter': 'Open a jar of almond butter',
    'Get/replace items from refrigerator/cabinets/drawers': 'Get items from refrigerator/cabinets/drawers',
    }

    train_final_df['description'] = train_final_df['description'].map(lambda x: activities_renamed[x] if x in activities_renamed else x)
    test_final_df['description'] = test_final_df['description'].map(lambda x: activities_renamed[x] if x in activities_renamed else x)
    
    train_final_df['description_class'] = train_final_df['description'].map(activities_to_classify).astype(int)
    test_final_df['description_class'] = test_final_df['description'].map(activities_to_classify).astype(int) 
   
    train_final_df = train_final_df.sample(frac=1).reset_index(drop=True)
    test_final_df = test_final_df.sample(frac=1).reset_index(drop=True)
   
    output_filepath = './emg_10s_train.pkl'
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(train_final_df, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    output_filepath = './emg_10s_test.pkl'
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(test_final_df, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
