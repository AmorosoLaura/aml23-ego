import glob
import math
from abc import ABC
from random import randint

import numpy as np
import pandas as pd
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger

class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info


        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                sampling_mode = "dense" if self.dense_sampling['RGB'] else "uniform"
                model_features = pd.DataFrame(pd.read_pickle(os.path.join(str(self.dataset_conf[m].data_path),
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          str(self.num_frames_per_clip[m]) + "_" +
                                                                          sampling_mode + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def _get_train_indices(self, record, modality='RGB'):
        ##################################################################
        # TODO: implement sampling for training mode                     #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        if self.dense_sampling[modality]:

            center_frames = np.linspace(0, record.num_frames[modality], self.num_clips + 2,
                                        dtype=np.int32)[1:-1]

            indices = []
            for center in center_frames:
                start = center - math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride)
                end = center + math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride)
                indices.extend(np.arange(start, end, self.stride))

            offset = -indices[0] if indices[0] < 0 else 0
            for i in range(0, len(indices), self.num_frames_per_clip[modality]):
                indices_old = indices[i]
                for j in range(self.num_frames_per_clip[modality]):
                    indices[i + j] = indices[i + j] + offset if indices_old < 0 else indices[i + j]
                    indices[i + j] = record.num_frames[modality] if indices[i+j] > record.num_frames[modality] else indices[i+j]

        else:
            average_duration = record.num_frames[modality] // self.num_frames_per_clip[modality]
            if average_duration > 0:
                frame_idx = np.multiply(np.arange(self.num_frames_per_clip[modality]), average_duration) + \
                            np.random.randint(average_duration, size=self.num_frames_per_clip[modality])
                indices = np.tile(frame_idx, self.num_clips)
            else:
                indices = np.zeros((self.num_frames_per_clip[modality] * self.num_clips,))

        return indices



    def _get_val_indices(self, record, modality):
        ##################################################################
        # TODO: implement sampling for testing mode                      #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        if self.dense_sampling[modality]:

            #generates the evenely space central frames
            center_frames = np.linspace(0, record.num_frames[modality], self.num_clips + 2,
                                        dtype=np.int32)[1:-1]

            indices = []
            for center in center_frames:
                start = center - math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride)
                end = center + math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride)
                indices.extend(np.arange(start, end, self.stride))

            #offset = -indices[0] if indices[0] < 0 else 0
            for i in range(0, len(indices), self.num_frames_per_clip[modality]):
                for j in range(self.num_frames_per_clip[modality]):
                    #indices[i + j] = indices[i + j] + offset if indices[i]< 0 else indices[i + j]
                    #if the index is gretater than the last frame repeat the last frame record.num_frames[modality] 
                    indices[i + j] = record.num_frames[modality] if indices[i + j] > record.num_frames[modality] else indices[i + j]

        else:
            #number of frames between each sampling index
            average_duration = record.num_frames[modality] // self.num_frames_per_clip[modality]
            
            if average_duration <= 0:
                indices = np.zeros((self.num_frames_per_clip[modality] * self.num_clips,))
            else:
                # if num_frames_per_clip is 5 this produced [0,1,2,3,4]
                range_numbers=np.arange(self.num_frames_per_clip[modality])
                #if average_duration is 10 this produces[0,10,20,30,40]
                intervals_idx=np.multiply(range_numbers, average_duration)
                # for each interval select a random frame index, for example [3, 7, 1, 8, 4]
                #i.e 3rd frame from [0,10], 7th frame from [10,20]
                random_indices=np.random.randint(average_duration, size=self.num_frames_per_clip[modality])
                #it selects the frame index with respect to the total number
                #0+3-> 3rd frame, 10+7->17th frame
                frame_index=intervals_idx+random_indices
                #repeat these indices for each clip 
                indices = np.tile(frame_index, self.num_clips)

        return indices

    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]

        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)
