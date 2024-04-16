from .video_record import VideoRecord


class ActionNetRecord(VideoRecord):
    def __init__(self, tup, dataset_conf):
        self._index = str(tup[0])
        self._series = tup[1]
        self.dataset_conf = dataset_conf

    @property
    def uid(self):
        return self._series['uid']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def label(self):
        if 'verb_class' not in self._series.keys().tolist():
            raise NotImplementedError
        return self._series['verb_class']