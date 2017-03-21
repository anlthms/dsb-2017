#
#   Copyright 2017 Anil Thomas
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
"""
Data loader that can be iterated over to retrieve minibatches
"""
import logging
import numpy as np
import pandas as pd
import os
import video
import settings
from neon import NervanaObject


class ChunkLoader(NervanaObject):
    def __init__(self, set_name, repo_dir, datum_dtype=np.uint8,
                 nclasses=2, augment=False, test_mode=False):
        assert test_mode is False, 'Test mode not implemented yet'
        super(ChunkLoader, self).__init__()
        np.random.seed(0)
        self.reset()
        self.set_name = set_name
        self.bsz = self.be.bsz
        self.augment = augment
        self.repo_dir = repo_dir
        self.is_training = (set_name == 'train')
        self.chunk_size = settings.chunk_size
        self.chunk_shape = (self.chunk_size, self.chunk_size, self.chunk_size)
        self.chunk_volume = np.prod(self.chunk_shape)
        self.metadata = pd.read_csv(os.path.join(self.repo_dir, set_name + '-metadata.csv'))
        self.nvids = self.metadata.shape[0]
        self.chunks_filled = 0
        # Load this many videos at a time
        self.vids_per_macrobatch = 128
        # Extract this many chunks from each video
        self.chunks_per_vid = 128
        self.macrobatch_offset = 0
        self.chunks_left_in_macrobatch = 0
        self.macrobatch_size = self.vids_per_macrobatch*self.chunks_per_vid

        self.ndata = self.nvids*self.chunks_per_vid
        self.labels = pd.read_csv(os.path.join(self.repo_dir, 'labels.csv'))
        # Host buffers for macrobatch data and targets
        self.data = np.empty((self.macrobatch_size, self.chunk_volume), dtype=datum_dtype)
        self.targets = np.empty((self.macrobatch_size, nclasses), dtype=np.float32)
        self.minibatch_data = np.empty((self.bsz, self.chunk_volume), dtype=datum_dtype)
        self.minibatch_targets = np.empty((self.bsz, nclasses), dtype=np.float32)
        self.test_mode = test_mode
        self.chunk_count = 0
        self.shape = (1, self.chunk_size, self.chunk_size, self.chunk_size)

        self.transform_buffer = np.empty(self.chunk_shape, dtype=datum_dtype)
        # Device buffers for minibatch data and targets
        self.dev_data = self.be.empty((self.chunk_volume, self.bsz), dtype=self.be.default_dtype)
        self.dev_targets = self.be.empty((nclasses, self.bsz), dtype=self.be.default_dtype)
        self.current_flag = self.current_meta = None

    def reset(self):
        self.start_idx = 0
        self.video_idx = 0
        self.macrobatch_offset = 0
        self.chunks_left_in_macrobatch = 0

    def next_macrobatch(self):
        curr_idx = 0
        self.targets[:] = 0
        for idx in range(self.vids_per_macrobatch):
            vid_data = self.next_video()
            self.chunk_count = self.chunks_per_vid
            self.extract_chunks(vid_data, curr_idx, self.current_flag, self.chunks_per_vid)
            curr_idx += self.chunks_per_vid
            self.chunks_filled += self.chunks_per_vid
        self.chunks_left_in_macrobatch = self.macrobatch_size
        if self.is_training:
            self.shuffle(self.data, self.targets)

    def next_minibatch(self, start):
        end = min(start + self.bsz, self.ndata)
        if end == self.ndata:
            self.start_idx = self.bsz - (self.ndata - start)

        if self.chunks_left_in_macrobatch == 0:
            self.next_macrobatch()
            self.macrobatch_offset = 0

        start = self.macrobatch_offset
        end = start + self.bsz

        self.minibatch_data[:] = self.data[start:end]
        self.minibatch_targets[:] = self.targets[start:end]
        self.dev_data[:] = self.minibatch_data.T.copy()
        self.dev_data[:] = self.dev_data / 255.
        self.dev_targets[:] = self.minibatch_targets.T.copy()
        self.macrobatch_offset += self.bsz
        self.chunks_left_in_macrobatch -= self.bsz
        return self.dev_data, self.dev_targets

    def next_video(self):
        self.current_meta = self.metadata.iloc[self.video_idx]
        uid = self.current_meta['uid']
        self.current_flag = int(self.current_meta['flag'])
        data_filename = os.path.join(self.repo_dir, uid + '.' + settings.file_ext)
        vid_shape = (int(self.current_meta['z_len']),
                     int(self.current_meta['y_len']),
                     int(self.current_meta['x_len']))
        data = video.read_blp(data_filename, vid_shape)
        self.video_idx += 1
        if self.video_idx == self.nvids:
            self.video_idx = 0
        return data

    @property
    def nbatches(self):
        return -((self.start_idx - self.ndata) // self.bsz)

    def __iter__(self):
        for start in range(self.start_idx, self.ndata, self.bsz):
            yield self.next_minibatch(start)

    def transform(self, vid):
        rand1 = np.random.randint(18)
        rand2 = np.random.randint(12)
        if rand1 == 0:
            vid = vid.transpose((0, 2, 1))
        elif rand1 == 1:
            vid = vid.transpose((1, 0, 2))
        elif rand1 == 2:
            vid = vid.transpose((1, 2, 0))
        elif rand1 == 3:
            vid = vid.transpose((2, 0, 1))
        elif rand1 == 4:
            vid = vid.transpose((2, 1, 0))

        if rand2 == 0:
            vid = vid[::-1]
        elif rand2 == 1:
            vid = vid[:, ::-1]
        elif rand2 == 2:
            vid = vid[:, :, ::-1]
        return vid

    def slice_chunk(self, start, data):
        return data[start[0]:start[0]+self.chunk_size,
                    start[1]:start[1]+self.chunk_size,
                    start[2]:start[2]+self.chunk_size].ravel()

    def extract_one(self, cur_idx, chunk_idx, data, data_shape, flag, uid_data):
        assert uid_data.shape[0] != 0
        rand = np.random.randint(8)
        if flag == 1 or rand > 0:
            # Could be a real nodule or a negative sample selected from
            # possible candidates
            i = np.random.randint(uid_data.shape[0])
            center = np.array((uid_data['z'].iloc[i],
                               uid_data['y'].iloc[i],
                               uid_data['x'].iloc[i]), dtype=np.int32)
            rad = 0.5 * uid_data['diam'].iloc[i]
            if rad == 0:
                # Assign an arbitrary radius to candidate nodules
                rad = 24 / settings.resolution
            low = np.int32(center + rad - self.chunk_size)
            high = np.int32(center - rad)
        else:
            # Let in a random negative sample
            low = np.zeros(3, dtype=np.int32)
            high = np.int32(low + data_shape - self.chunk_size)

        for j in range(3):
            low[j] = max(0, low[j])
            high[j] = max(low[j] + 1, high[j])
            high[j] = min(data_shape[j] - self.chunk_size, high[j])
            low[j] = min(low[j], high[j] - 1)
        # Jitter the location of this chunk
        start = [np.random.randint(low=low[i], high=high[i]) for i in range(3)]
        chunk = self.slice_chunk(start, data)

        if self.current_flag != -1:
            self.targets[cur_idx + chunk_idx, self.current_flag] = 1
        if self.augment:
            self.transform_buffer[:] = chunk.reshape(self.transform_buffer.shape)
            self.data[cur_idx + chunk_idx] = self.transform(self.transform_buffer).ravel()
        else:
            self.data[cur_idx + chunk_idx] = chunk
        return True

    def extract_chunks(self, data, cur_idx, flag, count):
        assert count <= self.chunk_count
        meta = self.current_meta
        data_shape = np.array(data.shape, dtype=np.int32)

        uid = meta['uid']
        uid_data = self.labels[self.labels['uid'] == uid]
        chunk_idx = 0
        while chunk_idx < count:
            if self.extract_one(cur_idx, chunk_idx, data, data_shape, flag, uid_data):
                chunk_idx += 1

    def shuffle(self, data, targets):
        inds = np.arange(self.data.shape[0])
        np.random.shuffle(inds)
        data[:] = data[inds]
        targets[:] = targets[inds]
