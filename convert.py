#!/usr/bin/env python
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
Convert DICOM images to video data
"""

from __future__ import print_function, division
import os
import sys
import SimpleITK as sitk
import numpy as np
import pandas as pd
import functools
import multiprocessing
from scipy import ndimage
from glob import glob

import settings
import video
import mask


def process_annotations(uid, annots, origin, labels, shape, starts):
    if not is_train:
        return (True, -1)

    uid_data = annots[annots['seriesuid'] == uid]
    if uid_data.shape[0] == 0:
        return (True, 0)

    locs = []
    for idx in range(uid_data.shape[0]):
        row = uid_data.iloc[idx]
        center = np.array([row['coordZ'], row['coordY'], row['coordX']])
        diam = row['diameter_mm']
        if diam == -1:
            diam = 0
        diam /= settings.resolution
        flag = 0 if diam == 0 else 1
        vox_center = np.int32(np.rint((center - origin)/settings.resolution))
        vox_center -= starts
        for i in range(3):
            if vox_center[i] < 0:
                return (False, 0)
            if vox_center[i] >= shape[i]:
                return (False, 0)
        vol = 0 if diam == 0 else 4*np.pi*((diam/2)**3)/3
        locs.append(dict(uid=uid, flag=flag,
                         z=vox_center[0], y=vox_center[1], x=vox_center[2],
                         diam=diam, vol=vol))

    filtered, is_positive = filter_cands(locs)
    for line in filtered:
        labels.loc[labels.shape[0]] = line
    return (True, is_positive)


def filter_cands(locs):
    diams = [loc['diam'] for loc in locs]
    if np.sum(diams) == 0:
        # No malignancy - no need to filter
        return locs, 0
    # Do not take negative candidates from a sample with malignancy
    result = [loc for loc in locs if loc['diam'] != 0]
    return result, 1


def read_scan(path):
    uid = os.path.basename(path)
    if uid.split('.')[-1] == 'mhd':
        uid = uid[:-4]
        return sitk.ReadImage(path), uid

    reader = sitk.ImageSeriesReader()
    image_files = reader.GetGDCMSeriesFileNames(path)
    assert len(image_files) > 0
    if len(image_files) < settings.chunk_size:
        print('Ignoring %s - only %d slices' % (path, len(image_files)))
        return None, uid

    reader.SetFileNames(image_files)
    return reader.Execute(), uid


def get_data(scan_data):
    data = sitk.GetArrayFromImage(scan_data)
    # Convert to (z, y, x) ordering
    spacing = np.array(list(reversed(scan_data.GetSpacing())))
    spacing /= settings.resolution
    slices = ndimage.interpolation.zoom(data, spacing, mode='nearest')
    origin = np.array(list(reversed(scan_data.GetOrigin())))
    return slices, spacing, origin


def trim(slices):
    starts = np.zeros(3, dtype=np.int32)
    end_vals = [slices.shape[i] for i in range(3)]
    ends = np.array(end_vals, dtype=np.int32)

    while slices[starts[0]].sum() == 0:
        starts[0] += 1
    while slices[:, starts[1]].sum() == 0:
        starts[1] += 1
    while slices[..., starts[2]].sum() == 0:
        starts[2] += 1

    while slices[ends[0] - 1].sum() == 0:
        ends[0] -= 1
    while slices[:, ends[1] - 1].sum() == 0:
        ends[1] -= 1
    while slices[..., ends[2] - 1].sum() == 0:
        ends[2] -= 1

    trimmed = slices[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    return trimmed, starts


def convert(path_list, annots, batch_size, max_idx, idx):
    start = idx * batch_size
    end = min(start + batch_size, max_idx)
    path_list = path_list[start:end]
    meta = pd.DataFrame(columns=['uid', 'flag', 'z_len', 'y_len', 'x_len'])
    labels = pd.DataFrame(columns=['uid', 'flag', 'z', 'y', 'x', 'diam', 'vol'])
    for i, path in enumerate(path_list):
        if os.path.basename(path) == 'b8bb02d229361a623a4dc57aa0e5c485':
            # ITK cannot read this file
            continue
        print('Converting %s' % path)
        scan_data, uid = read_scan(path)
        if scan_data is None:
            continue

        slices, spacing, origin = get_data(scan_data)
        skip = 0

        video.clip(slices, settings.low_thresh, settings.high_thresh)
        msk = mask.get_mask(slices, uid)
        slices = video.normalize(slices, settings.low_thresh, settings.high_thresh)
        mask.apply_mask(slices, msk)
        slices, starts = trim(slices)
        valid, flag = process_annotations(uid, annots, origin, labels, slices.shape, starts)
        if not valid:
            print('Ignoring %s - bad metadata' % path)
            continue

        video.write_data(slices, os.path.join(output_path, uid))
        meta.loc[meta.shape[0]] = dict(uid=uid, flag=flag, z_len=slices.shape[0],
                                       y_len=slices.shape[1], x_len=slices.shape[2])
    return meta, labels


if len(sys.argv) < 4:
    print('Usage %s <input-path> <output-path> train/test' % sys.argv[0])
    sys.exit(0)

np.random.seed(0)
input_path = sys.argv[1]
output_path = sys.argv[2]
is_train = sys.argv[3] == 'train'
if is_train:
    search_path = os.path.join(input_path, 'subset*', '*.mhd')
else:
    search_path = os.path.join(input_path, '*')

if not os.path.exists(output_path):
    os.mkdir(output_path)

path_list = glob(search_path)
path_list = sorted(path_list)
np.random.shuffle(path_list)
if is_train:
    annots = pd.read_csv(os.path.join(input_path, 'annotations_excluded.csv'))
else:
    annots = None

count = len(path_list)
assert count > 0, 'Could not find %s' % search_path
print('Converting %d scans...' % count)

cpu_count = multiprocessing.cpu_count()
batch_size = (count - 1) // cpu_count + 1
processes = (count - 1) // batch_size + 1

func = functools.partial(convert, path_list, annots, batch_size, count)
pool = multiprocessing.Pool(processes=processes)
ret_list = pool.map(func, range(processes))
pool.close()
meta_list, labels_list = zip(*ret_list)
meta = pd.concat(meta_list)
labels = pd.concat(labels_list)
meta.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)
if is_train:
    labels.to_csv(os.path.join(output_path, 'labels.csv'), index=False)
