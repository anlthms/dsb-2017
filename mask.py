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
Segmenting lung CT scans
"""
import os
import numpy as np
import settings
from skimage import measure
from scipy import ndimage


def get_mask(image, uid):
    mask = np.array(image > -320, dtype=np.int8)
    # Set the edges to zeros. This is to connect the air regions, which
    # may appear separated in some scans
    mask[:, 0] = 0
    mask[:, -1] = 0
    mask[:, :, 0] = 0
    mask[:, :, -1] = 0
    labels = measure.label(mask, connectivity=1, background=-1)
    vals, counts = np.unique(labels, return_counts=True)
    inds = np.argsort(counts)
    # Assume that the lungs make up the third largest region
    lung_val = vals[inds][-3]
    if mask[labels == lung_val].sum() != 0:
        print('Warning: could not get mask for %s' % uid)
        mask[:] = 1
        return mask

    mask[labels == lung_val] = 1
    mask[labels != lung_val] = 0
    fill_mask(mask)
    left_center = mask[mask.shape[0] // 2, mask.shape[1] // 2, mask.shape[2] // 4]
    right_center = mask[mask.shape[0] // 2, mask.shape[1] // 2, mask.shape[2] * 3 // 4]
    if (left_center == 0) or (right_center == 0):
        print('Warning: could not get mask for %s' % uid)
        mask[:] = 1
        return mask

    mask = ndimage.morphology.binary_dilation(mask, iterations=settings.mask_dilation)
    return mask


def apply_mask(image, mask):
    image[mask == 0] = 0


def fill_mask(mask):
    for i in range(mask.shape[0]):
        slc = mask[i]
        fill_mask_slice(slc)
    for i in range(mask.shape[1]):
        slc = mask[:, i]
        fill_mask_slice(slc)
    for i in range(mask.shape[2]):
        slc = mask[:, :, i]
        fill_mask_slice(slc)


def fill_mask_slice(slc):
    labels = measure.label(slc, connectivity=1, background=-1)
    vals, counts = np.unique(labels, return_counts=True)
    inds = np.argsort(counts)
    max_val = vals[inds][-1]
    if len(vals) > 1:
        next_val = vals[inds][-2]
        labels[labels == next_val] = max_val
    slc[labels != max_val] = 1
