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
Train and validate a model

Usage:
    ./run.py -w </path/to/data> -e 8 -r 0
"""
import os
import numpy as np
from neon import logger as neon_logger
from neon.layers import GeneralizedCost
from neon.optimizers import Adadelta
from neon.transforms import LogLoss, CrossEntropyBinary
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.initializers import Kaiming
from neon.layers import Conv, Dropout, Pooling, Affine
from neon.transforms import Rectlin, Softmax
from neon.models import Model
from data import ChunkLoader
import settings
import video


def create_network():
    init = Kaiming()
    padding = dict(pad_d=1, pad_h=1, pad_w=1)
    strides = dict(str_d=2, str_h=2, str_w=2)
    dilation = dict(dil_d=2, dil_h=2, dil_w=2)
    common = dict(init=init, batch_norm=True, activation=Rectlin())
    layers = [
        Conv((9, 9, 9, 16), padding=padding, strides=strides, init=init, activation=Rectlin()),
        Conv((5, 5, 5, 32), dilation=dilation, **common),
        Conv((3, 3, 3, 64), dilation=dilation, **common),
        Pooling((2, 2, 2), padding=padding, strides=strides),
        Conv((2, 2, 2, 128), **common),
        Conv((2, 2, 2, 128), **common),
        Conv((2, 2, 2, 128), **common),
        Conv((2, 2, 2, 256), **common),
        Conv((2, 2, 2, 1024), **common),
        Conv((2, 2, 2, 4096), **common),
        Conv((2, 2, 2, 2048), **common),
        Conv((2, 2, 2, 1024), **common),
        Dropout(),
        Affine(2, init=Kaiming(local=False), batch_norm=True, activation=Softmax())
    ]
    return Model(layers=layers)


# Parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('-tm', '--test_mode', action='store_true',
                    help='make predictions on test data')
args = parser.parse_args()

# Create model
model = create_network()

# Setup data provider
repo_dir = args.data_dir
common = dict(datum_dtype=np.uint8, repo_dir=repo_dir, test_mode=args.test_mode)
train = ChunkLoader(set_name='train', augment=not args.test_mode, **common)
test = ChunkLoader(set_name='val', augment=False, **common)

if args.test_mode:
    assert args.model_file is not None
    model.load_params(args.model_file)

    for dataset in [train, test]:
        pred = model.get_outputs(dataset)
        np.save(os.path.join(repo_dir, dataset.set_name + '-pred.npy'), pred[:, 1])
else:
    # Setup callbacks
    callbacks = Callbacks(model, eval_set=test, **args.callback_args)

    # Train model
    opt = Adadelta()
    cost = GeneralizedCost(costfunc=CrossEntropyBinary())
    model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

    # Output metrics
    neon_logger.display('Test Logloss = %.4f' % (model.eval(test, metric=LogLoss())))
