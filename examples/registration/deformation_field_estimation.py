"""
Practical Deep Learning for Genomic Prediction
==============================================

Credit: A Grigis

Based on:

- https://github.com/miguelperezenciso/DLpipeline

Loas the data
------------

Load some data.
You may need to change the 'datasetdir' parameter.
"""

import os
if "CI_MODE" in os.environ:
    sys.exit()
from pynet.datasets import DataManager, fetch_registration
from pynet.utils import setup_logging
from pynet.registration import VoxelMorphRegister
from torch.optim import lr_scheduler
from pynet.plotting import plot_history
from pynet.history import History
from pynet.losses import mse_loss, gradient_loss
import matplotlib.pyplot as plt

setup_logging(level="info")

outdir = "/neurospin/nsap/datasets/registration"
data = fetch_registration(
    datasetdir=outdir)
manager = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    number_of_folds=10,
    batch_size=1,
    sampler="random",
    #stratify_label="centers",
    test_size=0.1,
    add_input=True)
net = VoxelMorphRegister(
    vol_size=(128, 128, 128),
    enc_nf=[16, 32, 32, 32],
    dec_nf=[32, 32, 32, 32, 32, 16, 16],
    full_size=True,
    optimizer_name="Adam",
    learning_rate=1e-4,
    weight_decay=1e-5,
    loss=mse_loss,
    use_cuda=False)
print(net.model)

def flow_regularizer(signal):
    lambda1 = 0.01
    flow = signal.layer_outputs["flow"]
    grad_regularization = lambda1 * gradient_loss(flow)
    return grad_regularization
net.add_observer("regularizer", flow_regularizer)

scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer=net.optimizer,
    mode="min",
    factor=0.5,
    patience=5)
train_history, valid_history = net.training(
    manager=manager,
    nb_epochs=(1 if "CI_MODE" not in os.environ else 150000),
    checkpointdir=outdir,
    fold_index=0,
    scheduler=scheduler,
    with_validation=True)
print(train_history)
print(valid_history)
plot_history(train_history)


if "CI_MODE" not in os.environ:
    plt.show()
