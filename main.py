#################################################################################
#                                   main.py                                     #
#################################################################################

#-------------------------------------------------------------------------------#
#                               GENERATE DATASET                                #
#-------------------------------------------------------------------------------#
from generate_dataset import SyntheticSequenceDataset, get_dataloaders

num_examples = int(5e4)                                                                                     # Total number of examples in the dataset
omega_max    = 1.0                                                                                          # Maximum frequency for sine waves
train_size   = .85                                                                                          # Proportion of data used for training
bs           = 64                                                                                           # Batch size

train_dl, test_dl = get_dataloaders(SyntheticSequenceDataset(num_examples, omega_max), 
                                    train_size, bs)                                                         # Get data loaders for training and testing sets    

#-------------------------------------------------------------------------------#
#                                 CHECK DATA                                    #
#-------------------------------------------------------------------------------#
import torch
import matplotlib.pyplot as plt

x, y = next(iter(train_dl))                                                                                 # Get a batch of data from the training data loader

plt.plot(torch.arange( 0, 10), x[0].numpy(), 'o', label = "Observed sequence")                              # Plot the first observed sequence in the batch
plt.plot(torch.arange(10, 20), y[0].numpy(), 'o', label = "Target sequence")                                # Plot the corresponding target sequence
plt.legend()
plt.show()


