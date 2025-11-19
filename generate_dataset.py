#################################################################################
#                           generate_dataset.py                                 #
#################################################################################

import torch
from torch.utils.data import DataLoader

class SyntheticSequenceDataset:
    def __init__(self, num_examples = 1000, omega_max = 1.0):                                                   # Synthetic dataset for sequence prediction
        self.num_examples   = num_examples                                                                      # Number of examples in the dataset
        self.x_obs          = torch.zeros(num_examples, 10)                                                     # Observed sequences of length 10       
        self.y_tgt          = torch.zeros(num_examples, 10)                                                     # Target   sequences of length 10

        for i in range(num_examples):
            omega               = omega_max * torch.rand(1)                                                     # Random frequency for sine wave
            phi                 = 2 * 3.141592653589793 * torch.rand(1)                                         # Random phase shift (not used in this example)
            self.x_obs [i, :]    = torch.sin(omega * torch.arange( 0, 10) + phi)                                # Observed: sine values from t =  0 to t =  9
            self.y_tgt[i, :]    = torch.sin(omega * torch.arange(10, 20) + phi)                                 # Target  : sine values from t = 10 to t = 19   

    def __len__(self):
        return self.num_examples                                                                                # Return the number of examples in the dataset

    def __getitem__(self, idx):
        return self.x_obs[idx, :], self.y_tgt[idx, :]                                                           # Return observed-target pair at index idx        
    
def get_dataloaders(dataset, train_size, bs):
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, 1 - train_size],
                                                      generator = torch.Generator().manual_seed(42))            # Split dataset into training and testing sets 
    
    return (DataLoader(train_ds, batch_size = bs, shuffle = True),
            DataLoader( test_ds, batch_size = bs * 2))                                                          # Create data loaders for training and testing sets