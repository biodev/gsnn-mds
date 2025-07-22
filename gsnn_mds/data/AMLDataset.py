import torch 
import numpy as np
import pandas as pd

class AMLDataset(torch.utils.data.Dataset):
    def __init__(self, id2x, resp_data, input_names):

        self.id2x = id2x
        self.resp_data = resp_data 
        self.input_names = input_names

    def __len__(self):
        return len(self.resp_data)

    def __getitem__(self, idx):

        row = self.resp_data.iloc[idx]
        x = self.id2x[row.id].clone().detach()

        drug_idx = self.input_names.index('DRUG__' + row.inhibitor_1.lower())
        x[drug_idx] = 1.

        if pd.notna(row.inhibitor_2): 
            drug_idx = self.input_names.index('DRUG__' + row.inhibitor_2.lower())
            x[drug_idx] = 1.

        y = torch.tensor(row.response, dtype=torch.float32)

        return x,y 