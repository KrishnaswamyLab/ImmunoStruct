
from torch.utils.data import Dataset
from typing import Tuple
import copy
import torch
import numpy as np
from .immmunopred_dataloader import AMINO_ACIDS, PADDING_CHAR

__all__ = ["SplitDataset", "ExtendedDataset"]

class SplitDataset:
    def __init__(self, dataset, split, binary, full, comparative=False, return_amino_acid=False):
        self.dataset = dataset
        self.split = split
        self.binary = binary
        self.full = full
        self.comparative = comparative
        self.return_amino_acid = return_amino_acid

    def __getitem__(self, idx):
        d = self.dataset[idx]

        update_graph = d[0]
        amino_acid = None
        if self.split == "train":
            if not self.comparative:
                update_graph = copy.deepcopy(d[0])
                update_graph.ndata['x'][:,-3:] = self.dataset.dataset.transform()(update_graph.ndata['x'][:,-3:])

                if self.return_amino_acid:
                    update_graph, amino_acid = self.dataset.dataset.mask_single_structure(update_graph)

                if self.dataset.dataset.structure_pad_count > 0:
                    update_graph = self.dataset.dataset.mask_structure(update_graph)
            else:
                # It is an array of 2 DGL Graphs.
                assert len(d[0]) == 2
                update_graph = copy.deepcopy(d[0][0])
                update_graph.ndata['x'][:,-3:] = self.dataset.dataset.transform()(update_graph.ndata['x'][:,-3:])

                update_graph_wt = copy.deepcopy(d[0][1])
                update_graph_wt.ndata['x'][:,-3:] = self.dataset.dataset.transform()(update_graph_wt.ndata['x'][:,-3:])

                if self.return_amino_acid:
                    update_graph, update_graph_wt, amino_acid = self.dataset.dataset.mask_single_structure(update_graph, update_graph_wt)

                if self.dataset.dataset.structure_pad_count > 0:
                    update_graph, update_graph_wt = self.dataset.dataset.mask_structure(update_graph, update_graph_wt)

                update_graph = (update_graph, update_graph_wt)

        sequence = None
        if self.full:
            sequence = d[1]

            if self.split == "train" and self.dataset.dataset.sequence_pad_count > 0:
                if not self.comparative:
                    sequence = copy.deepcopy(d[1])
                    sequence = self.dataset.dataset.mask_sequence(sequence, d[2], PADDING_CHAR)
                else:
                    sequence, sequence_wt = d[1]
                    sequence = copy.deepcopy(sequence)
                    sequence_wt = copy.deepcopy(sequence_wt)
                    sequence, sequence_wt = self.dataset.dataset.mask_sequence(sequence, sequence_wt, d[2][0], d[2][1], PADDING_CHAR)
                    sequence = (sequence, sequence_wt)
        else:
            sequence = d[2]

        # format the output
        if self.return_amino_acid and self.binary:
            if self.split == "train":
                return update_graph, sequence, d[4], d[3], amino_acid
            else:
                return update_graph, sequence, d[4], d[3], torch.tensor([0])
            
        elif self.return_amino_acid and not self.binary:
            if self.split == "train":
                return update_graph, sequence, d[5], d[3], amino_acid
            else:
                return update_graph, sequence, d[5], d[3], torch.tensor([0])
            
        elif not self.return_amino_acid and self.binary:
            return d[0], sequence, d[4], d[3]
        
        else:
            return d[0], sequence, d[5], d[3]

    def __len__(self):
        return len(self.dataset)

class ExtendedDataset(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 desired_len: int):
        self.dataset = dataset
        self.desired_len = desired_len

    def __len__(self) -> int:
        return self.desired_len

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        return self.dataset.__getitem__(idx % len(self.dataset))
