import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from DataSet import DataSetQM9

#TODO: overvej at bruger nworkers

class DataLoaderQM9(DataLoader):
    def __init__(self, datapath: str = "data", batchsize: int = 50, r_cut: float = 5., self_edge: bool=False,test_split: float = 0.1,val_split: float=0.2):
        self.r_cut=5.0
        self.dataset = DataSetQM9(path=datapath, r_cut=r_cut,self_edge=self_edge)
        self.length=len(self.dataset)
        self.train_sampler = SubsetRandomSampler(np.array(range(self.length)))
        self.valid_sampler = None
        self.test_sampler = None
        self.batchsize = batchsize

        if test_split:
            self.test_sampler = self._split(test_split)
        if val_split:
            self.test_sampler = self._split(val_split)
        self.init_kwargs = {'batch_size': batchsize} #TODO: overvej nworkers
        #Return training set
        super().__init__(self.dataset, sampler=self.train_sampler, collate_fn=self.mol_dif, **self.init_kwargs)
    
    
    def mol_dif(self, data):
        """Handle how we stack a batch
        Args:
            data: the data before we output the batch (a tuple containing the dictionary for each molecule)
        """

        batch_dict = {k: [dic[k] for dic in data] for k in data[0].keys()} 

        # We need to define the id and the edges_coord differently (because we begin indexing from 0)
        n_atoms = torch.tensor(batch_dict["n_atom"])
        
        # Converting the n_atom into unique id
        ids = torch.repeat_interleave(torch.tensor(range(len(batch_dict['n_atom']))), n_atoms)
        # Adding the offset to the neighbours coordinate
        edges_coord = torch.cumsum(torch.cat((torch.tensor([0]), n_atoms[:-1])), dim=0)
        neighbours = torch.tensor([local_neigh.shape[0] for local_neigh in batch_dict['edges']])
        edges_coord = torch.cat([torch.repeat_interleave(edges_coord, neighbours).unsqueeze(dim=1), torch.repeat_interleave(edges_coord, neighbours).unsqueeze(dim=1)], dim=1)
        edges_coord += torch.cat(batch_dict['edges'])

        return {
            'z': torch.cat(batch_dict['z']),
            'xyz': torch.cat(batch_dict['xyz']),
            'edges': edges_coord,
            'r_ij': torch.cat(batch_dict['r_ij']),
            'r_ij_normalized': torch.cat(batch_dict['r_ij_normalized']),
            'graph_idx': ids,
            'targets': torch.cat(batch_dict['targets'])
        }
    def _split(self, validation_split: float):
        """ Creates a sampler to extract training and validation data
        Args:
            validation_split: decimal for the split of the validation
        """    
        train_idx = np.array(range(self.length))

        # Getting randomly the index of the validation split (we therefore don't need to shuffle)
        split_idx = np.random.choice(
            train_idx, 
            int(self.length*validation_split), 
            replace=False
        )
        
        # Deleting the corresponding index in the training set
        train_idx = np.delete(train_idx, split_idx)

        # Getting the corresponding PyTorch samplers
        train_sampler = SubsetRandomSampler(train_idx)
        self.train_sampler = train_sampler

        return SubsetRandomSampler(split_idx)

    def get_val(self) -> list:
        """ Return the validation data"""
        if self.valid_sampler is None:
            return None
        else: 
            return DataLoader(self.dataset, sampler=self.valid_sampler, mol_dif=self.mol_dif, **self.init_kwargs)

    def get_test(self) -> list:
        """ Return the test data"""
        if self.test_sampler is None:
            return None
        else: 
            return DataLoader(self.dataset, sampler=self.test_sampler, mol_dif = self.mol_dif, **self.init_kwargs)


