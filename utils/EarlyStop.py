import logging
import numpy as np
import torch
import os

class EarlyStoppingCriterion:
    
    def __init__(self, patience=5, verbose=False, delta=0, save_path='./model', model_name='checkpoint.pth'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_recall_max = -np.Inf 
        self.delta = delta
        self.save_path = save_path
        self.model_name = model_name


        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def __call__(self, val_recall, model):

        if self.best_score is None:
            self.best_score = val_recall
            self.save_checkpoint(val_recall, model)

        elif val_recall < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_recall
            self.save_checkpoint(val_recall, model)
            self.counter = 0

    def save_checkpoint(self, val_recall, model):

        if self.verbose:
            logging.info(f'Validation recall increased ({self.val_recall_max:.6f} --> {val_recall:.6f}). Saving model...')
        torch.save(model.state_dict(), os.path.join(self.save_path, self.model_name))
        self.val_recall_max = val_recall


