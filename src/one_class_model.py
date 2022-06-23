# Code inspired from https://github.com/lukasruff/Deep-SVDD-PyTorch
import json
import torch
import torch.nn as nn
from src.optim.ae_trainer import AETrainer

class one_class(object):
    """
    Attributes:
        ae_net: The autoencoder network corresponding to \phi for network weights pretraining.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
    """

    def __init__(self):
        """Inits the autoencoder network for training"""

        self.optimizer_name = None
        self.ae_net = None
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }


    def ae_train(self, ae_net, dataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, use_wandb=False, la_steps=0, la_alpha=0):
        
        self.ae_net = ae_net
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net, use_wandb)
        self.ae_trainer.test(dataset, self.ae_net)


    def save_model(self, export_model, save_ae=True):
        """Save the model to export_model."""
        
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False):
        """Load model from model_path."""

        model_dict = torch.load(model_path,map_location='cpu')

        if load_ae:

            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

