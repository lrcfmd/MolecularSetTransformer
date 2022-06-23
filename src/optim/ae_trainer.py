# Code inspired from https://github.com/lukasruff/Deep-SVDD-PyTorch
from sklearn.metrics import roc_auc_score
from transformers import logging
logging.set_verbosity_error()
import logging
import time
import torch
import torch.optim as optim
import numpy as np

def bidirectional_score(y_pred, y_true):
  """Function for calculating the bidirectional loss of a set input """
  split_idx = y_true.shape[1]//2
  flip = y_true[:, list(range(split_idx, split_idx * 2)) + list(range(split_idx))]
  scores_1 = torch.sum((y_pred - y_true) ** 2, dim=tuple(range(1, y_pred.dim())))
  scores_2 = torch.sum((y_pred - flip) ** 2, dim=tuple(range(1, y_pred.dim())))#
  scores = torch.min(scores_1, scores_2)
  return scores

class AETrainer():

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    def train(self, dataset, ae_net, use_wandb):
        logger = logging.getLogger()

        # Set device for network
        device = self.device if torch.cuda.is_available() else 'cpu'
        ae_net = ae_net.to(device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):
            
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                device = self.device if torch.cuda.is_available() else 'cpu'
                inputs = inputs.to(device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                scores = bidirectional_score(inputs, outputs)
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))
        if use_wandb:
            import wandb
            wandb.log({'loss': loss_epoch / n_batches})
        train_time = time.time() - start_time
        logger.info('Training time: %.3f' % train_time)
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset, ae_net):
        logger = logging.getLogger()

        # Set device for network
        device = self.device if torch.cuda.is_available() else 'cpu'
        ae_net = ae_net.to(device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                device = self.device if torch.cuda.is_available() else 'cpu'
                inputs = inputs.to(device)
                outputs = ae_net(inputs)
                scores = bidirectional_score(inputs, outputs)
                loss = torch.mean(scores)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1
