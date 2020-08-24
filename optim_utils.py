import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from utils import IteratorWrapper
from torch_utils import clip_gradient


class ExponentialLR(_LRScheduler):
    """
    Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    
    Arg(s):
        optimizer (torch.optim.Optimizer): The wrapped optimizer.
        end_lr (float): The final learning rate.
        n_iters (int): The number of iterations over which the test occurs.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, n_iters, last_epoch=-1):
        if n_iters <= 1:
            raise ValueError('`num_iter` must be larger than 1')
        self.end_lr = end_lr
        self.n_iters = n_iters
        super(ExponentialLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        r = self.last_epoch / self.n_iters
        return [base_lr * (self.end_lr / base_lr)**r for base_lr in self.base_lrs]
    
    
class LRFinder:
    """
    Learning Rate Finder
    
    Arg(s):
        model (torch.nn.Module): The wrapped model.
        optimizer (torch.optim.Optimizer): The wrapped optimizer where the defined learning
            is assumed to be the lower boundary of the range test.
        criterion (torch.nn.Module): The wrapped loss function.
        grad_clip (float, optional): For gradient clipping. Default: None.
    """
    
    def __init__(self, model, optimizer, criterion, grad_clip=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.grad_clip = grad_clip
        self.history = {'lrs': [], 'losses': []}
        
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, 'init_params.pt')
        
    def range_test(self, data_loader, end_lr=10, n_iters=100, smooth_f=0.05, diverge_th=5):
        """
        Performs learning rate range test
        
        Arg(s):
            data_loader (torch.utils.data.DataLoader): The training set data loader.
            end_lr (float, optional): The maximum learning rate to test. Default: 10.
            n_iters: (int, optional): The number of iterations over which the test
                occurs. Default: 100.
            smooth_f (float, optional): The loss smoothing factor within the [0, 1[
                interval. Disabled if set to 0, otherwise the loss is smoothed using
                exponential smoothing. Default: 0.05.
            diverge_th: (int, optional): The test is stopped when the loss surpasses the
                threshold:  diverge_th * best_loss. Default: 5.
        Return(s):
            history (dict(str, list)): 
        """
        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError('`smooth_f` is outside the range [0, 1[')
            
        best_loss = float('-inf')
        scheduler = ExponentialLR(self.optimizer, end_lr, n_iters)
        iterator = IteratorWrapper(data_loader)
        self.model.train()
        for iteration in tqdm.tqdm(range(n_iters)):
            # Train on a batch
            loss = self._train_batch(next(iterator))
            
             # Update LR
            self.history['lrs'].append(scheduler.get_last_lr()[0])
            scheduler.step()
            
            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                best_loss = loss
            else:
                if smooth_f > 0: # Exponential smoothing
                    loss = smooth_f * loss + (1 - smooth_f) * self.history['losses'][-1]
                if loss < best_loss: # Update the best loss
                    best_loss = loss
            
            # Check if the loss has diverged; if it has, stop the test
            self.history['losses'].append(loss)
            if loss > diverge_th * best_loss: # Stop if the loss diverges
                print('Stopping early, the loss has diverged!')
                break
                
        # Reset the model to its initial parameters
        self.model.load_state_dict(torch.load('init_params.pt').get('model'))
        self.optimizer.load_state_dict(torch.load('init_params.pt').get('optimizer'))
        os.remove('init_params.pt')
        
        print('Learning rate search finished. See the graph with {finder_name}.plot()')
        
    def _train_batch(self, data):
        # Forward prop.
        logits, sorted_dest_sequences, sorted_decode_lengths, sorted_indices = \
            self.model(*data.src, *data.dest, tf_ratio=0.)
        # Since we decoded starting with <sos>, the targets are all words after <sos>, up to <eos>
        sorted_dest_sequences = sorted_dest_sequences[1:, :]
        # Remove paddings
        logits_copy = logits.clone()
        logits = nn.utils.rnn.pack_padded_sequence(
            logits,
            sorted_decode_lengths
        ).data
        sorted_dest_sequences = nn.utils.rnn.pack_padded_sequence(
            sorted_dest_sequences,
            sorted_decode_lengths
        ).data
        # Calculate loss
        loss = self.criterion(logits, sorted_dest_sequences)
        # Back prop.
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        if self.grad_clip is not None:
            clip_gradient(self.optimizer, self.grad_clip)
        # Update weights
        self.optimizer.step()
        return loss.item()
    
    def plot(self, skip_start=10, skip_end=5, log_lr=True, show_lr=None, ax=None, suggest_lr=True):
        """
        Plots the learning rate range test.
        
        Arg(s):
            skip_start (int, optional): The number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): The number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): Plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
            show_lr (float, optional): if set, adds a vertical line to visualize the
                specified learning rate. Default: None.
            ax (matplotlib.axes.Axes, optional): The plot is created in the specified
                matplotlib axes object and the figure is not be shown. If `None`, then
                the figure and axes object are created in this method and the figure is
                shown . Default: None.
            suggest_lr (bool, optional): suggest a learning rate by
                - 'steepest': the point with steepest gradient (minimal gradient)
                you can use that point as a first guess for an LR. Default: True.
        
        Return(s):
            The matplotlib.axes.Axes object that contains the plot,
            and the suggested learning rate (if set suggest_lr=True).
        """
        
        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")
        if show_lr is not None and not isinstance(show_lr, float):
            raise ValueError("show_lr must be float")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lrs"]
        losses = self.history["losses"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Create the figure and axes object if axes was not already given
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)
        
        # Plot the suggested LR
        if suggest_lr:
            # 'steepest': the point with steepest gradient (minimal gradient)
            print("LR suggestion: steepest gradient")
            min_grad_idx = None
            try:
                min_grad_idx = (np.gradient(np.array(losses))).argmin()
            except ValueError:
                print("Failed to compute the gradients, there might not be enough points.")
            if min_grad_idx is not None:
                print("Suggested LR: {:.2E}".format(lrs[min_grad_idx]))
                ax.scatter(
                    lrs[min_grad_idx],
                    losses[min_grad_idx],
                    s=75,
                    marker="o",
                    color="red",
                    zorder=3,
                    label="steepest gradient",
                )
                ax.legend()

        if log_lr:
            ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")
        ax.grid(True, 'both', 'x')

        if show_lr is not None:
            ax.axvline(x=show_lr, color="red")

        # Show only if the figure was created internally
        if fig is not None:
            plt.show()

        if suggest_lr and min_grad_idx:
            return ax, lrs[min_grad_idx]
        else:
            return ax, None
