import tqdm
import numpy as np
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_embeddings(embeddings):
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)
    
def load_embeddings(nlp, field, dim=300):
    embeddings = torch.FloatTensor(len(field.vocab), dim)
    init_embeddings(embeddings)
    for token, index in tqdm.tqdm(field.vocab.stoi.items()):
        token = nlp(token)
        if token.has_vector:
            embeddings[index] = torch.tensor(token.vector, dtype=torch.float32)
    return embeddings

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, shrink_factor, verbose=False):
    if verbose:
        print("\nDecaying learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    if verbose:
        print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
        
def train_step(model, optimizer, criterion, loader, epoch, grad_clip, tf_ratio, device):
    pass

def validate(model, criterion, loader, epoch, device):
    pass

def train(model, optimizer, criterion, train_loader, valid_loader, field, n_epochs, grad_clip, tf_ratio, device):
    pass