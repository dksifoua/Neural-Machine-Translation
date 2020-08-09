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
        
def accuracy(outputs, target_sequences, k=5):
    batch_size = outputs.size(1)
    _, indices = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = indices.eq(target_sequences.view(-1, 1).expand_as(indices))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
        
def save_checkpoint(model, optimizer, data_name, epoch, last_improv, bleu4, is_best):
    state = {
        'epoch': epoch,
        'bleu-4': bleu4,
        'last_improv': last_improv,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, './checkpoint/' + data_name + '.pt')
    if is_best:
        torch.save(state, './checkpoint/' + 'BEST_' + data_name + '.pt')