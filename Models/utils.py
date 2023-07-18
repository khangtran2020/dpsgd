import numpy as np
import torch
from copy import deepcopy


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001, verbose=False, run_mode=None, skip_ep=100):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.run_mode = run_mode
        self.skip_ep = skip_ep

    def __call__(self, epoch, epoch_score, model, model_path):
        if self.run_mode == 'func' and epoch < self.skip_ep:
            return
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.verbose:
                print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            if self.run_mode != 'func':
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        self.val_score = epoch_score

def eval_fn(data_loader, model, criterion, device):
    model.to(device)
    fin_targets = []
    fin_outputs = []
    loss = 0
    num_data_point = 0
    model.eval()
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            features, target, _ = d
            features = features.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            num_data_point += features.size(dim=0)
            outputs = model(features)
            # if outputs.size(dim=0) > 1:
            outputs = torch.squeeze(outputs, dim=-1)
            loss_eval = criterion(outputs, target)
            loss += loss_eval.item()*features.size(dim=0)
            outputs = outputs.cpu().detach().numpy()

            fin_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
            fin_outputs.extend(outputs)

    return loss/num_data_point, fin_outputs, fin_targets

def train_fn_dpsgd(dataloader, model, criterion, optimizer, device, clip, ns):
    model.to(device)
    model.train()
    noise_std = clip*ns
    train_targets = []
    train_outputs = []
    train_loss = 0
    num_data_point = 0
    features, target, _ = next(iter(dataloader))
    features = features.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    optimizer.zero_grad()
    temp_par = {}
    for p in model.named_parameters():
        temp_par[p[0]] = torch.zeros_like(p[1])
    bz = features.size(dim=0)
    for i in range(bz):
        for p in model.named_parameters():
            p[1].grad = torch.zeros_like(p[1])
        feat = torch.unsqueeze(features[i], 0)
        targ = torch.unsqueeze(target[i], 0)
        output = model(feat)
        output = torch.squeeze(output, dim=1)
        # print(feat.size(), targ, output)
        loss = criterion(output, targ) # / bz
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip, norm_type=2)
        for p in model.named_parameters():
            temp_par[p[0]] = temp_par[p[0]] + deepcopy(p[1].grad)
        output = output.cpu().detach().numpy()
        train_targets.append(targ.cpu().detach().numpy().astype(int).tolist())
        train_outputs.append(output)
        num_data_point += 1

    for p in model.named_parameters():
        noise = torch.normal(mean=0, std=noise_std, size=temp_par[p[0]].size()).to(device)
        p[1].grad = deepcopy(temp_par[p[0]]) + noise
        p[1].grad = p[1].grad / bz

    optimizer.step()

    return train_loss / bz, train_outputs, train_targets
