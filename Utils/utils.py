import random
import os
import numpy as np
import torch
import time
import pickle
from contextlib import contextmanager
from rich import print as rprint
from rich.pretty import pretty_repr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score
# from rich.tree import


@contextmanager
def timeit(logger, task):
    logger.info(f'Started task {task} ...')
    t0 = time.time()
    yield
    t1 = time.time()
    logger.info(f'Completed task {task} - {(t1 - t0):.3f} sec.')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_name(args, current_date):
    dataset_str = f'{args.dataset}_run_{args.seed}_'
    date_str = f'{current_date.day}-{current_date.month}-{current_date.year}_{current_date.hour}-{current_date.minute}'
    model_str = f'{args.mode}_{args.epochs}_hops_{args.n_layers}_'
    dp_str = f'{args.trim_rule}_M_{args.clip_node}_C_{args.clip}_sigma_{args.ns}_'
    if args.mode == 'clean': res_str = dataset_str + model_str + date_str
    else: res_str = dataset_str + model_str + dp_str + date_str
    return res_str

def save_res(name, args, dct):
    save_name = args.res_path + name
    with open('{}.pkl'.format(save_name), 'wb') as f:
        pickle.dump(dct, f)

def print_args(args):
    arg_dict = {}
    keys = ['mode', 'seed', 'performance_metric', 'dataset', 'batch_size', 'sampling_rate', 'lr', 'n_layers',
            'epochs', 'clip', 'clip_node', 'ns', 'debug']
    for key in keys:
        arg_dict[key] = getattr(args, key)

    rprint("Running experiments with hyper-parameters as follows: \n", pretty_repr(arg_dict))
    # print(getattr(args, )

def print_dict(dict_, name):
    rprint(f"Dictionary of {name}: \n", pretty_repr(dict_))


def init_history():
    history = {
        'tr_loss': [],
        'tr_acc': [],
        'va_loss': [],
        'va_acc': [],
        'demo_parity': [],
        'acc_parity': [],
        'equal_opp': [],
        'equal_odd': [],
        'te_loss': [],
        'te_acc': [],
        'best_test': 0,
        'best_demo_parity': 0,
        'best_acc_parity': 0,
        'best_equal_opp': 0,
        'best_equal_odd': 0,
    }
    return history


def performace_eval(args, y_true, y_pred):
    if args.performance_metric == 'acc':
        return accuracy_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
    elif args.performance_metric == 'f1':
        return f1_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
    elif args.performance_metric == 'auc':
        return roc_auc_score(y_true=y_true, y_score=y_pred)
    elif args.performance_metric == 'pre':
        return precision_score(y_true=y_true, y_pred=np.round(np.array(y_pred)))
