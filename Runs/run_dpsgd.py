from Models.utils import *
from Models.model import init_model
from Utils.utils import *
from tqdm import tqdm

def run(args, tr_loader, va_loader, te_loader, name, device):
    model_name = '{}.pt'.format(name)
    model_name_best = '{}.pt'.format(name + '_best')

    # Defining Model for specific fold
    model = init_model(args=args)
    model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = init_history()

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        tr_loss, tr_out, tr_tar = train_fn_dpsgd(dataloader=tr_loader, model=model, criterion=criterion,
                                                 optimizer=optimizer, device=device, clip=args.clip, ns=args.ns)
        va_loss, va_out, va_tar = eval_fn(va_loader, model, criterion, device)
        te_loss, te_out, te_tar = eval_fn(te_loader, model, criterion, device)

        tr_acc = performace_eval(args, tr_tar, tr_out)
        te_acc = performace_eval(args, te_tar, te_out)
        va_acc = performace_eval(args, va_tar, va_out)

        tk0.set_postfix(loss=tr_loss, acc=tr_acc, val_loss=va_loss, val_acc=va_acc, dp=demo_p, eqopp=equal_odd, te_acc=te_acc)

        history['tr_loss'].append(tr_loss)
        history['tr_acc'].append(tr_acc)
        history['va_loss'].append(va_loss)
        history['va_acc'].append(va_acc)
        history['te_loss'].append(te_loss)
        history['te_acc'].append(te_acc)

        es(epoch=epoch, epoch_score=va_acc, model=model, model_path=args.save_path + model_name_best)
        torch.save(model, args.save_path + model_name)

    model.load_state_dict(torch.load(args.save_path + model_name_best))
    te_loss, te_out, te_tar = eval_fn(te_loader, model, criterion, device)
    te_acc = performace_eval(args, te_tar, te_out)
    history['best_test'] = te_acc
    save_res(args=args, dct=history, name=name)