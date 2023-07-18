import datetime
import warnings
from Runs.run_dpsgd import run as run_dpsgd
from Utils.utils import *
from config import parse_args
from Data.read_data import *



warnings.filterwarnings("ignore")


def run(args, current_time, device):
    # read data
    train_df, test_df, feature_cols, label = read_adult(args)
    args.feature = feature_cols
    args.target = label
    args.input_dim = len(feature_cols)
    args.output_dim = 1

    rprint(f'Running with dataset {args.dataset}: {len(train_df)} train, {len(test_df)} test, {len(feature_cols)} features')
    tr_loader, va_loader, te_loader = init_data(args=args, fold=0, train=train_df, test=test_df)

    rprint(f'Running with batch size {args.bs}, male {args.bs_male}, female {args.bs_female}')
    name = get_name(args=args, current_date=current_time)

    run_dpsgd(args=args, tr_loader=tr_loader, va_loader=va_loader, te_loader=te_loader, name=name, device=device)


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    args = parse_args()
    seed_everything(args.seed)
    print_args(args=args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run(args, current_time, device)
