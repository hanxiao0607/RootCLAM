import warnings
from utils import utils
import sys
from args import adult_ae, adult_deepsvdd, loan_ae, loan_deepsvdd, donors_ae, donors_deepsvdd
import pytorch_lightning as pl
from models import naiveam, rootclam

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def main():
    if len(sys.argv) < 3:
        print('Please give the dataset name (adult, loan, or donors) and the model name (ae or deepsvdd)!')
        return 1
    else:
        dataset_name = sys.argv[1]
        model_name = sys.argv[2]
        if dataset_name == 'adult' and model_name == 'ae':
            parser = adult_ae.get_args()
        elif dataset_name == 'adult' and model_name == 'deepsvdd':
            parser = adult_deepsvdd.get_args()
        elif dataset_name == 'loan' and model_name == 'ae':
            parser = loan_ae.get_args()
        elif dataset_name == 'loan' and model_name == 'deepsvdd':
            parser = loan_deepsvdd.get_args()
        elif dataset_name == 'donors' and model_name == 'ae':
            parser = donors_ae.get_args()
        elif dataset_name == 'donors' and model_name == 'deepsvdd':
            parser = donors_deepsvdd.get_args()
        else:
            print('Please give the dataset name (adult, loan, or donors) and the model name (ae or deepsvdd)!')
            return 1
        args, unknown = parser.parse_known_args()

    cfg = utils.prepare_cfg(args, dataset_name)
    # set seed for reproducibility
    pl.seed_everything(cfg['seed'])
    utils.set_seed(cfg['seed'])
    # prepare dataset
    data_module, lst_ab_data_module = utils.prepare_dataset(args, cfg, dataset_name)
    thres_n, thres_ab, df_train, df_valid, df_test, test_rc = utils.split_dataset(data_module,
                                                                                  lst_ab_data_module=lst_ab_data_module,
                                                                                  name=cfg['dataset']['name'],
                                                                                  training_size=args.training_size,
                                                                                  seed=args.sample_seed)

    # prepare vaca
    model_vaca = utils.prepare_vaca(args, cfg, data_module)
    # prepare anomaly detection model
    ad_model, lst_pred, input_dim, out_dim, train_X = utils.prepare_ad_model(args, cfg, data_module, df_test)
    # prepare data for rootclam
    x_train, u_train, x_valid, u_valid, x_test, u_test, df, rc_test = utils.prepare_rootclam_training_data(df_test,
                                                                                                           lst_pred,
                                                                                                           test_rc,
                                                                                                           data_module,
                                                                                                           cfg[
                                                                                                               'dataset'][
                                                                                                               'name'])
    # initial NaiveAM
    model_naiveam = naiveam.NaiveAM(input_dim, out_dim, ad_model, model_vaca, data_module,
                                    alpha=args.l2_alpha, batch_size=args.batch_size_RootCLAM,
                                    max_epoch=args.max_epoch_RootCLAM,
                                    device=args.device, data=cfg['dataset']['name'], cost_f=args.cost_function,
                                    R_ratio=args.r_ratio, lr=args.learning_rate_RootCLAM, print_all=args.print_all)

    if args.train_NaiveAM:
        print('Training NaiveAM:')
        model_naiveam.train_NaiveAM(x_train, u_train, x_valid, u_valid)
    print('Results for NaiveAM:')
    model_naiveam.predict(x_test, u_test, thres_n=thres_n)
    print('-' * 50)
    # set intervention features
    intervention_features = utils.set_intervention_features(cfg)
    # initial RootCLAM
    model_rootclam = rootclam.RootCLAM(cfg, input_dim, ad_model, model_vaca, data_module, intervention_features,
                                       train_X, x_test, rc_test, rc_quantile=args.rc_quantile,
                                       alpha=args.l2_alpha, batch_size=args.batch_size_RootCLAM,
                                       max_epoch=args.max_epoch_RootCLAM,
                                       device=args.device, data=cfg['dataset']['name'], cost_f=args.cost_function,
                                       R_ratio=args.r_ratio, lr=args.learning_rate_RootCLAM, print_all=args.print_all)
    if args.train_RootCLAM:
        print('Training RootCLAM:')
        model_rootclam.train_RootCLAM(x_train, u_train, x_valid, u_valid)
    print('Results for RootCLAM:')
    model_rootclam.predict(x_test, u_test, thres_n=thres_n)

    print('done')


if __name__ == '__main__':
    main()
