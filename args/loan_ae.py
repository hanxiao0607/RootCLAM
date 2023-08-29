from argparse import ArgumentParser
import utils.args_parser as argtools


def get_args():

    # VACA
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_file', default='_params/dataset_loan_all.yaml', type=str,
                        help='path to configuration file for the dataset')
    parser.add_argument('--model_file', default='_params/model_vaca_loan.yaml', type=str,
                        help='path to configuration file for the dataset')
    parser.add_argument('--trainer_file', default='_params/trainer.yaml', type=str,
                        help='path to configuration file for the training')
    parser.add_argument('--yaml_file', default='', type=str, help='path to trained model configuration')
    parser.add_argument('-d', '--dataset_dict', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='manually define dataset configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
    parser.add_argument('-m', '--model_dict', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='manually define model configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
    parser.add_argument('-o', '--optim_dict', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='manually define optimizer configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
    parser.add_argument('-t', '--trainer_dict', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='manually define trainer configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
    parser.add_argument('-s', '--seed', default=0, type=int, help='set random seed, default: random')
    parser.add_argument('-r', '--root_dir', default='', type=str, help='directory for storing results')
    parser.add_argument('--data_dir', default='', type=str, help='data directory')
    parser.add_argument('--sample_seed', default=0, type=int, help='seed for sampling')
    parser.add_argument('-i', '--is_training', default=1, type=int,
                        help='run with training (1) or without training (0)')
    parser.add_argument('--show_results', default=1, action="store_true",
                        help='run with evaluation (1) or without(0), default: 1')
    parser.add_argument('--plots', default=0, type=int, help='run code with plotting (1) or without (0), default: 0')
    parser.add_argument('--anomaly_detection_model', default='autoencoder', type=str,
                        help='anomaly detection model deepsvdd or autoencoder')
    parser.add_argument('--training_size', default=10000, type=int, help='training size')

    # AutoEncoder
    parser.add_argument('--train_autoencoder', default=1, type=int, help='train (1) or load(0) autoencoder')
    parser.add_argument('--max_epoch_autoencoder', default=1000, type=int, help='max epoch for training autoencoder')
    parser.add_argument('--batch_size_autoencoder', default=1024, type=int, help='batch size for training autoencoder')
    parser.add_argument('--out_dim_autoencoder', default=2048, type=int, help='output dim for autoencoder')
    parser.add_argument('--nu_autoencoder', default=0.005, type=float, help='quantile for autoencoder')

    # DeepSVDD
    parser.add_argument('--train_deepsvdd', default=1, type=int, help='train (1) or load (0) deepsvdd')
    parser.add_argument('--max_epoch_deepsvdd', default=1000, type=int, help='max epoch for training deepsvdd')
    parser.add_argument('--batch_size_deepsvdd', default=1024, type=int, help='batch size for training deepsvdd')
    parser.add_argument('--out_dim_deepsvdd', default=2048, type=int, help='output dim for deepsvdd')
    parser.add_argument('--nu_deepsvdd', default=0.005, type=float, help='quantile for deepsvdd')

    # RootCLAM
    parser.add_argument('--train_NaiveAM', default=1, type=int, help='train (1) or load (0) NaiveAM')
    parser.add_argument('--train_RootCLAM', default=1, type=int, help='train (1) or load (0) RootCLAM')
    parser.add_argument('--cost_function', default=1, type=int, help='using cost function')
    parser.add_argument('--l2_alpha', default=1e-5, type=float, help='Weight for the l2 loss')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use')
    parser.add_argument('--max_epoch_RootCLAM', default=100, type=int, help='max epoch for training RootCLAM')
    parser.add_argument('--batch_size_RootCLAM', default=128, type=int, help='batch size for training RootCLAM')
    parser.add_argument('--learning_rate_RootCLAM', default=1e-4, type=float, help='Learning rate for RootCLAM')
    parser.add_argument('--rc_quantile', default=0.150, type=float, help='Abnormal quantile for root cause (π)')
    parser.add_argument('--r_ratio', default=0.0, type=float, help='R ratio for flap samples (α)')
    parser.add_argument('--print_all', default=0, type=int, help='print all results (1) or not (0)')

    return parser
