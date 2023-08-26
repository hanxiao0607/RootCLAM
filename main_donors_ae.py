import argparse
import json
import os
import warnings
import numpy as np
from utils import utils

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import utils.args_parser as argtools
import utils.tools as utools
from utils.constants import Cte
from models import deepsvdd, naiveam, ae, rootclam

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_file', default='_params/dataset_donors_all.yaml', type=str,
                        help='path to configuration file for the dataset')
    parser.add_argument('--model_file', default='_params/model_vaca_donors.yaml', type=str,
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
    parser.add_argument('--out_dim_deepsvdd', default=32, type=int, help='output dim for deepsvdd')
    parser.add_argument('--nu_deepsvdd', default=0.005, type=float, help='quantile for deepsvdd')

    # RootCLAM
    parser.add_argument('--train_NaiveAM', default=1, type=int, help='train (1) or load (0) NaiveAM')
    parser.add_argument('--train_RootCLAM', default=1, type=int, help='train (1) or load (0) RootCLAM')
    parser.add_argument('--cost_function', default=1, type=int, help='using cost function')
    parser.add_argument('--l2_alpha', default=1e-4, type=float, help='Weight for the l2 loss')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use')
    parser.add_argument('--max_epoch_RootCLAM', default=50, type=int, help='max epoch for training RootCLAM')
    parser.add_argument('--batch_size_RootCLAM', default=128, type=int, help='batch size for training RootCLAM')
    parser.add_argument('--learning_rate_RootCLAM', default=1e-5, type=float, help='Learning rate for RootCLAM')
    parser.add_argument('--rc_quantile', default=0.100, type=float, help='Abnormal quantile for root cause')

    parser.add_argument('--r_ratio', default=0.1, type=float, help='R ratio for flap samples')

    args = parser.parse_args()

    # %%
    if args.yaml_file == '':
        cfg = argtools.parse_args(args.dataset_file)
        cfg.update(argtools.parse_args(args.model_file))
        cfg.update(argtools.parse_args(args.trainer_file))
    else:
        cfg = argtools.parse_args(args.yaml_file)
    if len(args.root_dir) > 0:  cfg['root_dir'] = args.root_dir
    if int(args.seed) >= 0:
        cfg['seed'] = int(args.seed)

    # %%
    pl.seed_everything(cfg['seed'])
    utils.set_seed(cfg['seed'])

    if args.dataset_dict is not None: cfg['dataset']['params2'].update(args.dataset_dict)
    if args.model_dict is not None: cfg['model']['params'].update(args.model_dict)
    if args.optim_dict is not None: cfg['optimizer']['params'].update(args.optim_dict)
    if args.trainer_dict is not None: cfg['trainer'].update(args.trainer_dict)

    cfg['trainer']['auto_select_gpus'] = False
    cfg['trainer']['gpus'] = 0

    cfg['dataset']['params'] = cfg['dataset']['params1'].copy()
    cfg['dataset']['params'].update(cfg['dataset']['params2'])

    if len(args.data_dir) > 0:
        cfg['dataset']['params']['data_dir'] = args.data_dir

    print(args.dataset_dict)
    print(cfg['dataset']['params'])
    print(cfg['model']['params'])

    # %% Load dataset

    data_module = None

    if cfg['dataset']['name'] in Cte.DATASET_LIST:
        from data_modules.het_scm import HeterogeneousSCMDataModule

        dataset_params = cfg['dataset']['params'].copy()
        dataset_params['dataset_name'] = cfg['dataset']['name']
        dataset_params['num_samples_tr'] = args.training_size

        data_module = HeterogeneousSCMDataModule(**dataset_params)

        data_module.prepare_data()

    assert data_module is not None, cfg['dataset']

    thres_n, thres_ab, df_train, df_valid, df_test, test_rc = utils.split_dataset(data_module, name=cfg['dataset']['name'], \
                                                                         training_size=args.training_size,
                                                                         seed=args.seed)

    df_test.to_csv('data/donors_test.csv')
    df_train.to_csv('data/donors_train.csv')

    # %% Load model
    model_vaca = None
    model_params = cfg['model']['params'].copy()
    # VACA
    from models.vaca.vaca import VACA

    model_params['is_heterogeneous'] = data_module.is_heterogeneous
    model_params['likelihood_x'] = data_module.likelihood_list

    model_params['deg'] = data_module.get_deg(indegree=True)
    model_params['num_nodes'] = data_module.num_nodes
    model_params['edge_dim'] = data_module.edge_dimension
    model_params['scaler'] = data_module.scaler

    model_vaca = VACA(**model_params)
    model_vaca.set_random_train_sampler(data_module.get_random_train_sampler())

    assert model_vaca is not None

    utools.enablePrint()

    print(model_vaca.model)
    model_vaca.summarize()
    model_vaca.set_optim_params(optim_params=cfg['optimizer'],
                                sched_params=cfg['scheduler'])

    # %% Evaluator

    evaluator = None

    if cfg['dataset']['name'] in Cte.DATASET_LIST:
        from models._evaluator import MyEvaluator

        evaluator = MyEvaluator(model=model_vaca,
                                intervention_list=data_module.train_dataset.get_intervention_list(),
                                scaler=data_module.scaler
                                )

    assert evaluator is not None

    model_vaca.set_my_evaluator(evaluator=evaluator)

    # %% Prepare training
    if args.yaml_file == '':
        save_dir = argtools.mkdir(os.path.join(cfg['root_dir'],
                                               argtools.get_experiment_folder(cfg),
                                               str(cfg['seed'])))
    else:
        save_dir = os.path.join(*args.yaml_file.split('/')[:-1])
    print(f'Save dir: {save_dir}')
    # trainer = pl.Trainer(**cfg['model'])
    logger = TensorBoardLogger(save_dir=save_dir, name='logs', default_hp_metric=False)
    out = logger.log_hyperparams(argtools.flatten_cfg(cfg))

    save_dir_ckpt = argtools.mkdir(os.path.join(save_dir, 'ckpt'))
    ckpt_file = argtools.newest(save_dir_ckpt)
    callbacks = []
    if args.is_training == 1:

        checkpoint = ModelCheckpoint(period=1,
                                     monitor=model_vaca.monitor(),
                                     mode=model_vaca.monitor_mode(),
                                     save_top_k=1,
                                     save_last=True,
                                     filename='checkpoint-{epoch:02d}',
                                     dirpath=save_dir_ckpt)

        callbacks = [checkpoint]

        if cfg['early_stopping']:
            early_stopping = EarlyStopping(model_vaca.monitor(), mode=model_vaca.monitor_mode(), min_delta=0.0,
                                           patience=50)
            callbacks.append(early_stopping)

        if ckpt_file is not None:
            print(f'Loading model training: {ckpt_file}')
            trainer = pl.Trainer(logger=logger, callbacks=callbacks, resume_from_checkpoint=ckpt_file,
                                 **cfg['trainer'])
        else:

            trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg['trainer'])

        # %% Train

        trainer.fit(model_vaca, data_module)
        # save_yaml(model.get_arguments(), file_path=os.path.join(save_dir, 'hparams_model.yaml'))
        argtools.save_yaml(cfg, file_path=os.path.join(save_dir, 'hparams_full.yaml'))
        # %% Testing

    else:
        # %% Testing
        trainer = pl.Trainer()
        print('\nLoading from: ')
        print(ckpt_file)

        model_vaca = model_vaca.load_from_checkpoint(ckpt_file, **model_params)
        evaluator.set_model(model_vaca)
        model_vaca.set_my_evaluator(evaluator=evaluator)

        if cfg['model']['name'] in [Cte.VACA_PIWAE, Cte.VACA, Cte.MCVAE]:
            model_vaca.set_random_train_sampler(data_module.get_random_train_sampler())

    model_parameters = filter(lambda p: p.requires_grad, model_vaca.parameters())
    params = int(sum([np.prod(p.size()) for p in model_parameters]))

    print(f'Model parameters: {params}')
    model_vaca.eval()
    model_vaca.freeze()  # IMPORTANT

    if args.show_results:
        output_valid = model_vaca.evaluate(dataloader=data_module.val_dataloader(),
                                           name='valid',
                                           save_dir=save_dir,
                                           plots=False)
        output_test = model_vaca.evaluate(dataloader=data_module.test_dataloader(),
                                          name='test',
                                          save_dir=save_dir,
                                          plots=args.plots)
        output_valid.update(output_test)

        output_valid.update(argtools.flatten_cfg(cfg))
        output_valid.update({'ckpt_file': ckpt_file,
                             'num_parameters': params})

        with open(os.path.join(save_dir, 'output.json'), 'w') as f:
            json.dump(output_valid, f)
        print(f'Experiment folder: {save_dir}')

    if cfg['dataset']['name'] == 'loan':
        input_dim = data_module.train_dataset.X0.shape[-1]
    elif cfg['dataset']['name'] == 'adult':
        input_dim = data_module.train_dataset.X0.shape[-1] - 4
    elif cfg['dataset']['name'] == 'donors':
        input_dim = data_module.train_dataset.X0.shape[-1] - 1
    else:
        NotImplementedError
    if args.anomaly_detection_model == 'deepsvdd':
        ad_model = deepsvdd.DeepSVDD(input_dim=input_dim, out_dim=args.out_dim_deepsvdd,
                                     batch_size=args.batch_size_deepsvdd, nu=args.nu_deepsvdd,
                                     max_epoch=args.max_epoch_deepsvdd, data=data_module.dataset_name,
                                     device=args.device)
    else:
        ad_model = ae.AutoEncoder(input_dim=input_dim, hid_dim=args.out_dim_autoencoder,
                                  batch_size=args.batch_size_autoencoder, max_epoch=args.max_epoch_autoencoder,
                                  nu=args.nu_autoencoder, device=args.device, data=data_module.dataset_name)

    if cfg['dataset']['name'] == 'loan':
        train_X = data_module.scaler.transform(data_module.train_dataset.X0)
        valid_X = data_module.scaler.transform(data_module.valid_dataset.X0)
        test_X = data_module.scaler.transform(data_module.test_dataset.X0)
    elif cfg['dataset']['name'] == 'adult':
        train_X = data_module.scaler.transform(data_module.train_dataset.X0)[:, :-4]
        valid_X = data_module.scaler.transform(data_module.valid_dataset.X0)[:, :-4]
        test_X = data_module.scaler.transform(data_module.test_dataset.X0)[:, :-4]
    elif cfg['dataset']['name'] == 'donors':
        train_X = data_module.scaler.transform(data_module.train_dataset.X0)[:, :-1]
        valid_X = data_module.scaler.transform(data_module.valid_dataset.X0)[:, :-1]
        test_X = data_module.scaler.transform(data_module.test_dataset.X0)[:, :-1]
    else:
        NotImplementedError
    if args.anomaly_detection_model == 'deepsvdd':
        if args.train_deepsvdd:
            print('Training DeepSVDD:')
            ad_model.train_DeepSVDD(train_X, valid_X)

        print('Results for DeepSVDD:')
        ad_model.load_model()
        ad_model.get_R(train_X)
    else:
        if args.train_autoencoder:
            print('Training AutoEncoder:')
            ad_model.train_AutoEncoder(train_X, valid_X)

        print('Results for AutoEncoder:')
        ad_model.load_model()
        ad_model.get_R(train_X)

    if cfg['dataset']['name'] == 'donors':
        lst_dist, lst_pred = ad_model.predict(test_X, label=df_test['is_exciting'].values, result=1)
    else:
        lst_dist, lst_pred = ad_model.predict(test_X, label=df_test['label'].values, result=1)

    if cfg['dataset']['name'] == 'loan':
        out_dim = 4
    elif cfg['dataset']['name'] == 'adult':
        out_dim = 3
    elif cfg['dataset']['name'] == 'donors':
        out_dim = 3
    else:
        NotImplementedError

    pl.seed_everything(cfg['seed'])
    utils.set_seed(cfg['seed'])
    print('-' * 50)

    x_train, u_train, x_valid, u_valid, x_test, u_test, df, rc_test = utils.prepare_rootclam_training_data(df_test,
                                                                                                        lst_pred,
                                                                                                        test_rc,
                                                                                                        data_module,
                                                                                                        cfg[
                                                                                                            'dataset'][
                                                                                                            'name'])

    model_naiveam = naiveam.NaiveAM(input_dim, out_dim, ad_model, model_vaca, data_module,
                           alpha=args.l2_alpha, batch_size=args.batch_size_RootCLAM, max_epoch=args.max_epoch_RootCLAM,
                           device=args.device, data=cfg['dataset']['name'], cost_f=args.cost_function,
                           R_ratio=args.r_ratio, lr=args.learning_rate_RootCLAM)

    if args.train_NaiveAM:
        print('Training NaiveAM:')
        model_naiveam.train_NaiveAM(x_train, u_train, x_valid, u_valid)
    print('Results for NaiveAM:')
    model_naiveam.predict(x_test, u_test, thres_n=thres_n)

    print('-' * 50)
    if cfg['dataset']['name'] == 'loan':
        intervention_features = [3, 4, 5, 6]
    elif cfg['dataset']['name'] == 'adult':
        intervention_features = [1, 4, 5]
    elif cfg['dataset']['name'] == 'donors':
        intervention_features = [7, 8, 9]
    else:
        NotImplementedError

    model_rootclam = rootclam.RootCLAM(cfg, input_dim, ad_model, model_vaca, data_module, intervention_features,
                                       train_X, x_test, rc_test, rc_quantile=args.rc_quantile,
                                       alpha=args.l2_alpha, batch_size=args.batch_size_RootCLAM,
                                       max_epoch=args.max_epoch_RootCLAM,
                                       device=args.device, data=cfg['dataset']['name'], cost_f=args.cost_function,
                                       R_ratio=args.r_ratio, lr=args.learning_rate_RootCLAM)
    if args.train_RootCLAM:
        print('Training RootCLAM:')
        model_rootclam.train_RootCLAM(x_train, u_train, x_valid, u_valid)
    print('Results for RootCLAM:')
    model_rootclam.predict(x_test, u_test, thres_n=thres_n)

    print('done')


if __name__ == '__main__':
    main()
