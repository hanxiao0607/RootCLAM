import numpy as np
import pandas as pd
import math
import random
import torch
import torch.nn.functional as F


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def split_dataset(data_module, lst_ab_data_module=None, name='loan', training_size=5000, seed=0):
    data_X = np.concatenate((data_module.train_dataset.X, data_module.valid_dataset.X, data_module.test_dataset.X),
                            axis=0)
    data_U = np.concatenate((data_module.train_dataset.U, data_module.valid_dataset.U, data_module.test_dataset.U),
                            axis=0)
    if name == 'loan':
        df = pd.DataFrame(np.concatenate((data_X, data_U), axis=1), \
                          columns=['G', 'A', 'E', 'L', 'D', 'I', 'S', \
                                   'U_G', 'U_A', 'U_E', 'U_L', 'U_D', 'U_I', 'U_S'])
        # gender G, age A, education E, loan amount L, loan duration D, income I, savings S
        exp_val = -0.3 * (- df['L'] - df['D'] + df['I'] + df['S'] + df['I'] * df['S'])
        lst_y = []
        for i in range(len(df)):
            lst_y.append((1 + math.exp(exp_val[i])) ** (-1))
        df['y'] = lst_y
        index = 6
        thres_n = 0.95
        thres_ab = 0.05

        df_n = df.loc[df['y'] > thres_n]
        df_n = df_n.sample(n=len(df_n), random_state=seed)
        df_n['label'] = 0

        size_train_n = training_size
        size_valid_n = int(training_size * 0.1)
        size_test_n = int(training_size * 1.0)
        size_test_ab = int(size_test_n * 0.1)
        portion_size_test_ab = int(size_test_ab/len(lst_ab_data_module))

        df_ab = pd.DataFrame()
        for i, module in enumerate(lst_ab_data_module):
            data_ab_X = np.concatenate((module.train_dataset.X, module.valid_dataset.X, module.test_dataset.X), axis=0)
            data_ab_U = np.concatenate((module.train_dataset.U, module.valid_dataset.U, module.test_dataset.U), axis=0)
            df_temp = pd.DataFrame(np.concatenate((data_ab_X, data_ab_U), axis=1), \
                              columns=['G', 'A', 'E', 'L', 'D', 'I', 'S', \
                                       'U_G', 'U_A', 'U_E', 'U_L', 'U_D', 'U_I', 'U_S'])
            # gender G, age A, education E, loan amount L, loan duration D, income I, savings S
            exp_val = -0.3 * (- df_temp['L'] - df_temp['D'] + df_temp['I'] + df_temp['S'] + df_temp['I'] * df_temp['S'])
            lst_y = []
            for j in range(len(df_temp)):
                lst_y.append((1 + math.exp(exp_val[j])) ** (-1))
            df_temp['y'] = lst_y
            df_ab_temp = df_temp.loc[df_temp['y'] < thres_ab]
            df_ab_temp['label'] = 1
            df_ab_temp['rc'] = i
            df_ab_temp = df_ab_temp.sample(n=portion_size_test_ab, random_state=seed)
            df_ab = pd.concat([df_ab, df_ab_temp])

        df_ab = df_ab.sample(n=len(df_ab), random_state=seed)

        train = df_n.iloc[:size_train_n]
        valid = df_n.iloc[size_train_n:(size_train_n + size_valid_n)]
        test = pd.concat([df_n.iloc[(size_train_n + size_valid_n):(size_train_n + size_valid_n + size_test_n)] \
                             , df_ab.iloc[:,:-1]], ignore_index=True)
        test_rc = df_ab['rc'].values
        train_X = train.iloc[:, :(index + 1)].values
        valid_X = valid.iloc[:, :(index + 1)].values
        test_X = test.iloc[:, :(index + 1)].values
        train_U = train.iloc[:, (index + 1):-2].values
        valid_U = valid.iloc[:, (index + 1):-2].values
        test_U = test.iloc[:, (index + 1):-2].values
        data_module.train_dataset.X = train_X
        data_module.valid_dataset.X = valid_X
        data_module.test_dataset.X = test_X
        data_module.train_dataset.U = train_U
        data_module.valid_dataset.U = valid_U
        data_module.test_dataset.U = test_U

        data_module.train_dataset.X0 = train_X
        data_module.valid_dataset.X0 = valid_X
        data_module.test_dataset.X0 = test_X
        print(f'Training size: {len(train)}')
        print(f'Validation size: {len(valid)}')
        print(f'Testing size: {len(test)}')

        return thres_n, thres_ab, train, valid, test, test_rc

    elif name == 'adult':
        df = pd.DataFrame(np.concatenate((data_X, data_U), axis=1), \
                          columns=['R_0', 'R_1', 'R_2', 'A', 'N_0', 'N_1', 'N_2', 'N_3', 'S', 'E', 'H', \
                                   'W_0', 'W_1', 'W_2', 'W_3', 'M_0', 'M_1', 'M_2', 'O_0', 'O_1', 'O_2', \
                                   'L_0', 'L_1', 'L_2', 'I', 'U_R', 'U_A', 'U_N', 'U_S', 'U_E', 'U_H', \
                                   'U_W', 'U_M', 'U_O', 'U_L', 'U_I'])
        # race R, age A, native country N, sex S, Education E, hours per week H, work status W, martial status M, \
        # occupation O, relationship L, income I
        index = 24
        thres_n = np.quantile(df['I'].values, 0.50)
        thres_ab = np.quantile(df['I'].values, 0.90)
        thres_ab_l = np.quantile(df['I'].values, 1.0)
        print(f'Normal {thres_n}, Abnormal {thres_ab}, Abnormal upper {thres_ab_l}')

        df_n = df.loc[df['I'] <= thres_n]
        df_n = df_n.sample(n=len(df_n), random_state=seed)
        df_n['label'] = 0

        size_train_n = training_size
        size_valid_n = int(training_size * 0.2)
        size_test_n = int(training_size * 1.0)
        size_test_ab = int(size_test_n * 1)
        portion_size_test_ab = int(size_test_ab / len(lst_ab_data_module))

        df_ab = pd.DataFrame()
        for i, module in enumerate(lst_ab_data_module):
            data_ab_X = np.concatenate((module.train_dataset.X, module.valid_dataset.X, module.test_dataset.X), axis=0)
            data_ab_U = np.concatenate((module.train_dataset.U, module.valid_dataset.U, module.test_dataset.U), axis=0)
            df_temp = pd.DataFrame(np.concatenate((data_ab_X, data_ab_U), axis=1), \
                                   columns=['R_0', 'R_1', 'R_2', 'A', 'N_0', 'N_1', 'N_2', 'N_3', 'S', 'E', 'H', \
                                            'W_0', 'W_1', 'W_2', 'W_3', 'M_0', 'M_1', 'M_2', 'O_0', 'O_1', 'O_2', \
                                            'L_0', 'L_1', 'L_2', 'I', 'U_R', 'U_A', 'U_N', 'U_S', 'U_E', 'U_H', \
                                            'U_W', 'U_M', 'U_O', 'U_L', 'U_I'])

            df_ab_temp = df_temp.loc[df_temp['I'] > thres_ab]
            df_ab_temp['label'] = 1
            df_ab_temp['rc'] = i
            df_ab_temp = df_ab_temp.sample(n=portion_size_test_ab, random_state=seed)
            df_ab = pd.concat([df_ab, df_ab_temp])

        df_ab = df_ab.sample(n=len(df_ab), random_state=seed)

        train = df_n.iloc[:size_train_n]
        valid = df_n.iloc[size_train_n:(size_train_n + size_valid_n)]
        test = pd.concat([df_n.iloc[(size_train_n + size_valid_n):(size_train_n + size_valid_n + size_test_n)] \
                             , df_ab.iloc[:,:-1]], ignore_index=True)
        test_rc = df_ab['rc'].values
        train_X = train.iloc[:, :(index + 1)].values
        valid_X = valid.iloc[:, :(index + 1)].values
        test_X = test.iloc[:, :(index + 1)].values
        train_U = train.iloc[:, (index + 1):-1].values
        valid_U = valid.iloc[:, (index + 1):-1].values
        test_U = test.iloc[:, (index + 1):-1].values
        data_module.train_dataset.X = train_X
        data_module.valid_dataset.X = valid_X
        data_module.test_dataset.X = test_X
        data_module.train_dataset.U = train_U
        data_module.valid_dataset.U = valid_U
        data_module.test_dataset.U = test_U

        data_module.train_dataset.X0 = data_module.train_dataset.fill_up_with_zeros(data_module.train_dataset.X)[0]
        data_module.valid_dataset.X0 = data_module.train_dataset.fill_up_with_zeros(data_module.valid_dataset.X)[0]
        data_module.test_dataset.X0 = data_module.train_dataset.fill_up_with_zeros(data_module.test_dataset.X)[0]
        print(f'Training normal size: {len(train)}')
        print(f'Testing normal size: {len(test.loc[test["label"] == 0])}')
        print(f'Testing abnormal size: {len(test.loc[test["label"] == 1])}')
        return thres_n, thres_ab, train, valid, test, test_rc

    elif name == 'donors':
        thres_n = 0
        thres_ab = 1
        train = pd.DataFrame(data_module.train_dataset.X, columns=['at_least_1_teacher_referred_donor', 'fully_funded',
                                                                   'at_least_1_green_donation', 'great_chat',
                                                                   'three_or_more_non_teacher_referred_donors',
                                                                   'one_non_teacher_referred_donor_giving_100_plus',
                                                                   'donation_from_thoughtful_donor',
                                                                   'great_messages_proportion',
                                                                   'teacher_referred_count',
                                                                   'non_teacher_referred_count', 'is_exciting'])
        valid = pd.DataFrame(data_module.valid_dataset.X, columns=['at_least_1_teacher_referred_donor', 'fully_funded',
                                                                   'at_least_1_green_donation', 'great_chat',
                                                                   'three_or_more_non_teacher_referred_donors',
                                                                   'one_non_teacher_referred_donor_giving_100_plus',
                                                                   'donation_from_thoughtful_donor',
                                                                   'great_messages_proportion',
                                                                   'teacher_referred_count',
                                                                   'non_teacher_referred_count', 'is_exciting'])
        test = pd.DataFrame(data_module.test_dataset.X, columns=['at_least_1_teacher_referred_donor', 'fully_funded',
                                                                 'at_least_1_green_donation', 'great_chat',
                                                                 'three_or_more_non_teacher_referred_donors',
                                                                 'one_non_teacher_referred_donor_giving_100_plus',
                                                                 'donation_from_thoughtful_donor',
                                                                 'great_messages_proportion',
                                                                 'teacher_referred_count', 'non_teacher_referred_count',
                                                                 'is_exciting'])
        return thres_n, thres_ab, train, valid, test, None
    else:
        raise NotImplementedError


def get_summed_xGT(data_module, org_u, delta_u, data='loan'):
    if data == 'loan':
        u_pad = F.pad(delta_u, (3, 0, 0, 0))
        u_pad = u_pad.detach().cpu().numpy()
        x, u = data_module.train_dataset.get_interGT(u_g=(org_u + u_pad))
        return data_module.scaler.transform(x)
    elif data == 'adult':
        delta_u = delta_u.detach().cpu().numpy()
        u_pad = np.zeros((len(delta_u), 11))
        u_pad[:, 1] += delta_u[:, 0]
        u_pad[:, 4] += delta_u[:, 1]
        u_pad[:, 5] += delta_u[:, 2]
        x, u = data_module.train_dataset.get_interGT(u_g=(org_u + u_pad))
        x = data_module.scaler.transform(x)
        x = data_module.train_dataset.fill_up_with_zeros(x)[0][:, :-4]
        return torch.tensor(x).float()
    elif data == 'donors':
        return torch.zeros((len(org_u), 10))
    else:
        NotImplementedError


def prepare_adcar_training_data(df, y_pred, test_rc, data_module, data='loan'):
    if data == 'loan':
        df['pred'] = y_pred
        df['rc'] = 0
        df['rc'].iloc[-len(test_rc):] = test_rc
        df = df.loc[df['pred'] == 1]
        df = df.sample(len(df), random_state=42).reset_index(drop=True)
        train_index = int(0.8 * len(df))
        valid_index = int(0.9 * train_index)
        df_train = df.iloc[:valid_index]
        df_valid = df.iloc[valid_index:train_index]
        df_test = df.iloc[train_index:]
        df_test = df_test.loc[df_test['label'] == 1]
        x_train = data_module.scaler.transform(df_train.iloc[:, :7].values)
        u_train = df_train.iloc[:, 7:14].values
        x_valid = data_module.scaler.transform(df_valid.iloc[:, :7].values)
        u_valid = df_valid.iloc[:, 7:14].values
        x_test = data_module.scaler.transform(df_test.iloc[:, :7].values)
        u_test = df_test.iloc[:, 7:14].values
        rc_test = df_test.iloc[:, -1].values
        return x_train, u_train, x_valid, u_valid, x_test, u_test, df, rc_test
    elif data == 'adult':
        df['pred'] = y_pred
        df['rc'] = 0
        df['rc'].iloc[-len(test_rc):] = test_rc
        df = df.loc[df['pred'] == 1]
        df = df.sample(len(df), random_state=42).reset_index(drop=True)
        train_index = int(0.8 * len(df))
        valid_index = int(0.9 * train_index)
        df_train = df.iloc[:valid_index]
        df_valid = df.iloc[valid_index:train_index]
        df_test = df.iloc[train_index:]
        df_test = df_test.loc[df_test['label'] == 1]
        x_train = df_train.iloc[:, :25].values
        x_train = data_module.train_dataset.fill_up_with_zeros(x_train)[0]
        x_train = data_module.scaler.transform(x_train)[:, :-4]
        u_train = df_train.iloc[:, 25:-3].values

        x_valid = df_valid.iloc[:, :25].values
        x_valid = data_module.train_dataset.fill_up_with_zeros(x_valid)[0]
        x_valid = data_module.scaler.transform(x_valid)[:, :-4]
        u_valid = df_valid.iloc[:, 25:-3].values

        x_test = df_test.iloc[:, :25].values
        x_test = data_module.train_dataset.fill_up_with_zeros(x_test)[0]
        x_test = data_module.scaler.transform(x_test)[:, :-4]
        u_test = df_test.iloc[:, 25:-3].values
        rc_test = df_test.iloc[:, -1].values
        return x_train, u_train, x_valid, u_valid, x_test, u_test, df, rc_test

    elif data == 'donors':
        df['pred'] = y_pred
        df['rc'] = 0
        df = df.loc[df['pred'] == 1]
        df = df.sample(len(df), random_state=42).reset_index(drop=True)
        train_index = int(0.8 * len(df))
        valid_index = int(0.9 * train_index)
        df_train = df.iloc[:valid_index]
        df_valid = df.iloc[valid_index:train_index]
        df_test = df.iloc[train_index:]
        df_test = df_test.loc[df_test['is_exciting'] == 1]
        x_train = df_train.iloc[:, :-2].values
        x_train = data_module.train_dataset.fill_up_with_zeros(x_train)[0]
        x_train = data_module.scaler.transform(x_train)[:, :-1]
        u_train = torch.zeros((len(x_train), 11))

        x_valid = df_valid.iloc[:, :-2].values
        x_valid = data_module.train_dataset.fill_up_with_zeros(x_valid)[0]
        x_valid = data_module.scaler.transform(x_valid)[:, :-1]
        u_valid = torch.zeros((len(x_valid), 11))

        x_test = df_test.iloc[:, :-2].values
        x_test = data_module.train_dataset.fill_up_with_zeros(x_test)[0]
        x_test = data_module.scaler.transform(x_test)[:, :-1]
        u_test = torch.zeros((len(x_test), 11))
        rc_test = df_test.iloc[:, -1].values
        return x_train, u_train, x_valid, u_valid, x_test, u_test, df, rc_test
    else:
        NotImplementedError
