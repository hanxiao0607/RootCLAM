# source code: https://github.com/psanch21/VACA/
import os

from datasets._heterogeneous import HeterogeneousSCM
from utils.distributions import *
import pandas as pd


class DonorsSCM(HeterogeneousSCM):

    def __init__(self, root_dir,
                 split: str = 'train',
                 num_samples_tr: int = 10000,
                 lambda_: float = 0.05,
                 transform=None,
                 ):
        assert split in ['train', 'valid', 'test']

        self.name = 'donors'
        self.split = split
        self.num_samples_tr = num_samples_tr
        self.df = pd.read_csv(os.path.join(root_dir, 'datasets', 'donors.csv'), header=0, index_col=0)
        self._load_data()

        super().__init__(root_dir=root_dir,
                         transform=transform,
                         nodes_to_intervene=['great_messages_proportion', 'teacher_referred_count',
                                             'non_teacher_referred_count'],
                         structural_eq=None,
                         noises_distr=None,
                         nodes_list=['at_least_1_teacher_referred_donor', 'fully_funded',
                                     'at_least_1_green_donation', 'great_chat',
                                     'three_or_more_non_teacher_referred_donors',
                                     'one_non_teacher_referred_donor_giving_100_plus',
                                     'donation_from_thoughtful_donor', 'great_messages_proportion',
                                     'teacher_referred_count', 'non_teacher_referred_count', 'is_exciting'],
                         adj_edges={'at_least_1_teacher_referred_donor': ['is_exciting'],
                                    'fully_funded': ['is_exciting'],
                                    'at_least_1_green_donation': ['donation_from_thoughtful_donor', 'is_exciting'],
                                    'great_chat': ['donation_from_thoughtful_donor', 'is_exciting'],
                                    'three_or_more_non_teacher_referred_donors': ['is_exciting'],
                                    'one_non_teacher_referred_donor_giving_100_plus': ['is_exciting'],
                                    'donation_from_thoughtful_donor': ['is_exciting'],
                                    'great_messages_proportion': ['three_or_more_non_teacher_referred_donors',
                                                                  'fully_funded', 'at_least_1_green_donation',
                                                                  'great_chat', 'donation_from_thoughtful_donor'],
                                    'teacher_referred_count': ['at_least_1_teacher_referred_donor',
                                                               'great_messages_proportion', 'fully_funded',
                                                               'at_least_1_green_donation'],
                                    'non_teacher_referred_count': ['one_non_teacher_referred_donor_giving_100_plus',
                                                                   'three_or_more_non_teacher_referred_donors',
                                                                   'fully_funded', 'at_least_1_green_donation'],
                                    'is_exciting': []
                                    },
                         lambda_=lambda_,
                         )

    def _load_data(self):
        self.df = self.df.dropna()
        cleanup_dic = {'is_exciting': {'f': 1, 't': 0},
                       'at_least_1_teacher_referred_donor': {'f': 0, 't': 1},
                       'fully_funded': {'f': 0, 't': 1},
                       'at_least_1_green_donation': {'f': 0, 't': 1},
                       'great_chat': {'f': 0, 't': 1},
                       'three_or_more_non_teacher_referred_donors': {'f': 0, 't': 1},
                       'one_non_teacher_referred_donor_giving_100_plus': {'f': 0, 't': 1},
                       'donation_from_thoughtful_donor': {'f': 0, 't': 1}}
        self.df = self.df.replace(cleanup_dic)
        self.df = self.df[['at_least_1_teacher_referred_donor', 'fully_funded',
                           'at_least_1_green_donation', 'great_chat',
                           'three_or_more_non_teacher_referred_donors',
                           'one_non_teacher_referred_donor_giving_100_plus',
                           'donation_from_thoughtful_donor', 'great_messages_proportion',
                           'teacher_referred_count', 'non_teacher_referred_count', 'is_exciting']]
        self.df_norm = self.df.loc[self.df['is_exciting'] == 0]
        self.df_norm = self.df_norm.sample(len(self.df_norm), random_state=42).reset_index(drop=True)
        self.df_ab = self.df.loc[self.df['is_exciting'] == 1]
        self.df_ab = self.df_ab.sample(int((len(self.df_norm) - self.num_samples_tr) * 0.1),
                                       random_state=42).reset_index(drop=True)

    @property
    def likelihoods(self):
        likelihoods_tmp = []

        for i, lik_name in enumerate(self.nodes_list):  # Iterate over nodes
            if self.nodes_list[i] in ['is_exciting', 'at_least_1_teacher_referred_donor', 'fully_funded',
                                      'at_least_1_green_donation', 'great_chat',
                                      'three_or_more_non_teacher_referred_donors',
                                      'one_non_teacher_referred_donor_giving_100_plus',
                                      'donation_from_thoughtful_donor']:
                dim = 1
                lik_name = 'b'
            else:
                dim = 1
                lik_name = 'd'
            likelihoods_tmp.append([self._get_lik(lik_name,
                                                  dim=dim,
                                                  normalize='dim')])
        return likelihoods_tmp

    def _create_data(self):
        X_norm = self.df_norm.values
        X_ab = self.df_ab.values
        if self.split == 'train':
            self.X = X_norm[:int(self.num_samples_tr * 0.8)]
        elif self.split == 'valid':
            self.X = X_norm[int(self.num_samples_tr * 0.8):self.num_samples_tr]
        elif self.split == 'test':
            self.X = np.concatenate((X_norm[self.num_samples_tr:], X_ab), axis=0)
        self.U = np.zeros([self.X.shape[0], 1])

    def node_is_image(self):
        return [False for _ in range(len(self.nodes_list))]
