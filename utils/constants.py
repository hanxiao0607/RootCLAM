# source code: https://github.com/psanch21/VACA/
class Cte:
    # equation-types
    LINEAR = "linear"
    NONLINEAR = 'non-linear'
    NONADDITIVE = 'non-additive'

    # Datasets
    SENS = 'sex'  # sensitive attribute for CF fairness

    TRIANGLE = 'triangle'  # a.k.a. Connected fork
    COLLIDER = 'collider'
    LOAN = 'loan'
    # One feature
    LOAN_AB_LAMOUNT = 'loan_ab_lamount'
    LOAN_AB_LDURATION = 'loan_ab_lduration'
    LOAN_AB_INCOME = 'loan_ab_income'
    LOAN_AB_SAVINGS = 'loan_ab_savings'
    # Two features
    LOAN_AB_LA_LD = 'loan_ab_la_ld'
    LOAN_AB_LA_I = 'loan_ab_la_i'
    LOAN_AB_LA_S = 'loan_ab_la_s'
    LOAN_AB_LD_I = 'loan_ab_ld_i'
    LOAN_AB_LD_S = 'loan_ab_ld_s'
    LOAN_AB_I_S = 'loan_ab_i_s'
    # Three features
    LOAN_AB_LA_LD_I = 'loan_ab_la_ld_i'
    LOAN_AB_LA_LD_S = 'loan_ab_la_ld_s'
    LOAN_AB_LA_I_S = 'Loan_ab_la_i_s'
    LOAN_AB_LD_I_S = 'loan_ab_ld_i_s'
    # Four features
    LOAN_AB_LA_LD_I_S = 'loan_ab_la_ld_i_s'

    ADULT = 'adult'
    # One feature
    ADULT_AB_AGE = 'adult_ab_age'
    ADULT_AB_EDU = 'adult_ab_edu'
    ADULT_AB_HOURS = 'adult_ab_hours'
    # Two features
    ADULT_AB_A_E = 'adult_ab_a_e'
    ADULT_AB_A_H = 'adult_ab_a_h'
    ADULT_AB_E_H = 'adult_ab_e_h'
    # Three features
    ADULT_AB_A_E_H = 'adult_ab_a_e_h'

    MGRAPH = 'mgraph'
    CHAIN = 'chain'
    GERMAN = 'german'
    DONORS = 'donors'

    ADULT_AB_LIST = ['adult_ab_age', 'adult_ab_edu', 'adult_ab_hours',
                  'adult_ab_a_e', 'adult_ab_a_h', 'adult_ab_e_h',
                  'adult_ab_a_e_h']

    ADULT_RC_LIST = [[0,1,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],
                     [0,1,0,0,1,0,0,0,0,0],[0,1,0,0,0,1,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],
                     [0,1,0,0,1,1,0,0,0,0]]

    LOAN_AB_LIST = ['loan_ab_lamount', 'loan_ab_lduration', 'loan_ab_income', 'loan_ab_savings',\
                  'loan_ab_la_ld', 'loan_ab_la_i', 'loan_ab_la_s', 'loan_ab_ld_i', 'loan_ab_ld_s', 'loan_ab_i_s',\
                  'loan_ab_la_ld_i', 'loan_ab_la_ld_s', 'Loan_ab_la_i_s', 'loan_ab_ld_i_s',\
                  'loan_ab_la_ld_i_s']

    LOAN_RC_LIST = [[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1],
                     [0,0,0,1,1,0,0],[0,0,0,1,0,1,0],[0,0,0,1,0,0,1],[0,0,0,0,1,1,0],[0,0,0,0,1,0,1],[0,0,0,0,0,1,1],
                     [0,0,0,1,1,1,0],[0,0,0,1,1,0,1],[0,0,0,1,0,1,1],[0,0,0,0,1,1,1],
                     [0,0,0,1,1,1,1]]


    DATASET_LIST = [COLLIDER,
                    TRIANGLE,
                    LOAN,
                    LOAN_AB_LAMOUNT,
                    LOAN_AB_LDURATION,
                    LOAN_AB_INCOME,
                    LOAN_AB_SAVINGS,
                    LOAN_AB_LA_LD,
                    LOAN_AB_LA_I,
                    LOAN_AB_LA_S,
                    LOAN_AB_LD_I,
                    LOAN_AB_LD_S,
                    LOAN_AB_I_S,
                    LOAN_AB_LA_LD_I,
                    LOAN_AB_LA_LD_S,
                    LOAN_AB_LA_I_S,
                    LOAN_AB_LD_I_S,
                    LOAN_AB_LA_LD_I_S,
                    MGRAPH,
                    CHAIN,
                    ADULT,
                    ADULT_AB_AGE,
                    ADULT_AB_EDU,
                    ADULT_AB_HOURS,
                    ADULT_AB_A_E,
                    ADULT_AB_A_H,
                    ADULT_AB_E_H,
                    ADULT_AB_A_E_H,
                    GERMAN,
                    DONORS]
    DATASET_LIST_TOY = [COLLIDER,
                        TRIANGLE,
                        LOAN,
                        MGRAPH,
                        CHAIN,
                        ADULT]
    # Models
    VACA = 'vaca'
    VACA_PIWAE = 'vaca_piwae'
    MCVAE = 'mcvae'
    CARELF = 'carefl'

    # Optimizers
    ADAM = 'adam'
    RADAM = 'radam'
    ADAGRAD = 'adag'
    ADADELTA = 'adad'
    RMS = 'rms'
    ASGD = 'asgd'

    # Scheduler
    STEP_LR = 'step_lr'
    EXP_LR = 'exp_lr'

    # Activation
    TAHN = 'tahn'
    RELU = 'relu'
    RELU6 = 'relu6'
    SOFTPLUS = 'softplus'
    RRELU = 'rrelu'
    LRELU = 'lrelu'
    ELU = 'elu'
    SELU = 'selu'
    SIGMOID = 'sigmoid'
    GLU = 'glu'
    IDENTITY = 'identity'

    # Distribution
    BETA = 'beta'
    CONTINOUS_BERN = 'cb'
    BERNOULLI = 'ber'
    GAUSSIAN = 'normal'
    CATEGORICAL = 'cat'
    EXPONENTIAL = 'exp'
    DELTA = 'delta'
