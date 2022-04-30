#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from models.ISTA_en_de import ISTA_en_de
from models.FISTA_en_de import FISTA_en_de
from models.LISTA_en_de import LISTA_en_de
from models.LISTA_CP_en_de import LISTA_CP_en_de
from models.TiLISTA_en_de import TiLISTA_en_de
from models.ALISTA_en_de import ALISTA_en_de
from models.LFISTA_en_de import LFISTA_en_de
from models.LAMP_en_de import LAMP_en_de
from models.GLISTA_en_de import GLISTA_en_de
from models.TsLISTA_en_de import TsLISTA_en_de
from models.LsLISTA_en_de import LsLISTA_en_de
from models.LM_FLISTA_en_de import LM_FISTA_en_de
from models.LM_LFISTA_cp_en_de import LM_FISTA_cp_en_de
from models.LM_ALISTA_en_de import LM_ALISTA_en_de


def model_choose(config, layers):
    if config.net == "ISTA_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "ISTA_en_de{is_en_de}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(is_en_de=config.is_en_de, T=config.T, lam=config.lam, classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "ISTA_en_de_T{T}_lam{lam}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(T=config.T, lam=config.lam, nums=config.numbers, is_en_de=config.is_en_de, has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = ISTA_en_de(layers, _A, config.is_unshared, config.numbers, config.is_en_de,
                               config.D_cl2num, config.classify_type,
                               config.classes, config.net, _lam=config.lam)

    elif config.net == "FISTA_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "FISTA_en_de{is_en_de}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(is_en_de=config.is_en_de, T=config.T, lam=config.lam, classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "FISTA_en_de_T{T}_lam{lam}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(T=config.T, lam=config.lam, nums=config.numbers, is_en_de=config.is_en_de, has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = FISTA_en_de(layers, _A, config.is_unshared, config.numbers, config.is_en_de,
                                config.D_cl2num, config.classify_type,
                                config.classes, config.net, _lam=config.lam)

    elif config.net == "LISTA_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "LISTA_en_de{is_en_de}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(is_en_de=config.is_en_de, T=config.T, lam=config.lam, classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "LISTA_en_de_T{T}_lam{lam}_na{is_na}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(T=config.T, lam=config.lam, nums=config.numbers, is_na=config.is_na, is_en_de=config.is_en_de,
                            has_noise=config.has_noise, SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = LISTA_en_de(layers, _A, config.is_unshared, config.numbers, config.is_na, config.is_en_de, config.D_cl2num,
                                config.classify_type, config.classes, config.net, _lam=config.lam)

    elif config.net == "LISTA_CP_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "LISTA_CP_en_de_ss{has_ss}{is_en_de}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, is_en_de=config.is_en_de, T=config.T, lam=config.lam,
                            classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "LISTA_CP_en_de_ss{has_ss}_T{T}_lam{lam}_na{is_na}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, T=config.T, lam=config.lam, nums=config.numbers, is_na=config.is_na,
                            is_en_de=config.is_en_de,
                            has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = LISTA_CP_en_de(layers, _A, config.is_unshared, config.numbers, config.has_ss, config.percent,
                                   config.max_percent,
                                   config.is_na, config.is_en_de, config.D_cl2num, config.classify_type,
                                   config.classes, config.net, _lam=config.lam)

    elif config.net == "TiLISTA_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "TiLISTA_en_de_ss{has_ss}{is_en_de}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, is_en_de=config.is_en_de, T=config.T, lam=config.lam,
                            classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "TiLISTA_en_de_ss{has_ss}_T{T}_lam{lam}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, T=config.T, lam=config.lam, nums=config.numbers, is_en_de=config.is_en_de,
                            has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = TiLISTA_en_de(layers, _A, config.is_unshared, config.numbers, config.has_ss, config.percent,
                                  config.max_percent,
                                  config.is_en_de, config.D_cl2num, config.classify_type,
                                  config.classes, config.net, _lam=config.lam)

    elif config.net == "ALISTA_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "ALISTA_en_de_ss{has_ss}{is_en_de}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, is_en_de=config.is_en_de, T=config.T, lam=config.lam,
                            classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "ALISTA_en_de_ss{has_ss}_T{T}_lam{lam}_na{is_na}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, T=config.T, lam=config.lam, nums=config.numbers, is_na=config.is_na,
                            is_en_de=config.is_en_de,
                            has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = ALISTA_en_de(layers, _A, config.W, config.is_unshared, config.numbers, config.has_ss, config.percent,
                                 config.max_percent,
                                 config.is_na, config.is_en_de, config.D_cl2num, config.classify_type,
                                 config.classes, config.net, _lam=config.lam)

    elif config.net == "LFISTA_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "LFISTA_en_de_ss{has_ss}{is_en_de}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, is_en_de=config.is_en_de, T=config.T, lam=config.lam,
                            classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "LFISTA_en_de_ss{has_ss}_T{T}_lam{lam}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, T=config.T, lam=config.lam, nums=config.numbers, is_en_de=config.is_en_de,
                            has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = LFISTA_en_de(layers, _A, config.is_unshared, config.numbers, config.is_en_de, config.D_cl2num,
                                 config.classify_type,
                                 config.classes, config.net, _lam=config.lam)

    elif config.net == "LAMP_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "LAMP_en_de_ss{has_ss}{is_en_de}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, is_en_de=config.is_en_de, T=config.T, lam=config.lam,
                            classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "LAMP_en_de_ss{has_ss}_T{T}_lam{lam}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, T=config.T, lam=config.lam, nums=config.numbers, is_en_de=config.is_en_de,
                            has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = LAMP_en_de(layers, _A, config.is_unshared, config.numbers, config.is_en_de, config.D_cl2num,
                               config.classify_type,
                               config.classes, config.net, _lam=config.lam)

    elif config.net == "GLISTA_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "GLISTA_en_de_ss{has_ss}{is_en_de}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, is_en_de=config.is_en_de, T=config.T, lam=config.lam,
                            classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "GLISTA_en_de_ss{has_ss}_T{T}_lam{lam}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, T=config.T, lam=config.lam, nums=config.numbers, is_en_de=config.is_en_de,
                            has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = GLISTA_en_de(layers, _A, config.is_unshared, config.numbers, config.has_ss, config.percent,
                                 config.max_percent,
                                 config.uf, config.alti, config.overshoot, config.gain, config.gain_fun,
                                 config.over_fun, config.both_gate, config.T_combine, config.T_middle,
                                 config.is_en_de, config.D_cl2num, config.classify_type, config.classes, config.net,
                                 _lam=config.lam)

    elif config.net == "TsLISTA_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "TsLISTA_en_de_ss{has_ss}{is_en_de}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, is_en_de=config.is_en_de, T=config.T, lam=config.lam,
                            classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "TsLISTA_en_de_ss{has_ss}_T{T}_lam{lam}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, T=config.T, lam=config.lam, nums=config.numbers, is_en_de=config.is_en_de,
                            has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = TsLISTA_en_de(layers, _A, config.is_unshared, config.numbers, config.has_ss, config.percent,
                                  config.max_percent,
                                  config.is_en_de, config.D_cl2num, config.classify_type,
                                  config.classes, config.net, _lam=config.lam)

    elif config.net == "LsLISTA_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "LsLISTA_en_de_ss{has_ss}{is_en_de}_nums{nums}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, is_en_de=config.is_en_de, nums=config.numbers, T=config.T, lam=config.lam,
                            classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "LsLISTA_en_de_ss{has_ss}_T{T}_lam{lam}_na{is_na}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, T=config.T, lam=config.lam, nums=config.numbers, is_na=config.is_na,
                            is_en_de=config.is_en_de,
                            has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = LsLISTA_en_de(layers, _A, config.is_unshared, config.numbers, config.has_ss, config.percent,
                                  config.max_percent,
                                  config.is_na, config.is_en_de, config.D_cl2num, config.classify_type,
                                  config.classes, config.net, config.lam)

    elif config.net == "LM_FISTA_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "LM_FISTA_en_de_ss{has_ss}{is_en_de}_nums{nums}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, is_en_de=config.is_en_de, nums=config.numbers, T=config.T, lam=config.lam,
                            classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "LM_FISTA_en_de_ss{has_ss}_T{T}_lam{lam}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, T=config.T, lam=config.lam, nums=config.numbers, is_en_de=config.is_en_de,
                            has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = LM_FISTA_en_de(layers, _A, config.is_unshared, config.numbers, config.has_ss, config.percent,
                                   config.max_percent,
                                   config.is_en_de, config.D_cl2num, config.classify_type,
                                   config.classes, config.net, _lam=config.lam)

    elif config.net == "LM_FISTA_cp_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "LM_FISTA_cp_en_de_ss{has_ss}{is_en_de}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, is_en_de=config.is_en_de, T=config.T, lam=config.lam,
                            classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "LM_FISTA_cp_en_de_ss{has_ss}_T{T}_lam{lam}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, T=config.T, lam=config.lam, nums=config.numbers, is_en_de=config.is_en_de,
                            has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = LM_FISTA_cp_en_de(layers, _A, config.is_unshared, config.numbers, config.has_ss, config.percent,
                                      config.max_percent,
                                      config.is_en_de, config.D_cl2num, config.classify_type,
                                      config.classes, config.net, _lam=config.lam)

    elif config.net == "LM_ALISTA_en_de":
        # encoder and decoder
        if config.is_en_de:
            config.model = (
                "LM_ALISTA_en_de_ss{has_ss}{is_en_de}_T{T}_lam{lam}_classify_type{classify_type}_feature_type{feature_type}_feature_shape{feature_shape}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, is_en_de=config.is_en_de, T=config.T, lam=config.lam,
                            classify_type=config.classify_type,
                            feature_type=config.feature_type, feature_shape=config.feature_shape, US=config.is_unshared,
                            exp=config.exp_id))
            _A = config.D
        # encoder
        else:
            config.model = (
                "LM_ALISTA_en_de_ss{has_ss}_T{T}_lam{lam}_na{is_na}_en_de{is_en_de}_nums{nums}_noise{has_noise}_SNR{SNR}_M{M}_N{N}_US{US}_exp{exp}"
                    .format(has_ss=config.has_ss, T=config.T, lam=config.lam, nums=config.numbers, is_na=config.is_na,
                            is_en_de=config.is_en_de,
                            has_noise=config.has_noise,
                            SNR=config.SNR, M=config.M, N=config.N,
                            US=config.is_unshared, exp=config.exp_id))
            _A = config.A
        model_out = LM_ALISTA_en_de(layers, _A, config.W, config.is_unshared, config.numbers, config.has_ss, config.percent,
                                    config.max_percent,
                                    config.is_na, config.is_en_de, config.D_cl2num, config.classify_type,
                                    config.classes, config.net, _lam=config.lam)

    if config.is_en_de:
        setattr(config, 'fr_exp_dir',
                os.path.join(config.exp_base, config.data_type + config.feature_type + str(config.feature_shape)))
        setattr(config, 'fr_res_dir',
                os.path.join(config.res_base, config.data_type + config.feature_type + str(config.feature_shape)))

        setattr(config, 'fr_model_fn', os.path.join(config.fr_exp_dir, config.model))
        setattr(config, 'fr_res_fn', os.path.join(config.fr_res_dir, config.model))
        if not os.path.exists(config.fr_exp_dir):
            os.mkdir(config.fr_exp_dir)
        if not os.path.exists(config.fr_res_dir):
            os.mkdir(config.fr_res_dir)
        if not config.test:
            if not os.path.exists(config.fr_model_fn):
                os.mkdir(config.fr_model_fn)
    else:
        setattr(config, 'exp_dir', os.path.join(config.exp_base, "en_de" + str(config.has_noise) + str(config.SNR)))
        setattr(config, 'res_dir', os.path.join(config.res_base, "en_de" + str(config.has_noise) + str(config.SNR)))

        setattr(config, 'model_fn', os.path.join(config.exp_dir, config.model))
        setattr(config, 'res_fn', os.path.join(config.res_dir, config.model))
        if not os.path.exists(config.exp_dir):
            os.mkdir(config.exp_dir)
        if not os.path.exists(config.res_dir):
            os.mkdir(config.res_dir)
        if not config.test:
            if not os.path.exists(config.model_fn):
                os.mkdir(config.model_fn)

    return model_out
