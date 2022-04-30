#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import datetime
import sys, os, gc
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import get_config
from model_init import model_choose


def get_dataset_fr(config,bs,samples):
    ytr_, xtr_ = get_fr_test_data(config.ytr_, config.ltr_)
    tr_dataset = tf.data.Dataset.from_tensor_slices((tf.transpose(ytr_), xtr_))
    tr_dataset = tr_dataset.cache()
    tr_dataset = tr_dataset.map(lambda src, tgt: (src, tgt), num_parallel_calls=8)
    tr_dataset = tr_dataset.batch(bs)
    tr_dataset = tr_dataset.shuffle(samples)
    tr_dataset = tr_dataset.prefetch(bs)
    return tr_dataset


# k-layered model easy to overfit and get data for once
def fr_train(config, _model, _y_val, _x_val, _y_te, _x_te, _lay, _new_varis, _lay_results):
    # train and val
    # config.init_lr = 5e-4
    # config.lr_decay = [1, 0.1]
    lrs = [config.init_lr * decay for decay in config.lr_decay]

    train_varis = _new_varis
    for lr in lrs:
        if len(train_varis) == 0:
            train_varis = _model.trainable_variables
            continue
        print(_lay + 1, "Layer Training Variables numbers = {:2d}".format(len(train_varis)))
        varis_names = []
        [(varis_names.append(v.name)) for v in train_varis]
        print(varis_names)
        print("Learning Rate = {:6f}".format(lr))
        valloss_his = []  # collect all the validation results
        valacc_his = []  # collect all the validation results
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        cout = 0
        for epoch in range(config.epochs):
            _tr_dataset = get_dataset_fr(config, config.tbs, config.tr_samples)
            for (y_tr, x_tr) in _tr_dataset:
                y_tr = tf.cast(tf.transpose(y_tr), dtype=tf.float32)  # (M, batch_size)
                # y_val = tf.cast(tf.transpose(y_val), dtype=tf.float32)  # (M, batch_size)
                cout += 1
                with tf.GradientTape() as tape:
                    # train
                    pred_xtr, pred_resi_tr = _model(y_tr)  ##(N, batch_size), (batch_size, classes)

                    # train loss
                    # tr_label = tf.one_hot(x_tr, depth=pred_resi_tr.shape[0]) ##(batch_size, classes)
                    tr_cr_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(x_tr, pred_resi_tr))
                    # tr_cr_loss = -tf.reduce_sum(tr_label * tf.math.log(tf.transpose(pred_resi_tr)))
                    tr_n2_loss = tf.nn.l2_loss(y_tr - tf.matmul(config.D, pred_xtr)) * 2 / config.tbs
                    tr_n1_loss = tf.reduce_mean(config.lam * tf.linalg.norm(pred_xtr, ord=1, axis=0))
                    tr_loss = tr_cr_loss + tr_n2_loss + tr_n1_loss
                    # tr_loss = tr_cr_loss
                    tr_acc = tf.reduce_mean(tf.cast(tf.equal(x_tr, tf.argmax(pred_resi_tr, axis=1)), dtype=tf.float32))

                    # val loss
                    pred_xval, pred_resi_val = _model(_y_val)  ##(N, batch_size), (classes, batch_size)
                    val_cr_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(_x_val, pred_resi_val))
                    # val_label = tf.one_hot(_x_val, depth=pred_resi_val.shape[0])  ##(batch_size, classes)
                    # val_cr_loss = -tf.reduce_sum(val_label * tf.math.log(tf.transpose(pred_resi_val)))
                    val_n2_loss = tf.nn.l2_loss(_y_val - tf.matmul(config.D, pred_xval)) * 2 / config.vbs
                    val_n1_loss = tf.reduce_mean(config.lam * tf.linalg.norm(pred_xval, ord=1, axis=0))
                    val_loss = val_cr_loss + val_n2_loss + val_n1_loss
                    # val_loss = val_cr_loss
                    val_acc = tf.reduce_mean(
                        tf.cast(tf.equal(_x_val, tf.argmax(pred_resi_val, axis=1)), dtype=tf.float32))
                grads = tape.gradient(tr_loss, train_varis)
                optimizer.apply_gradients(zip(grads, train_varis))

                valloss_his.append(val_loss)
                valacc_his.append(val_acc)
                sys.stdout.write(
                    "\r| i={i:<7d} | tr_loss={tr_loss:.6f} | "
                    "tr_acc={tr_acc:.6f} | val_loss ={val_loss:.6f} | "
                    "val_acc={val_acc:.6f} | best_val_acc={best_val_acc:.6f} | best_val_loss={best_val_loss:.6f} )" \
                        .format(i=cout, tr_loss=tr_loss, tr_acc=tr_acc, val_loss=val_loss, val_acc=val_acc,
                                best_val_acc=max(valacc_his), best_val_loss=min(valloss_his)))
                sys.stdout.flush()

                if cout % 50 == 0:
                    print(" ")

            curr_his = len(valloss_his)  # the length of validation data
            max_his = valloss_his.index(min(valloss_his))  # the postion of best result
            len_mis = curr_his - max_his - 1  # get the length of loss has been stopping decrease

            if len_mis >= config.better_wait:  # if wait epochs is more than max wait
                print('')
                train_varis = _model.trainable_variables
                break

    # save variables of K-layered model
    save_trainable_variables(config, _model.trainable_variables, config.fr_save_path)

    pred_xte, pred_resi_te = _model(_y_te)  ##(N, batch_size), (classes, batch_size)
    te_cr_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(_x_te, pred_resi_te))
    te_n2_loss = tf.nn.l2_loss(_y_te - tf.matmul(config.D, pred_xte)) * 2 / pred_xte.shape[1]
    te_n1_loss = tf.reduce_mean(config.lam * tf.linalg.norm(pred_xte, ord=1, axis=0))
    te_loss = te_cr_loss + te_n2_loss + te_n1_loss
    # te_loss = te_cr_loss
    te_acc = tf.reduce_mean(tf.cast(tf.equal(_x_te, tf.argmax(pred_resi_te, axis=1)), dtype=tf.float32))
    print("te_loss={te_loss:.6f}dB | te_acc={te_acc:.4f}db )".format(te_loss=te_loss, te_acc=te_acc))
    # save test results
    _lay_results[str(_lay)] = te_acc
    return _lay_results


def run_fr_train(config):
    # set config file
    set_fr_data_file(config)
    '''
    if not os.path.exists(config.tr_tfrecord_file):
        write_fr_dataset(config, "train", config.tr_tfrecord_file)
        write_fr_dataset(config, "val", config.val_tfrecord_file)
        write_fr_dataset(config, "test", config.te_tfrecord_file)
    # get TFRecord dataset
    #tr_dataset = get_fr_dataset(config.tr_tfrecord_file, config.tbs, config.tr_samples)
    #val_dataset = get_fr_dataset(config.val_tfrecord_file, config.vbs, config.val_samples)
    #tr_dataset = tr_dataset.repeat()
    #val_dataset = val_dataset.repeat()
    '''
    y_val, x_val = get_fr_test_data(config.yval_, config.lval_)
    y_te, x_te = get_fr_test_data(config.yte_, config.lte_)

    # train
    # save every layer variables and test results
    lay_results = dict()
    # for lay in layers:
    last_varis_name = []
    start_time_total = datetime.datetime.now()
    end_time_total = datetime.datetime.now()

    # for lay in range(config.T):
    for lay in range(config.T):
        # get k-layered model
        model_obj = model_choose(config, lay + 1)  # initiation
        # make sure the current layer model need to be trained.
        config.fr_save_path = os.path.join(config.fr_model_fn, str(lay) + ".npy")
        if os.path.exists(config.fr_save_path):
            print("layer", lay + 1, "already be trained, turn to next layer !")
            continue
        # restore variables
        last_varis_name = restore_varis(config, lay, last_varis_name, model_obj)
        # trainable parameters
        new_varis_tuple = tuple([var for var in model_obj.trainable_variables
                                 if var.name not in last_varis_name])
        # training
        start_time = datetime.datetime.now()
        results = fr_train(config, model_obj, y_val, x_val, y_te, x_te, lay, new_varis_tuple, lay_results)
        end_time = datetime.datetime.now()
        print(lay + 1, "layer Time:", end_time - start_time)
        if lay == 0:
            start_time_total = start_time
        if lay == config.T - 1:
            end_time_total = end_time
        # save results
        np.save(config.fr_res_fn, results)
        del model_obj
        gc.collect()
    total_time = end_time_total - start_time_total
    print("Time:", total_time)

def get_fr_test_data(_y_path, _x_path):
    # get test data
    input_test = tf.constant(np.load(_y_path, allow_pickle=True), tf.float32)
    output_test = tf.constant(np.load(_x_path, allow_pickle=True))
    return input_test, output_test


def set_fr_data_file(config):
    # create database
    de_database = os.path.join(config.data_base, "de" + config.data_type)
    setattr(config, 'pre_test_data', os.path.join(de_database, "pre" + str(config.feature_shape)))
    setattr(config, 'de_test_data', os.path.join(de_database, str(config.feature_shape)))
    setattr(config, 'class2num', os.path.join(de_database, "class2num.npy"))
    setattr(config, 'D', None)
    setattr(config, 'save_D', os.path.join(de_database, "D" + str(config.feature_shape) + ".npy"))
    setattr(config, 'D_cl2num', None)
    setattr(config, 'D_cl2num_path', os.path.join(de_database, "D_label2num.npy"))

    setattr(config, 'tr_tfrecord_file', os.path.join(config.de_test_data, 'train.tfrecords'))
    setattr(config, 'val_tfrecord_file', os.path.join(config.de_test_data, 'val.tfrecords'))
    setattr(config, 'fr_save_path', None)

    setattr(config, 'ytr_', os.path.join(de_database, "train" + str(config.feature_shape) + ".npy"))
    setattr(config, 'ltr_', os.path.join(de_database, "train_label.npy"))
    setattr(config, 'yval_', os.path.join(de_database, "val" + str(config.feature_shape) + ".npy"))
    setattr(config, 'lval_', os.path.join(de_database, "val_label.npy"))
    setattr(config, 'yte_', os.path.join(de_database, "test" + str(config.feature_shape) + ".npy"))
    setattr(config, 'lte_', os.path.join(de_database, "test_label.npy"))

    if not os.path.exists(de_database):
        os.mkdir(de_database)
    if not os.path.exists(config.pre_test_data):
        os.mkdir(config.pre_test_data)
    if not os.path.exists(config.de_test_data):
        os.mkdir(config.de_test_data)

    if not os.path.exists(config.save_D):
        print("No test dataset, please create!!!")
        return
    config.D = (np.load(config.save_D, allow_pickle=True)).astype(np.float32)
    config.D_cl2num = np.load(config.D_cl2num_path, allow_pickle=True)
    config.D_cl2num = config.D_cl2num.tolist()


def run_fr_test(config):
    # set data save path
    set_fr_data_file(config)
    input_test, output_test = get_fr_test_data(config.yte_, config.lte_)
    lay_results = dict()

    if config.net == "ISTA_en_de" or config.net == "FISTA_en_de" or config.net == "oracle_ISTA_fr":
        for i in range(config.T):
            model = model_choose(config, i + 1)  # initiation
            # test
            pred_xte, pred_resi_te = model(input_test)  ##(N, batch_size), (classes, batch_size)
            te_cr_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output_test, pred_resi_te))
            # te_label = tf.one_hot(output_test, depth=pred_resi_te.shape[0])  ##(batch_size, classes)
            # te_cr_loss = -tf.reduce_sum(te_label * tf.math.log(tf.transpose(pred_resi_te)))
            te_n2_loss = tf.nn.l2_loss(input_test - tf.matmul(config.D, pred_xte)) * 2 / pred_xte.shape[1]
            te_n1_loss = tf.reduce_mean(config.lam * tf.linalg.norm(pred_xte, ord=1, axis=0))
            te_loss = te_cr_loss + te_n2_loss + te_n1_loss
            # te_loss = te_cr_loss
            te_acc = tf.reduce_mean(tf.cast(tf.equal(output_test, tf.argmax(pred_resi_te, axis=1)), dtype=tf.float32))
            print("te_loss={te_loss:.6f} | te_acc={te_acc:.6f} )".format(te_loss=te_loss, te_acc=te_acc))
            lay_results[i] = te_acc
        np.save(config.fr_res_fn, lay_results)

    else:
        for i in range(config.T):
            model = model_choose(config, i + 1)  # initiation

            # Get variables
            vars_fold = np.load(config.fr_model_fn + "/" + str(i) + ".npy", allow_pickle=True).item()

            for k, d in vars_fold.items():
                for v in model.trainable_variables:
                    if k == v.name:  # A*
                        print("restore", k)
                        v.assign(d)
            # test
            pred_xte, pred_resi_te = model(input_test)  ##(N, batch_size), (classes, batch_size)
            te_cr_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output_test, pred_resi_te))
            # te_label = tf.one_hot(output_test, depth=pred_resi_te.shape[0])  ##(batch_size, classes)
            # te_cr_loss = -tf.reduce_sum(te_label * tf.math.log(tf.transpose(pred_resi_te)))
            te_n2_loss = tf.nn.l2_loss(input_test - tf.matmul(config.D, pred_xte)) * 2 / pred_xte.shape[1]
            te_n1_loss = tf.reduce_mean(config.lam * tf.linalg.norm(pred_xte, ord=1, axis=0))
            te_loss = te_cr_loss + te_n2_loss + te_n1_loss
            # te_loss = te_cr_loss
            te_acc = tf.reduce_mean(tf.cast(tf.equal(output_test, tf.argmax(pred_resi_te, axis=1)), dtype=tf.float32))
            print("te_loss={te_loss:.6f} | te_acc={te_acc:.6f} )".format(te_loss=te_loss, te_acc=te_acc))
            lay_results[i] = te_acc

            np.save(config.fr_res_fn, lay_results)


def save_trainable_variables(config, _variables, _path):
    save = dict()
    for v in _variables:
        save[str(v.name)] = v

    np.save(_path, save)


def save_results(config, _lay_results):
    if not config.res_fn.endswith(".npy"):
        config.res_fn += ".npy"
    if os.path.exists(config.res_fn):
        sys.st.write('Already exist !\n')
    np.save(config.res_fn, _lay_results)


def get_loss(_model, _labels, _inputs):
    predict_x = _model(_inputs)
    # NMSE
    nmse_denom_ = tf.nn.l2_loss(_labels)
    loss_ = tf.nn.l2_loss(predict_x - _labels)
    nmse_ = loss_ / nmse_denom_
    return loss_, nmse_


def restore_varis(config, _lay, _last_varis_name, _model_obj):
    if config.is_en_de:
        last_varis_path = os.path.join(config.fr_model_fn, str(_lay - 1) + ".npy")
    else:
        last_varis_path = os.path.join(config.model_fn, str(_lay - 1) + ".npy")
    if os.path.exists(last_varis_path):
        vars_fold = np.load(last_varis_path, allow_pickle=True).item()
        for k, d in vars_fold.items():
            _last_varis_name.append(k)
            for v in _model_obj.trainable_variables:
                if k == v.name:
                    print("restore:", k)
                    v.assign(d)
    return _last_varis_name


def load_test_data(config):
    """
        load existed test data
    """
    print(config.y_test)
    if not os.path.exists(config.y_test):
        print("No test dataset, please create!!!")
        return

    A = np.load(config.save_A, allow_pickle=True)
    config.A = A

    # get test data
    input_test = np.load(config.y_test, allow_pickle=True)
    output_test = np.load(config.x_test, allow_pickle=True)
    return input_test, output_test


def run_simu_test(config):
    # set data save path
    set_data_file(config)
    input_test, output_test = load_test_data(config)
    lay_results = dict()
    if config.net == "ISTA_en_de" or config.net == "FISTA_en_de" or config.net == "oracle_ISTA":
        for i in range(config.T):
            model = model_choose(config, i + 1)  # initiation
            telo, tenmse = get_loss(model, output_test, input_test)
            tedb = 10. * np.log10(tenmse)
            print(tedb)
            lay_results[i] = tedb
        save_results(config, lay_results)
    else:
        for i in range(config.T):
            model = model_choose(config, i + 1)  # initiation
            # Get variables
            vars_fold = np.load(config.model_fn + "/" + str(i) + ".npy", allow_pickle=True).item()
            for k, d in vars_fold.items():
                for v in model.trainable_variables:
                    if k == v.name:  # A*
                        print("restore", k)
                        v.assign(d)
            telo, tenmse = get_loss(model, output_test, input_test)
            tedb = 10. * np.log10(tenmse)
            print(tedb)
            lay_results[i] = tedb
        np.save(config.res_fn, lay_results)


# k-layered model easy to overfit and get data for once
def simu_train(config, _model, _y_te, _x_te, _lay, _cur_new_varis, _lay_results):
    # train and val
    lrs = [config.init_lr * decay for decay in config.lr_decay]
    train_varis = _cur_new_varis
    for lr in lrs:
        if len(train_varis) == 0:
            train_varis = _model.trainable_variables
            continue
        print(_lay + 1, "Layer Training Variables numbers = {:2d}".format(len(train_varis)))
        print([v.name for v in train_varis])
        print("Learning Rate = {:6f}".format(lr))
        valdb_his = []  # collect all the validation results
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        for epoch in range(config.epochs):
            # get endless data each time get 6400
            y_tr, x_tr = generate_data(config, config.tbs)
            y_val, x_val = generate_data(config, config.vbs)
            with tf.GradientTape() as tape:
                # train
                pred_xtr = _model(y_tr)
                # train NMSE
                tr_nmse_denom = tf.nn.l2_loss(x_tr)
                tr_loss = tf.nn.l2_loss(pred_xtr - x_tr)
                tr_nmse = tr_loss / tr_nmse_denom
                tr_db = 10. * np.log10(tr_nmse)

                # val
                pred_xval = _model(y_val)
                val_nmse_denom = tf.nn.l2_loss(x_val)
                val_loss = tf.nn.l2_loss(pred_xval - x_val)
                val_nmse = val_loss / val_nmse_denom
                val_db = 10. * np.log10(val_nmse)
                valdb_his.append(val_db)

            grads = tape.gradient(tr_loss, train_varis)
            optimizer.apply_gradients(zip(grads, train_varis))
            sys.stdout.write(
                "\r| i={i:<7d} | loss_tr={loss_tr:.6f} | "
                "db_tr={db_tr:.6f}dB | loss_val ={loss_val:.6f} | "
                "db_val={db_val:.6f}dB | best_val_db={best_val_db:.6f}db )" \
                    .format(i=epoch, loss_tr=tr_loss, db_tr=tr_db, loss_val=val_loss, db_val=val_db,
                            best_val_db=min(valdb_his)))
            sys.stdout.flush()
            if epoch % 1000 == 0:
                print(" ")

            curr_his = len(valdb_his)  # the length of validation data
            min_his = valdb_his.index(min(valdb_his))  # the postion of best result
            len_mis = curr_his - min_his - 1  # get the length of loss has been stopping decrease

            if len_mis >= config.better_wait:  # if wait epochs is more than max wait
                print('')
                train_varis = _model.trainable_variables
                break

    # save variables of K-layered model
    save_trainable_variables(config, _model.trainable_variables, config.save_path)
    # test
    pred_xte = _model(_y_te)
    te_nmse_denom = tf.nn.l2_loss(_x_te)
    te_loss = tf.nn.l2_loss(pred_xte - _x_te)
    te_nmse = te_loss / te_nmse_denom
    te_db = 10. * np.log10(te_nmse)
    print("loss_te={loss_te:.6f} | db_te={db_te:.6f}dB".format(loss_te=te_loss, db_te=te_db))
    '''
    # log test nmse
    if config.log:
        with summary_writer.as_default():
            tf.summary.scalar("test nmse" + config.modelfn, tedb, step=lay)
    '''
    # save test results
    _lay_results[str(_lay)] = te_db
    return _lay_results


def run_simu_train(config):
    """
        input_test: Y
        output_test: X
    """
    # set data save path
    set_data_file(config)
    # get test data
    input_test, output_test = load_test_data(config)
    # train
    # save every layer variables and test results
    lay_results = {}
    # for lay in layers:
    last_varis_name = []
    start_time_total = datetime.datetime.now()
    end_time_total = datetime.datetime.now()

    for lay in range(config.T):
        # get k-layered model
        model_obj = model_choose(config, lay + 1)  # initiation
        # make sure the current layer model need to be trained.
        config.save_path = os.path.join(config.model_fn, str(lay) + ".npy")
        if os.path.exists(config.save_path):
            print("layer", lay + 1, "already be trained, turn to next layer !")
            continue
        # restore variables
        last_varis_name = restore_varis(config, lay, last_varis_name, model_obj)
        # trainable parameters
        new_varis_tuple = tuple([var for var in model_obj.trainable_variables
                                 if var.name not in last_varis_name])
        # training
        start_time = datetime.datetime.now()
        results = simu_train(config, model_obj, input_test, output_test, lay, new_varis_tuple, lay_results)
        end_time = datetime.datetime.now()
        print(lay + 1, "layer Time:", end_time - start_time)
        if lay == 0:
            start_time_total = start_time
        if lay == config.T - 1:
            end_time_total = end_time

        # save results
        np.save(config.res_fn, results)
        del model_obj
        gc.collect()
    total_time = end_time_total - start_time_total
    print("Time:", total_time)


def generate_data(config, _samples):
    """
        y = Ax
        _samples : Numbers of Dataset :
          train data: 10000
          validation data: vbs: 1000
          test data: tbs: 1000

        (m, n) : are the shape of Dictionary A
          (m, n) = (250, 500)

        Dictionary A is sampled from standard Gaussian distribution
          Aij ~ N(0, 1/m)
          and each column is normalized to unit length

        supp: is the supports of x_true
        snr: is SNR
        input_data: y
        output_data: x
    """

    # generate synthetic data for demo
    m, n = config.M, config.N
    # Matrix A just need to generate for one time!
    if config.task_type == "gd":
        if not os.path.exists(config.save_A):
            # use a random matrix as a basis (design matrix)
            A = np.random.normal(scale=1.0 / np.sqrt(m), size=(m, n)).astype(np.float32)
            if config.is_col_normalized:
                A = A / tf.linalg.norm(A, axis=0)
            np.save(config.save_A, A)
        else:
            A = np.load(config.save_A, allow_pickle=True)

        config.A = A
    _A = config.A

    snr = config.SNR
    supp = tf.random.uniform((n, _samples), minval=0.0, maxval=1.0)
    supp = tf.cast((supp <= config.support), tf.float32)

    x_true = tf.random.normal((n, _samples), dtype=tf.float32)
    x_true = x_true * supp

    y_ = tf.matmul(_A, x_true)

    if config.has_noise:
        # add noise
        """Add noise with SNR."""
        std = (tf.sqrt(tf.nn.moments(y_, axes=[0], keepdims=True)[1]) * tf.math.pow(10.0, -snr / 10.0))
        tr_noise = tf.random.normal(y_.shape, stddev=std, dtype=tf.float32)
        y_ = y_ + tr_noise

    input_data = tf.cast(y_, dtype=tf.float32)
    output_data = tf.cast(x_true, dtype=tf.float32)
    return input_data, output_data


def set_data_file(config):
    """
        data_base: ./data/en_deMN, help="folder for different dimensions of data."
        test_data: ./data/en_deMN/NoiseSNR, help="folder for different settings of noise".
        save_A: ./data/en_deMN/NoiseSNR/A.npy, help="matrix A".
        A: None, matrix A
        y_test: ./data/en_deMN/NoiseSNR/y_test.npy, help="test data of y".
        x_test: ./data/en_deMN/NoiseSNR/x_test.npy, help="test data of x".
        save_path: None, path for saving layer parameters
        W: None, help="pre-calculated matrix W".
        save_W: ./data/en_deMN/NoiseSNR/W.npy, help="matrix W".
    """
    # create data_base
    data_base = os.path.join(config.data_base, "en_de" + str(config.M) + str(config.N))
    #
    setattr(config, 'test_data', os.path.join(data_base, str(config.has_noise) + str(config.SNR)))
    setattr(config, 'save_A', os.path.join(config.test_data, "A.npy"))
    setattr(config, 'A', None)
    setattr(config, 'y_test', os.path.join(config.test_data, "y_test.npy"))
    setattr(config, 'x_test', os.path.join(config.test_data, "x_test.npy"))
    # setattr(config, 'new_save_path', None)
    setattr(config, 'save_path', None)

    if config.net == "ALISTA_en_de" or config.net == "LM_ALISTA_en_de":
        setattr(config, 'W', None)
        setattr(config, 'save_W', os.path.join(config.test_data, "W.npy"))
        if not os.path.exists(config.save_W):
            print("Please create W.npy!!!")
        else:
            config.W = np.load(config.save_W, allow_pickle=True)

    if not os.path.exists(data_base):
        os.mkdir(data_base)
    if not os.path.exists(config.test_data):
        os.mkdir(config.test_data)


def create_test_data(config):
    """
        generate A and X and Y, this only generate for one time
    """
    # set data save path
    set_data_file(config)
    # get test data, generate A and test X and Y
    input_data, output_data = generate_data(config, config.vbs)

    # save dataset
    if not os.path.exists(config.y_test):
        np.save(config.y_test, input_data)
    if not os.path.exists(config.x_test):
        np.save(config.x_test, output_data)


def test(config):
    """
        "ste": simulated test
        "fte": face recognition test
    """
    if config.task_type == "ste":  # when get layer result failed.
        run_simu_test(config)
    elif config.task_type == "fte":  # Face recognition
        run_fr_test(config)


def train(config):
    """
        "gd": get test data
        "str": simulated train
        "ftr": face recognition train
    """
    # get test data
    if config.task_type == "gd":
        create_test_data(config)
    # simulated training
    elif config.task_type == "str":
        # train and test
        run_simu_train(config)
    # Face recognition train
    elif config.task_type == "ftr":
        run_fr_train(config)


def gpu_avaliable():
    """
        gpu setting
    """
    print(tf.__version__)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6976)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    tf.config.set_soft_device_placement(True)


def main(**kwargs):
    # set gpu
    gpu_avaliable()

    # parse configuration
    config, _ = get_config()
    # train or test
    if config.test:
        test(config)
    else:
        train(config)


if __name__ == '__main__':
    main()
