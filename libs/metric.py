import logging
import numpy as np
import os

def confusion_matrix(pred, label, num_classes):
    mask = (label >= 0) & (label < num_classes)
    conf_mat = np.bincount(num_classes * label[mask].astype(int) + pred[mask], minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf_mat

def evaluate(conf_mat):
    acc = np.diag(conf_mat).sum() / conf_mat.sum()
    acc_per_class = np.diag(conf_mat) / conf_mat.sum(axis=1)
    acc_cls = np.nanmean(acc_per_class)

    IoU = np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat))
    mean_IoU = np.nanmean(IoU)

    print_evaluate_results(conf_mat, IoU)

    # 求kappa
    pe = np.dot(np.sum(conf_mat, axis=0), np.sum(conf_mat, axis=1)) / (conf_mat.sum()**2)
    kappa = (acc - pe) / (1 - pe)
    return acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa

def save_log(prefix, output_dir):
    fmt = '%(asctime)s.%(msecs)03d %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    filename = os.path.join(output_dir, prefix+ '.log')
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=date_fmt,
                        filename=filename, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def print_evaluate_results(hist, iu):

    iu_false_positive = hist.sum(axis=1) - np.diag(hist)
    iu_false_negative = hist.sum(axis=0) - np.diag(hist)
    iu_true_positive = np.diag(hist)

    logging.info('label_id          IoU    Precision Recall')
    for idx, i in enumerate(iu):
        idx_string = "{:2d}".format(idx)
        iu_string = '{:5.2f}'.format(i * 100)
        precision = '{:5.2f}'.format(100 *iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_positive[idx]))
        recall = '{:5.2f}'.format(100 *iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_negative[idx]))


        logging.info('{}       {}  {}     {}'.format(idx_string,  iu_string, precision, recall))