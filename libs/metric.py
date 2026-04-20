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

    # 计算F1分数
    precision = np.diag(conf_mat) / conf_mat.sum(axis=0)  # TP / (TP + FP)
    recall = np.diag(conf_mat) / conf_mat.sum(axis=1)     # TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    mean_f1 = np.nanmean(f1_score)

    # 计算Dice系数
    dice_coeff = 2 * np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0))
    mean_dice = np.nanmean(dice_coeff)

    # 求kappa
    pe = np.dot(np.sum(conf_mat, axis=0), np.sum(conf_mat, axis=1)) / (conf_mat.sum()**2)
    kappa = (acc - pe) / (1 - pe)
    
    return acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa, f1_score, mean_f1, dice_coeff, mean_dice

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

    # 只打印类别1（非背景类）的指标
    idx = 1  # 只关注类别1

    if idx < len(iu) and hist.sum(axis=1)[idx] > 0:  # 确保类别存在且有像素
        idx_string = "{:2d}".format(idx)
        iu_string = '{:5.2f}'.format(iu[idx] * 100)
        precision = '{:5.2f}'.format(100 * iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_positive[idx]))
        recall = '{:5.2f}'.format(100 * iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_negative[idx]))

        # 计算F1分数
        precision_val = iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_positive[idx])
        recall_val = iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_negative[idx])
        f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        f1_string = '{:5.2f}'.format(f1 * 100)

        # 计算Dice系数
        dice = (2 * iu_true_positive[idx]) / (iu_true_positive[idx] + iu_false_positive[idx] + iu_true_positive[idx] + iu_false_negative[idx])
        dice_string = '{:5.2f}'.format(dice * 100)

        logging.info('label_id          IoU    Precision Recall   F1-Score Dice')
        logging.info('{}       {}  {}     {}     {}    {}'.format(idx_string, iu_string, precision, recall, f1_string, dice_string))
    else:
        # 如果类别1不存在，打印所有类别
        logging.info('label_id          IoU    Precision Recall')
        for idx, i in enumerate(iu):
            if hist.sum(axis=1)[idx] > 0:  # 确保类别存在且有像素
                idx_string = "{:2d}".format(idx)
                iu_string = '{:5.2f}'.format(i * 100)
                precision = '{:5.2f}'.format(100 * iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_positive[idx]))
                recall = '{:5.2f}'.format(100 * iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_negative[idx]))

                # 计算F1分数
                precision_val = iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_positive[idx])
                recall_val = iu_true_positive[idx] / (iu_true_positive[idx] + iu_false_negative[idx])
                f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
                f1_string = '{:5.2f}'.format(f1 * 100)

                # 计算Dice系数
                dice = (2 * iu_true_positive[idx]) / (iu_true_positive[idx] + iu_false_positive[idx] + iu_true_positive[idx] + iu_false_negative[idx])
                dice_string = '{:5.2f}'.format(dice * 100)

                logging.info('{}       {}  {}     {}     {}    {}'.format(idx_string, iu_string, precision, recall, f1_string, dice_string))