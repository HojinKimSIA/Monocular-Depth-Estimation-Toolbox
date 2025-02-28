from collections import OrderedDict

import mmcv
import numpy as np
import torch

def calculate(gt, pred):
    if gt.shape[0] == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)

    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    if np.isnan(silog):
        silog = 0
        
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel

def metrics(gt, pred, min_depth=1e-3, max_depth=80):
    mask_1 = gt > min_depth
    mask_2 = gt < max_depth
    mask = np.logical_and(mask_1, mask_2)

    gt = gt[mask]
    pred = pred[mask]

    a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = calculate(gt, pred)

    return a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel

def eval_metrics(gts, preds, val_masks, min_depth=1e-3, max_depth=80):
    a1s         = []
    a2s         = []
    a3s         = []
    abs_rels    = []
    rmses       = []
    rmse_logs   = []
    log_10s     = []
    silogs      = []
    sq_rels     = []

    for i in range(len(gts)):
        gt      = gts[i]
        pred    = preds[i]
        val_mask= val_masks[i]

        gt = gt[val_mask]
        
        if np.shape(pred) != np.shape(gt):
            pred = pred.squeeze()
            pred = pred[val_mask]
        else:
            pred = pred[val_mask]

        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a1s.append(a1)
        a2 = (thresh < 1.25 ** 2).mean()
        a2s.append(a2)
        a3 = (thresh < 1.25 ** 3).mean()
        a3s.append(a3)

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        abs_rels.append(abs_rel)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)
        sq_rels.append(sq_rel)

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())
        rmses.append(rmse)

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        rmse_logs.append(rmse_log)

        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
        silogs.append(silog)

        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
        log_10s.append(log_10)
    return dict(a1=a1s, 
                a2=a2s, 
                a3=a3s, 
                abs_rel=abs_rels, 
                rmse=rmses, 
                log_10=log_10s, 
                rmse_log=rmse_logs,
                silog=silogs, 
                sq_rel=sq_rels)


def pre_eval_to_metrics(pre_eval_results):

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    ret_metrics = OrderedDict({})

    ret_metrics['a1'] = np.nanmean(pre_eval_results[0])
    ret_metrics['a2'] = np.nanmean(pre_eval_results[1])
    ret_metrics['a3'] = np.nanmean(pre_eval_results[2])
    ret_metrics['abs_rel'] = np.nanmean(pre_eval_results[3])
    ret_metrics['rmse'] = np.nanmean(pre_eval_results[4])
    ret_metrics['log_10'] = np.nanmean(pre_eval_results[5])
    ret_metrics['rmse_log'] = np.nanmean(pre_eval_results[6])
    ret_metrics['silog'] = np.nanmean(pre_eval_results[7])
    ret_metrics['sq_rel'] = np.nanmean(pre_eval_results[8])

    ret_metrics = {
        metric: value
        for metric, value in ret_metrics.items()
    }

    return ret_metrics

