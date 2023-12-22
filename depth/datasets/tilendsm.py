from logging import raiseExceptions

import os
import csv
import os.path as osp
import warnings
from collections import OrderedDict
from functools import reduce

from PIL import Image
import torch
import json
import glob

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from depth.core import pre_eval_to_metrics, metrics, eval_metrics
from depth.utils import get_root_logger

from depth.datasets.builder import DATASETS
from depth.datasets.pipelines import Compose

from depth.ops import resize
##############################

import rasterio
from rasterio.plot import show
##############################

@DATASETS.register_module()
class nDSMTileDataset(Dataset):
    """ DSM Tile Dataset for DSM estimation. An example of file structure
    is as followed.
    CORE3D
    ├── Tiled-Images
    │   ├── Training
    │   │   ├── AoIs (ex, JAX)
    │   │   │   ├── RGB
    │   │   │   │   ├── single (Mono-view Setting)
    │   │   │   │   │   ├── JAX_000_RGB.tif
    │   │   │   │   │   ├── JAX_004_006_RGB.tif
    │   │   │   │   │
    │   │   │   │   ├── pair (Stereo-view Setting)
    │   │   │   │   │   ├── JAX_161_008_016_LEFT_RGB.tif
    │   │   │   │   │   ├── JAX_161_008_016_RIGHT_RGB.tif
    │   │   │   │   │   ├── JAX_161_008_016_METADATA.json
    │   │   │   │   │
    │   │   │   │   ├── multi (Multi-view Setting)
    │   │   │   │       ├── JAX_117_001_RGB.tif
    │   │   │   │       ├── JAX_117_002_RGB.tif
    │   │   │   │       ├── JAX_117_003_RGB.tif
    │   │   │   │       ├── JAX_117_004_RGB.tif
    │   │   │   ├── DSM
    │   │   │   │   ├── single
    │   │   │   │   │   ├── JAX_000_DSM.tif
    │   │   │   │   │   ├── JAX_004_006_AGL.tif
    │   │   │   │
    │   │   │   │
    │   │   │   ├── DTM
    │   │   │   │   ├── single
    │   │   │   │   │   ├── JAX_000_DTM.tif
    │   │   │   │   │  
    │   │   │   │
    │   │   │   │
    │   │   │   ├── SemanticMask
    │   │   │   │   ├── single
    │   │   │   │   │   ├── JAX_000_GTC.tif
    │   │   │   │   │   ├── JAX_000_GTI.tif
    │   │   │   │   │   ├── JAX_000_GTL.tif
    │   │   │   │   │   ├── JAX_004_006_CLS.tif

    """
    
    def __init__(self,
                 pipeline,
                 data_root='Toy_tile',
                 img_dir='RGB',
                 dsm_dir='DSM',
                 dtm_dir='DTM',
                 phase='Training',
                 AoIs=['JAX'],
                 test_mode=False,
                 min_depth = 0,
                 max_depth = 200,
                 depth_scale=1):

        self.data_root = data_root
        self.pipeline = Compose(pipeline)
        self.path_template = os.path.join(data_root, '{}', '{}', 'single_tile', 'imgs')
        self.img_path = img_dir
        self.dsm_path = dsm_dir
        self.dtm_path = dtm_dir
        self.aois = AoIs
        self.test_mode = test_mode
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.phase = phase
        self.img_infos = self.load_annotations(self.img_path)
    
    def __len__(self):
        # Total number of samples of data.
        return len(self.img_infos)

    def load_annotations(self, img_dir):
        # Load image file abs name / dsm file abs name / dtm file abs name
        img_infos = []
        
        '''
        imgs = []
        for aoi in self.aois:
            imgs.extend(glob.glob(os.path.join(self.img_path.format(aoi), '*RGB.tif'), recursive=True))
        imgs.sort()
        print('number of total images : ', len(imgs))

        dsms = []
        dtms = []
        ndsms = []
        sems = []
         
        for img in imgs:
            print('img : ', img)
            if len(img.split('/')[-1].split('_')) == 3:
                dsm = img.replace('/RGB', '/DSM').replace('RGB.t','DSM.t')
                dtm = img.replace('/RGB', '/DTM').replace('RGB.t','DTM.t')
                sem = img.replace('/RGB', '/SemanticMask').replace('RGB.t','GTL.t')
            else:
                dsm = img.replace('/RGB', '/DSM').replace('RGB.t', 'AGL.t')
                dtm = None
                sem = img.replace('/RGB', '/SemanticMask').replace('RGB.t','CLS.t')

            img_info = dict()
            img_info['filename'] = img
            img_info['ann'] = dict(dsm=dsm, dtm=dtm, semantic=sem)
            img_infos.append(img_info)

        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images.', logger=get_root_logger())
        '''
        tr_sample_namelist = 'trlist91.csv'
        te_sample_namelist = 'telist91.csv'
        for aoi in self.aois:
            if self.phase == 'train':
                flist_dir = os.path.join(self.data_root, aoi, img_dir, 'single_tile', tr_sample_namelist)
            else:
                flist_dir = os.path.join(self.data_root, aoi, img_dir, 'single_tile', te_sample_namelist)

            self.img_files = list(csv.reader(open(flist_dir), delimiter=','))[0]
            for img in self.img_files:
                img_info = dict()

                img_info['filename'] = os.path.join(self.path_template.format(aoi, self.img_path), img)
                dsm_name = os.path.join(self.path_template.format(aoi, self.dsm_path), img)
                dtm_name = os.path.join(self.path_template.format(aoi, self.dtm_path), img)
                img_info['ann'] = dict(dsm=dsm_name, dtm=dtm_name)
                img_info['name'] = img

                img_infos.append(img_info)
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images.', logger=get_root_logger())


        return img_infos
    
    def get_ann_info(self, idx):
        """ Get annotation by index.
        Args:
            idx (Int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """
        return self.img_infos[idx]['ann']

    def __getitem__(self, idx):
        """ Get training/testing data after pipeline
        Args: 
            idx (int): Index of data.
        Returns:
            dict: Training/testing data with annotation
        """

        return self.prepare_img(idx)

    def prepare_img(self, idx):
        """ Get training/testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        results['depth_fields'] = []
        results['depth_scale'] = 200
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """ Placeholder to format result to dataset specific output. """
        raise NotImplementedError

    def get_gt_dsm(self):
        """ Get gt dsm for evaluation. """
        dsms = []
        nodata_masks = []
        for img_info in self.img_infos:
            '''
            if img_info['ann']['dtm'] is None:
                _dsm            = rasterio.open(img_info['ann']['dsm'])

                nodata_value    = _dsm.nodata
                nodata_mask     = _dsm.read_masks(1)
                
                dsm             = _dsm.read()
                #dsm             = np.transpose(dsm, (1,2,0))
                dsm             = np.squeeze(dsm)
                dsm[dsm == nodata_value] = self.min_depth
            else:
                _dsm = rasterio.open(img_info['ann']['dsm'])
                _dtm = rasterio.open(img_info['ann']['dtm'])
                
                nodata_mask     = _dsm.read_masks(1)
                dsm = _dsm.read() - _dtm.read()
                #dsm = np.transpose(dsm, (1,2,0))
                dsm = np.squeeze(dsm)
            '''
            dsm = rasterio.open(img_info['ann']['dsm'])
            dtm = rasterio.open(img_info['ann']['dtm'])

            nodata_mask = dsm.read_masks(1)
            nodata_mask = nodata_mask.astype(np.bool_)

            dsm = dsm.read()
            dtm = dtm.read()

            dsm[dsm<0] = 0
            dtm[dtm<0] = 0

            ndsm = dsm-dtm
            # dsm shape -> HxW
            # nodata mask shape -> HxW
            dsms.append(ndsm)
            nodata_masks.append(nodata_mask)

        # dsms shape -> NxHxW
        # nodata masks -> NxHxW
        return dsms, nodata_masks

    def pre_eval(self, preds, indices):
        """ Collect eval result from each iteration.
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the dsm estimation, shape(N, H, W).
            indices (list[int] | int): the prediction related gt indices.
        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction, area_ground_truth).
        """
        def eval_mask(nodata_mask):
            nodata_mask = nodata_mask.astype(np.bool_)
            nodata_mask = np.expand_dims(nodata_mask, axis=0)
            # Nodata mask shape -> 1xHxW
            return nodata_mask
            
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results    = []
        pre_eval_preds      = []

        for i, (pred, index) in enumerate(zip(preds, indices)):
            img_info = self.img_infos[index]
            '''
            if img_info['ann']['dtm'] is None:
                _dsm = rasterio.open(img_info['ann']['dsm'])
                
                nodata_value = _dsm.nodata
                nodata_mask  = _dsm.read_masks(1)
            
                dsm_gt = _dsm.read()
                # dsm shape -> 1xHxW
                #dsm_gt = np.transpose(dsm_gt, (1,2,0))
                dsm_gt[dsm_gt == nodata_value] = self.min_depth
                #dsm_gt = np.expand_dims(dsm_gt, axis=0)
            else:
                _dsm = self.img_infos[index]['ann']['dsm']
                _dtm = self.img_infos[index]['ann']['dtm']

                nodata_value = _dsm.nodata
                nodata_mask  = _dsm.read_masks(1)
        
                dsm_gt = _dsm.read() - _dtm.read()
                # dsm shape -> 1xHxW
                #dsm_gt = np.transpose(dsm_gt, (1,2,0))
                #dsm_gt = np.expand_dims(dsm_gt, axis=0)
            '''
            dsm = rasterio.open(img_info['ann']['dsm'])
            dtm = rasterio.open(img_info['ann']['dtm'])

            nodata_mask = dsm.read_masks(1)

            dsm = dsm.read()
            dtm = dtm.read()

            dsm[dsm<0] = 0
            dtm[dtm<0] = 0

            ndsm = dsm-dtm

            valid_mask = eval_mask(nodata_mask)
            # valid mask shape -> 1xHxW
            eval = metrics(ndsm[valid_mask], pred[valid_mask], self.min_depth, self.max_depth)
            pre_eval_results.append(eval)
            
            # save prediction results
            pre_eval_preds.append(eval)

        return pre_eval_results, pre_eval_preds

    def evaluate(self, results, metric='rmse', logger=None, **kwargs):
        """ Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                results or predict depth map for computing evaluation metric.
            logger (logging.Logger | None | str): Logger used for printing related information during evaluation.
                Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """
        metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
        eval_results = {}

        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            gt_dsms, val_masks = self.get_gt_dsm()
            # gt_dsms shape -> NxHxW
            # val masks shape -> NxHxW

            ret_metrics = eval_metrics(gt_dsms, results, val_masks, min_depth=self.min_depth, max_depth=self.max_depth)
        else:
            ret_metrics = pre_eval_to_metrics(results)

        ret_metric_names    = []
        ret_metric_values   = []

        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = len(ret_metric) // 9
        
        for i in range(num_table):
            names   = ret_metric_names[i*9: i*9 + 9]
            values  = ret_metric_values[i*9: i*9 + 9]

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 4)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            # for logger
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics.items():
            eval_results[key] = value

        return eval_results
