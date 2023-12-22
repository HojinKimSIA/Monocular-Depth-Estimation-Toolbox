TIME=$(date +'%Y_%m_%d_%H_%M')
MMCV_DIR='/nas/k8s/dev/research/yongjin117/PROJECT_nautilus/mmcv'
WORK_DIR='/nas/k8s/dev/research/yongjin117/PROJECT_nautilus/MMMDE'

pip3 uninstall mmcv-full
cd ${MMCV_DIR}
MMCV_WITH_OPS=1 pip3 install -e .

cd ${WORK_DIR}
pip3 install rasterio albumentations timm
pip3 install -e .
python3 tools/train.py configs/depthformer/depthformer_swinl_22k_w7_dsm.py --work-dir w_dirs/depthformer/${TIME} --gpus 1
