_base_ = ['./detr_r50_ruod_120e_pretrained.py']

# Keep this aligned with the protocol you want to compare against.
# 100 matches your current DETR config.
model = dict(test_cfg=dict(max_per_img=100))

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    metric_items=['mAP', 'mAP_50', 'mAP_75'],
    outfile_prefix=work_dir + '/test_ruod/results'
)