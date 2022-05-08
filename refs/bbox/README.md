## Introduction
These data contain tracking box results on AQA-7 and MTL-AQA. 
File Tree:
```
.
├── README.txt
├── tracking_boxes_for_AQA-7
│   ├── diving.json
│   ├── gym_vault.json
│   ├── ski_big_air.json
│   ├── snowboard_big_air.json
│   ├── sync_diving_10m.json
│   └── sync_diving_3m.json
└── tracking_boxes_for_MTL-AQA
    ├── MTL-AQA_start_frame.csv
    └── MTL_AQA_boxes.json
```

## Data format
```
[point-0_x, point-0_y, point-1_x, point-1_y, point-2_x, point-2_y, point-3_x, point-3_y]
```
Note that the first box should be adjust before using. The code of adjusting is as fellow:
```
tmp_a, tmp_b = box[0][6], box[0][7] 
box[0][6], box[0][7] = box[0][4], box[0][5]
box[0][4], box[0][5] = tmp_a, tmp_b
```

The data format of AQA-7 is pretty simple. It should be noted that some frames in MTL_AQA dataset are irrelevant to AQA task, so we delete these frames. All start frames are recorded in _MTL-AQA_start_frame.csv_

If you find our work useful in your research, please consider citing:
```
@inproceedings{TSA-Net,
  title={TSA-Net: Tube Self-Attention Network for Action Quality Assessment},
  author={Wang, Shunli and Yang, Dingkang and Zhai, Peng and Chen, Chixiao and Zhang, Lihua},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  year={2021},
  pages={4902–4910},
  numpages={9}
}
```