**Vision Meets Wireless Positioning: Effective Person Re-identification with Recurrent Context Propagation**

[Yiheng Liu](https://yolomax.com/), [Wengang Zhou](http://staff.ustc.edu.cn/~zhwg/), Mao Xi, Sanjing Shen, [Houqiang Li](http://staff.ustc.edu.cn/~lihq/research.html)

MM, Oral Paper，2020.

[[PDF](https://dl.acm.org/doi/10.1145/3394171.3413984)]

## WP-ReID Dataset
![WP-ReID Dataset](./../cover/dataset.png)

```
├── /path/to/WP-ReID
│   ├── cropped_data            # The cropped video frames.  
│   ├── wp_reid_info.json   
```

|  | Probe | Gallery |
| :-----:| :----: | :----: |
| #ID | 41 | 79 |
| #Tracklet | 201 | 868 |
| #Images | 29353 | 106578 |
| #Camera | 6 | 6 |
| #MaxLen | 1165 | 1165 |
| #MinLen | 3 | 3 |
| #AvgLen | 146 | 122 |

wp_reid_info.json contains the mapping between iamges and GPS information. The GPS location of the pedestrian's mobile phone is also included. Please refer to the code in [wp_reid_dataset.py](https://github.com/yolomax/WP-ReID/blob/master/wp_reid_dataset/wp_reid_dataset.py) to utilize this file.

Function get_vision_record_dist in [update_module.py](https://github.com/yolomax/WP-ReID/blob/52640a98a4aa680e63a4c42329d2f7aa494a3bd0/update_module.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L28) returns the initial distance between the video trajectories of each video and the wireless trajectories of pedestrian's mobile phones.

## Contact
If you are interested in WP-ReID dataset, please contact us by email lyh156@mail.ustc.edu.cn.
