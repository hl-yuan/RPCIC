# Robust Prototype Completion for Incomplete Multi-view Clustering（ACM MM 2024）

This repo contains the code and data of our ACM MM'2024 paper "Robust Prototype Completion for Incomplete Multi-view Clustering". If you have any questions about the source code, please contact: hl_yuan0822@163.com.

![framework](/figure/Overview.png)

## Requirements

pytorch==1.2.0 

numpy>=1.19.1

scikit-learn>=0.23.2

munkres>=1.1.4

## Datasets

You can find these datasets used in the paper at [Quark](https://pan.quark.cn/s/8d8c394501f7).

## Training

Run the code by
```bash
python main.py --i_d 0 --missrate 0.3
```
## Citation

If you find our work useful in your research, please consider citing:

```latex
@inproceedings{yuan2024robust,
  title={Robust Prototype Completion for Incomplete Multi-view Clustering},
  author={Yuan, Honglin and Lai, Shiyun and Li, Xingfeng and Dai, Jian and Sun, Yuan and Ren, Zhenwen},
  booktitle={ACM Multimedia 2024}
}
```
