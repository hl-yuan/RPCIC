# Robust Prototype Completion for Incomplete Multi-view Clustering（ACM MM 2024）

This repo contains the code and data of our ACM MM'2024 paper Robust Prototype Completion for Incomplete Multi-view Clustering.

## Requirements

pytorch==1.2.0 

numpy>=1.19.1

scikit-learn>=0.23.2

munkres>=1.1.4

## Datasets

The Caltech101_7, HandWritten datasets are placed in "Datasets" folder. 

## Usage

The code includes:

- an example implementation of the model,
- an example clustering task for different missing rates.

```bash
python main.py --i_d 0
```


