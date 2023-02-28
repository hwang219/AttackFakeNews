# Attacking Fake News Detectors via Manipulating News Social Engagement

## Paper
Implementation for our WWW'23 [paper](https://arxiv.org/pdf/2302.07363.pdf) below:
```bibtex
@inproceedings{wang2023attacking,
  title={Attacking Fake News Detectors via Manipulating News Social Engagement},
  author={Wang, Haoran and Dou, Yingtong and Chen, Canyu and Sun, Lichao and Yu, Philip S and Shu, Kai},
  booktitle={Proceedings of the ACM Web Conference 2023},
  year={2023}
}
```

## Setup
To run the code, you need [FakeNewsNet](https://arxiv.org/abs/1809.01286) dataset. Due to Twitter privacy concerns, we cannot release user Tweet data.
Please send email with the title `MARL Dataset Request` to [hwang219@hawk.iit.edu](mailto:hwang219@hawk.iit.edu) to download the file with user raw Tweet data. You can unzip the dataset file under the root directory of the project.

## Run the code
```sh
cd MARL
```
To run the two random baselines:
```sh
python attack_base.py
```
To run MARL:
```sh
python attack.py
```