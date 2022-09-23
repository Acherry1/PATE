# Semi-supervised learning ([MixMatch](https://arxiv.org/abs/2001.07685)) on Medical Imaging 

![MixMatch archtecture](https://github.com/vasudev-sharma/tutorial-MixMatch/blob/master/images/MixMatchFinal.png)
 > Architecture Image credit: https://amitness.com/2020/07/semi-supervised-learning/#a-mixmatch

## Problem Statement ❓: 
Chest X Ray Classification into 3 classes namely **COVID**, **Normal**, and **Pneumonia**<br> with and without limited labels (semi-supervised approach)

For more details, read the problem statement available [here](https://github.com/vasudev-sharma/tutorial-MixMatch/blob/master/Analysis.pdf)

## Data 💽
Dataset is in this Gdrive location:-> https://drive.google.com/file/d/1PVJcXmKz-QPAvPV-rRFYnjh7BsFnAeq8/view?usp=sharing

## Usage 🧑‍💻:
The Colab notebooks for the [Supervised approach](https://github.com/vasudev-sharma/tutorial-MixMatch/blob/master/supervised.ipynb) and [Semi-Supervised appraoch](https://github.com/vasudev-sharma/tutorial-MixMatch/blob/master/MixMatch-semi_supervised.ipynb) on toy COVID dataset are self-explantory. 

**NOTE** It is recommended to run the notebook on a GPU instance. If not available, kindly use the free Google Colab GPU instance

## RESULTS 💡: 
### Supervised approach [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/vasudev-sharma/tutorial-MixMatch/blob/master/supervised.ipynb) [![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg)](https://wandb.ai/vs74/Expand-ai-problem-1?workspace=user-vs74) <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" />

| Base Model      | Macro_f1_score | Accuracy    | Micro_F1_score | Model Logs                                                                  |
|-----------------|----------------|-------------|----------------|-----------------------------------------------------------------------------|
| densenet121     | 0.64502938     | 0.989583373 | 0.98958333     | https://wandb.ai/vs74/Expand-ai-problem-1/runs/1f2z5217?workspace=user-vs74 |
| efficientnet-b5 | 0.59190719     | 0.979166687 | 0.97916667     | https://wandb.ai/vs74/Expand-ai-problem-1/runs/3czw9l6w?workspace=user-vs74 |
| efficientnet-b3 | 0.65858588     | 0.989583373 | 0.98958333     | https://wandb.ai/vs74/Expand-ai-problem-1/runs/g3folx1r?workspace=user-vs74 |
| efficientnet-b0 | 0.60034412     | 0.989583373 | 0.98958333     | https://wandb.ai/vs74/Expand-ai-problem-1/runs/2urz30gh?workspace=user-vs74 |
| resnet101       | 0.57482045     | 0.963541687 | 0.96354167     | https://wandb.ai/vs74/Expand-ai-problem-1/runs/2w044b3y?workspace=user-vs74 |
| densenet169     | 0.59401157     | 0.984375    | 0.984375       | https://wandb.ai/vs74/Expand-ai-problem-1/runs/1f136ly7?workspace=user-vs74 |

---

### MixMatch (semi-supervised learning approach) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/vasudev-sharma/tutorial-MixMatch/blob/master/MixMatch-semi_supervised.ipynb) [![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg)](https://wandb.ai/vs74/Expand-ai-problem-2?workspace=user-vs74) <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" />


To tackle limited labels for Chest XRay images, a semi-supervised learning apprach known as [MixMatch](https://arxiv.org/abs/1905.02249) approach is adopted.<br>

The [MixMatch tutorial](https://github.com/vasudev-sharma/tutorial-MixMatch/blob/master/MixMatch-semi_supervised.ipynb) solution has been inspired from this https://github.com/YU1ut/MixMatch-pytorch repository. 

| Base Model | Accuracy | Macro_F1_Score | Micro_F1_score | Model Logs                                                                        |
|------------|----------|----------------|----------------|-----------------------------------------------------------------------------|
| ResNet-50  | 96.90%   | 0.3452         | 0.9609         | https://wandb.ai/vs74/Expand-ai-problem-2/runs/3bistr67?workspace=user-vs74 |

## [Analysis](https://github.com/vasudev-sharma/tutorial-MixMatch/blob/master/Analysis.pdf)
Well, you might be wondering why a specific model/algorithm was tried for this problem, right? To know about the thought process, kindly refer [here](https://github.com/vasudev-sharma/tutorial-MixMatch/blob/master/Analysis.pdf)


## TODO
- [ ] Play with the hyperparameters of the MixMatch algorithm
- [ ] Experiment with differnt baseline models. Eg.. DenseNets, EfficientNets, etc...
