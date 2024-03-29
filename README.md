# Update extracted & processed N-CMAPSS dataset, 6/10/2023
To promote relevant researches on domain adaptive RUL prediction in the community, we provide the N-CMAPSS dataset used in this paper, 
with 7 domains readily available. Download <a href="https://drive.google.com/file/d/1p7SuC0vZTls-ZmuNKRVxMgk4PjUp0tfK/view?usp=sharing">here<a>.

# Domain Adaptive Remaining Useful Life Prediction with Transformer
Pytorch implementation for **Domain Adaptive Remaining Useful Life Prediction with Transformer**, https://doi.org/10.1109/TIM.2022.3200667

- Prognostic health management (PHM) has become a crucial part in building highly automated systems, whose primary task
is to precisely predict the remaining useful life (RUL) of the system. In this paper, we leverage **domain adaptation** for RUL prediction and propose a novel method by
aligning distributions at both the feature level and the semantic level. The proposed method facilitates a large improvement of model
performance as well as faster convergence. Besides, we propose to use Transformer as backbone, which can capture long-term
dependency more efficiently. We test our model on **CMAPSS** dataset and its newly published variant **N-CMAPSS** provided by NASA, achieving
state-of-the-art results on both source-only RUL prediction and domain adaptive RUL prediction tasks.
![1677138917679](https://user-images.githubusercontent.com/68037940/220850214-9661b173-4b7e-4ecc-a3ae-f34a9ceb4bca.png)

# Environment
**This is a strict constraint!**

- pytorch==1.10
- torchvision==0.11.0
# Dataset
- Unzip `CMAPSS.zip` into `CMAPSS` folder, which contains processed CMAPSS dataset used in this code.
# Usage
- train on CMAPSS  
`python train_cmapss.py --source $S --target $T`   
where `$S` is source domain, `$T` is target domain. Domains include "FD001,FD002,FD003,FD004". Trained models are saved to `/online`.
- evaluate on CMAPSS  
You can evaluate our best performing models saved in folder `save/final` by running:  
`python validation_cmapss.py --source $S --target $T`  
where `$S` is source domain, `$T` is target domain. Domains include "FD001,FD002,FD003,FD004".
# Results
![image](https://user-images.githubusercontent.com/68037940/220871054-2bd6ac31-16ed-4958-bbaa-b87495b6d10c.png)
# Citation
```
@article{li2022domain,
  title={Domain Adaptive Remaining Useful Life Prediction With Transformer},
  author={Li, Xinyao and Li, Jingjing and Zuo, Lin and Zhu, Lei and Shen, Heng Tao},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={71},
  pages={1--13},
  year={2022},
  publisher={IEEE}
}
```
