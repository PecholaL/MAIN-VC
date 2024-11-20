# MAIN-VC
[![](https://img.shields.io/badge/LICENSE-Apache_2.0-yellow?style=flat)](https://github.com/PecholaL/MAIN-VC/blob/main/LICENSE) 
[![](https://img.shields.io/badge/IJCNN-2024-green?style=flat)](https://2024.ieeewcci.org/) 
[![](https://img.shields.io/badge/AI-speech-pink?style=flat)](https://github.com/PecholaL/MAIN-VC) 
[![](https://img.shields.io/badge/arXiv-2405.00930-red?style=flat)](https://arxiv.org/pdf/2405.00930)
[![](https://img.shields.io/badge/Pechola_L-blue?style=flat)](https://github.com/PecholaL)

Lightweight Speech Representation Learning for One-Shot Voice Conversion  
[**MAIN-VC** home page](https://pecholal.github.io/MAIN-VC-demo/)  
[**MAIN-VC** 中文导读](https://pecholal.github.io/2024/02/14/main_vc/)  

## Abstract
One-shot voice conversion aims to change the timbre of any source speech to match that of the unseen target speaker with only one speech sample. Existing methods face difficulty in satisfactory speech representation disentanglement and suffer from sizable networks. We propose a method to effectively disentangle with a concise neural network. Our model learns clean speech representations via siamese encoders with the enhancement of the designed mutual information estimator. The siamese structure and the newly designed convolution module contribute to the lightweight of our model while ensuring the performance in diverse voice conversion tasks.

## Usage
### I. Process training data.
`make_dataset.py`->`sample_dataset.py`  
Excute the bash `./data/preprocess/preprocess.sh` (.\data\preprocess\preprocess.sh for Windows) after modifying the configuration.

### II. Train MAIN-VC.
The CMI module of MAIN-VC is packaged in `mi.py`. Then all the components are assmebled in `model.py`.  
The configuration of the model is in `./config.yaml`.  
Excute the bash `./train.sh` (.\train.bat for Windows) after modifying the configuration. The configuration in the file is our recommended. You can also adjust the size of layers in the network for better performance or less training consuming.

### III. Inference via trained MAIN-VC.
Any suitably sized (i.e. the bank size of Mel-spectrogram) **pre-trained** vocoder model can be leveraged as a vocoder for MAIN-VC for logMel-spectrogram to waveform conversion.  
The pre-trained vocoder of [MAIN-VC demo](https://pecholal.github.io/MAIN-VC-demo/) is available at: [vocoder.pt](https://drive.google.com/file/d/1r0exien35ddN303dtYdCriHwDxVSFY_7/view?usp=sharing).  
Set the path to the check-point file of pre-trained vocoder in `inference.sh` with the argument '-v'.  
Set the path to source/target/converted(output) wave file in `inference.sh` then excute it. 

### IV. Attention
**Absolute path is preferred** for all the paths in our project.  
MAIN-VC is not very demanding on computing devices. It is sufficient to use a single Tesla V100 to train in our experiment.  

## Thanks
[AdaIN-VC](https://github.com/jjery2243542/adaptive_voice_conversion) [CLUB](https://github.com/Linear95/CLUB)    

## Citation
If **MAIN-VC** helps your research, please cite it as,  
Bibtex: 
```
@inproceedings{li2024mainvc,
  title={MAIN-VC: Lightweight Speech Representation Disentanglement for One-shot Voice Conversion},
  author={Li, Pengcheng and Wang, Jianzong and Zhang, Xulong and Zhang, Yong and Xiao, Jing and Cheng, Ning},
  booktitle={2024 International Joint Conference on Neural Networks},
  pages={1--7},
  year={2024},
  organization={IEEE}
}
```

or with a [hyperlink](https://github.com/PecholaL/MAIN-VC),  
Markdown: `[MAIN-VC](https://github.com/PecholaL/MAIN-VC)`  
Latex: `\href{https://github.com/PecholaL/MAIN-VC}{\textsc{MAIN-VC}}`
