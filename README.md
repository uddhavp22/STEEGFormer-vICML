# STEEGFormer

A fair EEG BCI benchmark framework and a simple STEEGFormer foundation model

Codes written by Liuyin Yang (liuyin.yang@kuleuven.be), Qiang Sun (qiang.sun@kuluven.be)



All rights reserved.





\# 1.0 Environment:



The model is trained with PyTorch and thus can be used in Python environments. The following packages meet the minimal requirements to load the model.



\*The Python version used in the pre-training phase is Python 3.11.5.



|   Package   |   Version   |   Note                                    |

|-------------|:-----------:|------------------------------------------:|

|  timm       |  1.0.10     |Basic implementations of transformer models|

|  torch      |  2.4.1      |Deep learning framework                    |



If you want to run the training codes for foundation models and classic neural networks, you will also need the following packages:



|   Package   |   Version   |   Note                                    |

|-------------|:-----------:|------------------------------------------:|

|  wandb      |  0.22.2     |For training monitor and data logging      |

|  mat73      |  0.65       |Data loading                               |

|scikit-learn |  1.3.2      |Classification metrics                     |



If you want to run the training codes for classic models, you will also need the following packages:



|   Package   |   Version   |   Note                                    |

|-------------|:-----------:|------------------------------------------:|

|  mne        |  0.22.2     |For training monitor and data logging      |

|  pyriemann  |  0.65       |Data loading                               |

|  lightgbm   |  1.3.2      |Classification metrics                     |

|  meegkit    |  1.3.2      |Classification metrics                     |

|  scipy      |  1.3.2      |Classification metrics                     |





\# 2.0 Model specs:

ST-EEGFormer works with 128Hz EEG Data. During pre-training, it was pre-trained to reconstruct 6-second EEG segments with up to 145 different channels. Therefore, we recommend using it with maximally 6-second EEG segments under a 128 Hz sampling rate. The available channels can be found in \\pretrain\\senloc\_file.



