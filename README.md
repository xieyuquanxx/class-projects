# EntityLinkNLP

## quick start
```
conda create -n el-nlp python=3.9
conda activate el-nlp
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install lightning -c conda-forge

pip install .
```

## 模型
> 均从hugging face上下载，解压后放在根目录
- bert-base-chinese
- chinese-roberta-wwm-ext
- ernie-3.0-base-zh

## 训练数据
之后放在谷歌网盘上，下载后放在`data/`下。

## 模型
在`model/`目录下，现在放了3个实现，`v2_bert_model.py`的模型架构于baseline基本一样，`v3_model.py`实现了更深的网络结果。

## 训练
在根目录下有`v2_trian.py`和`v3_train.py`，分别启动`v2`版本和`v3`版本，同时需要设置对应的超参数。


## 测试和推理
在`model_eval.py`里实现了推理，`scripts/dev.sh`里给出了一个example。