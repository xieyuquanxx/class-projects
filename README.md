# EntityLinkNLP


## 模型
> 均从hugging face上下载
- bert-base-chinese（目前这个最好）
- chinese-roberta-wwm-ext
- ernie-3.0-base-zh

## 训练数据
之后放在谷歌网盘上。

## 模型
在`model/`目录下，现在放了3个实现，`v2_bert_model.py`的模型架构于baseline基本一样，`v3_model.py`实现了更深的网络结果。

## 训练
在根目录下有`v2_trian.py`和`v3_train.py`，分别启动`v2`版本和`v3`版本，同时需要设置对应的超参数。


## 测试和推理
在`model_eval.py`里实现了推理，`scripts/dev.sh`里给出了一个example。