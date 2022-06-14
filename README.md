# A Hybrid Method Based on Semi-Supervised Learning for Relation Extraction in Chinese EMRs

## experimental environment
- CPU：Intel(R) Core(TM) i7-8700K CPU@3.70GHz
- GPU：NVIDIA GeForce GTX 1080
- OS：Ubuntu18.04 LTS
- DP framework：Pytorch 1.2.0

## useage
1. First Train:
> $python main.py
2. BootStrapping:
> $python bootStrapping.py
3. Relation Extraction and KG builting: 
> $python KG_builting.py

## Document and file descrition
- **/data**  : datasets
  - **/original_data** : original datasets
  - **/extended_data** : Expanded training corpus during semi-supervised training
  - **/predict_results** : Prediction results 
  - **char2isdx.pkl** : Word vectors
  - **pos2idx.pkl** :  Position vectors
- **/KG**    : The entities and relatins of Chinese EMRs
- **/model** : Trained model
- **/module**  : Neural network model framework
- **/pretrained**  : Word embedding pre training model
- **bootstrappint.py**  : Semi supervised learning adjusts the training process
- **util.py, preprocs.py, data.py** : Data preprocessor
- **config.py** : Parameter settings
- **predict.py**  : Prediction
- **train.py** : training
- **evaluate.py** : Model evaluation
- **main.py** : Main program
