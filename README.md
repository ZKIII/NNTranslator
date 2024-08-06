# NNTranslator



* Requirements
  * Necessary:
    * System: Windows 11
    * RAM: 32GB+
    * Python 3.9.2
    
  * Recommended:
    
    * hardware:
    
      * GPU Memory: 16GB+
    
    * driver
    
      > check the availability from [here](https://tensorflow.google.cn/install/source_windows?hl=en#gpu). The following options are for `tensorflow==2.9.0`
    
      * CUDA 11.2
      * Bazel 5.0.0
      * cuDNN 8.1
      * MSVC 2019
  
* Packages
  * All packages listed in `requirements.txt`
  * Using command `pip install -r requirements.txt` to install all required packages
  
* Scripts
  * `visualization.py`: visualize the dataset by confusion matrix and t-SNE plot
  * `preprocessing.py`: preprocessing the dataset in `data`(`cmn.txt` and `news.tsv`)
  * `model.py`: training models (LSTM and LSTM with Attention)
  * `inference.py`: predicting input sentences
  
* Execution

  * just run the `inference.py` it will automatically train the model. After that it will start the prediction process and you can type any sentences