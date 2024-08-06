# NNTranslator



* Requirements
  * Necessary:
    * System: Windows 11
    * RAM: 32GB+
    * Python 3.9.2
  * Recommended:
    * GPU Memory: 16GB+
    * CUDA supported
* Packages
  * All packages listed in `requirements.txt`
  * Using command `pip install -r requirements.txt` to install all required packages
* Scripts
  * `visualization.py`: visualize the dataset by confusion matrix and t-SNE plot
  * `preprocessing.py`: preprocessing the dataset in `data`(`cmn.txt` and `news.tsv`)
  * `model.py`: training models (LSTM and LSTM with Attention)