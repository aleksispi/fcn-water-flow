# Fully Convolutional Networks for Dense Water Flow Intensity Prediction in Swedish Catchment Areas

![krycklan-overview](https://user-images.githubusercontent.com/32370520/229508169-f3d070d5-5005-47be-b3d4-4e9188eea97d.png)

Official PyTorch implementation of the paper _Fully Convolutional Networks for Dense Water Flow Intensity Prediction in Swedish Catchment Areas_ by [Aleksis Pirinen](https://www.ri.se/en/person/aleksis-pirinen), [Olof Mogren](http://mogren.one/) and Mårten Västerdal.

### Dataset
The dataset used in the paper can be downloaded from [this link](https://www.dropbox.com/s/6i2shosy7ddnz09/dataset.zip?dl=0).

### Code structure overview
Model training is done using `training.py`. Results are sent to a log folder (see the variable `BASE_PATH_LOG`), and result plots can then be generated using the file `plot_result.py`.

### Training
_Prior to this, ensure you have the dataset and that the variable `BASE_PATH_DATA` points to this dataset folder._

Model training (and validation on validation data) is performed using `training.py`. See the file `plot_results.py` if you are interested in tracking the progress and results throughout training. Models are saved during and upon completion of training, and are sent to the log folder.

### Citation
TODO
