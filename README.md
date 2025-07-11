[paper]: #
[webdemo]: http://comprehensive-stroke-ct.scienceontheweb.net/
[dataset]: #

# 3D CT Stroke Diagnosis using GNN: Classification, Segmentation, and Severity Prediction
Authors: Zhicheng Lu, Mohammad Ali Moni*<br/>
*Corresponding author: mmoni@csu.edu.au<br/>
[[Paper][paper]]
[[Website Demo][webdemo]]
[[Dataset](#dataset)]
[<a href="#">BibTex</a>]<br/>

## Table of Contents
<ul>
  <li><a href="#paper-abstract">Paper Abstract</a></li>
  <li><a href="#dataset">Dataset</a></li>
  <li><a href="#dependency">Dependency</a></li>
  <li><a href="#test-the-model">Test the Model</a></li>
  <li><a href="#train-the-model">Train the Model</a></li>
  <li><a href="#citation">Citation</a></li>
</ul>

## Paper Abstract
Stroke is one of the leading causes of disability and fatality worldwide, particularly among the aging population. Early and accurate stroke diagnosis plays a vital role in facilitating immediate treatment and improving stroke outcomes. The extreme shortage of doctors and clinicians poses a significant challenge for stroke diagnosis globally, and this situation is even worse in rural areas. In response to this critical issue, we propose Graph Neural Networks for Stroke (StrokeGNN) based on emerging Graph Neural Networks (GNNs) for 3D feature extraction of computed tomography (CT) scan, and build an Intelligent Integrated Stroke Diagnosis System (IISDS) based on U-Net and StrokeGNN to alleviate the strain on medical resource. Notably, IISDS not only segments the stroke location, but also classifies patients into "no stroke", "ischemic stroke", and "hemorrhagic stroke" groups, as well as predicts stroke severity level, thus offering a comprehensive diagnosis solution for stroke. To justify general applicability and effectiveness of the proposed IISDS, we collect \textcolor{red}{xxx} dataset with \textcolor{red}{xxx} CT scans from Bangladesh and use multiple publicly available datasets. The experimental results illustrate that proposed StrokeGNN achieves top-tier performance compared with other state-of-the-art algorithms while being generalized. Our experimental results prove the feasibility of our IISDS for future clinical uses.<br>

[Full Paper][paper]

## Dataset
Dataset we collected from Bangladesh:<br/>
[xxx][dataset]<br>
Public dataset:<br/>
[[AISD](https://github.com/GriffinLiang/AISD)]
[[PhysioNet-ICH](https://physionet.org/content/ct-ich/1.3.1/)]
[[BHSD](https://www.kaggle.com/datasets/stevezeyuzhang/bhsd-dataset)]
[[Seg-CQ500](https://zenodo.org/records/8063221)]
[[RSNA](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data)]
[[CQ500](http://headctstudy.qure.ai/dataset)]

## Dependency
This project runs under Anaconda Python 3.9 with Cuda 11.8 and Pytorch 2.0.0.
<details>
  <summary>Click here or find <code>requirements.txt</code> for dependencies of this project.</summary>
  <pre><code>kmeans_pytorch==0.3
matplotlib==3.7.2
nibabel==5.2.1
numpy==1.24.3
opencv_python==4.8.1.78
pandas==2.0.3
Pillow==9.4.0
Pillow==10.2.0
pydicom==2.4.4
scipy==1.12.0
torch==2.0.0</code></pre>
</details>

## Test the Model
Put CT scan input in the `test_input/` folder. Please include numeric indices for ordering like `CT_01.jpg, CT_02.jpg, ...`. Then run the following command:
```
$ python test.py
```
Classification and severity prediction results are printed out in command line, and segmentation results can be found in `test_output/` folder.

## Train the Model
### Segmentation
Put training CT data in `data/segmentation/{fold}/train/{dataset}/images/{patient_ID}/` and mask data in `data/segmentation/{fold}/train/{dataset}/masks/{patient_ID}/` for each patient (image format is like previous part). Testing data are put under `data/segmentation/{fold}/test` folders in similar format. Specify training set and testing set in `config.ini`. Training epochs and batch size can also be initialized in `config.ini`. Run the following command to train the segmentation model:
```
$ python main.py -t segmentation -s train
```
Trained model for all epochs are in `checkpoints/segmentation_model_{time}/`. Trained model from any epoch may be copied and rename to `checkpoints/segmentation_model.pt`. Then run the following command for testing this model:
```
$ python main.py -t segmentation -s test
```

### Classification
Put training CT data in `data/classification/{fold}/train/{dataset}/{patient_ID}` (image format is like previous part). Testing data are put under `data/classification/{fold}/test/{dataset}/{patient_ID}` folders. Specify training set and testing set in `config.ini`. Training epochs and batch size can also be initialized in `config.ini`. Labels need also to be prepared in the following format in `labels/{dataset}.txt` for corresponding dataset:
```
000 [0,0,1]
001 [1,0,0]
```
where 000,001 are patient_ID and [normal, ischemic, hemorrhagic] represents the probability of each class.
Run the following command to train the classification model:
```
$ python main.py -t classification -s train
```
Trained model for all epochs are in `checkpoints/classification_model_{time}/`. Trained model from any epoch may be copied and rename to `checkpoints/classification_model.pt`. Then run the following command for testing this model:
```
$ python main.py -t classification -s test
```

### Severity Prediction
File paths for ischemic and hemorhagic stroke can be modified in `config.ini`. CT scan and masks are put under `{severity_stroketype}/{dataset}/images/{patient_ID}` and `{severity_stroketype}/{dataset}/masks/{patient_ID}`. Then we may run the following command to train the severity prediction model:
```
$ python main.py -t severity
```
Trained model will be saved as `checkpoints/severity_{stroketype}.pt`.

## Citation
For academic use, please cite:
```
@article{,
  title={},
  author={},
  year={}
}
```