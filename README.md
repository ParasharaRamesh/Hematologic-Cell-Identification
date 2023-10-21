# <u> Hematologic-Cell-Identification </u>
This repo contains the project code for the CS5242 (Neural Networks &amp; Deep Learning) course taken @ NUS

## <u> Objective </u>

The primary objective of this project is to develop a classification system capable
of distinguishing between five classes of white blood cells: basophil, eosinophil,
lymphocyte, monocyte, and neutrophil.

We have 3 datasets:
1. pRCC and Camelyon16 dataset (for pre-training)
   - Also contains annotation masks in some cases
2. WBC dataset (for actual classification)
   - Use 100%,50%,10% & 1% for training (with the pretraining and without the pretraining)

Here are some traits of each dataset:
1. The pRCC dataset has no label so some kind of unsupervised learning needs to happen there
2. The Camelyon16 dataset and WBC dataset both have masks as well for segmentation
3. The Camelyon16 dataset has normal and tumour but not related to WBC from the looks of it.

## <u> Approach </u>

1. On WBC use the segmentation masks to create new augmented data and train a model for classification
2. On pRCC train an autoencoder but get the features from the encoder 
3. On Camelyon16 train yet another model for classification
4. For end to end training use pRCC encoder (non trainable) + Camelyon16 (non trainable) + Classifier from WBC (weights)(trainable)

For more details refer to the report.

## <u>Code organization</u>

1. ./config: Contains a file where the global constants are defined
2. ./data/balancing: Contains code needed for balancing the wbc dataset by adding augmented datapoints into the dataset
3. ./data/datasets: Contains the main dataset class needed for running each of the models
4. ./data/debug: Contains util code for taking a subset of the dataset size for testing the model training locally
5. ./data/maskification: Contains code needed for applying the mask onto the original dataset images to create new dataset points
6. ./data/move: Contains a util wrapper class which can move tensors to the cuda/cpu device
7. ./details: Contains the description of the problem statement
8. ./loss: Contains a custom loss class used when training the pRCC autoencoder
9. ./models: Contains the model architectures for each experiment conducted
10. ./utils: Contains code needed for plotting the training and testing graphs
11. ./experiments/base: Contains the base class for the generic trainer
12. ./experiments/classify: Contains the trainer class for classification tasks which subclasses the generic trainer
12. ./experiments/cam_classifier: Contains the trainer class for Camelyon 16 classification
13. ./experiments/pRCC_autoencoder: Contains the trainer class for pRCC Autoencoder
14. ./experiments/wbc_classifier: Contains the trainer class for WBC classification
15. ./experiments/wbc_pretrained: Contains the trainer class for WBC classification with pretraining from the pRCC and the Cam16 model

<b>NOTE:</b> All of the jupyter notebooks were run on colab therefore all the classes defined in other directories needed to be explicitly copy-pasted due to issues importing the relevant python files in the google colab environment. If the reader chooses to run this locally feel free to remove the class definitions and just import the relevant classes instead.

## <u>Datasets used</u>

### <u> Raw Datasets </u>
Use this <a href ="https://www.dropbox.com/sh/954r9ib45wz27x7/AAAchJJxjNCOjKFcPoogzIkXa?dl=0">dropbox link</a> for downloading the datasets


## <u> Processed Datasets & Model weights </u>

This <a href="https://drive.google.com/drive/folders/1lJLDTF6k3GGs2Oj7f07gVDdxNYfitAEy?usp=sharing">google drive link</a> contains both the weights of all models trained along with the preprocessed datasets uploaded as zip files (which can be directly usable in colab)