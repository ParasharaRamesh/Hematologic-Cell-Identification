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

1. On WBC use the segmentation masks to create new augmented data and train a UNet for classification
2. On pRCC train an autoencoder (also using unet architecture) but get the features from the encoder 
3. On Camelyon16 use another UNet for classification
4. For end to end training use pRCC encoder (non trainable) + Camelyon16 (non trainable, everything except classifier head) + Unet from WBC (weights)(trainable)

## <u>TODO</u>

[ ] Push dataset into google drive
<br>
[ ] Understand white blood cells and the biology behind it
<br>
[x] Understand the datasets ( need to balance the WBC and camelyon datasets)
<br>
[ ] Are the pre-trainable datasets combinable to form one dataset?
<br>
[ ] From each dataset figure out a rough outline for the inputs and outputs for models built on each & see how to make use of common weights (multi task learning for the pretraining datasets?)
<br>
[ ] See if there are any models which can be used as my base model to train the pretrainable network ( say vgg or resnet or something )
<br>
[ ] See what kind of models exist for this kind of problem/ For each dataset (UNet / multi task/ object detection?)
<br>
[ ] Come up with metrics we need to use
<br>
[ ] Come up with ways to show visualization (eg attention heat maps, test results etc)
<br>
[ ] Push dataset into google drive and get code to use it in colab
<br>
[ ] Finish coding the data pipelines + viz in a generic reusable way
<br>
[ ] Train the pretrained model to the best of my ability and save the weights ( But figure out how to efficiently load them )
<br>
[ ] Whats the conclusion about scientific discoveries?
<br>

## <u> Evaluation of pretrained CNN models we can use </u>

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9777002/#:~:text=The%20WBC%20count%20represents%20essential,as%20depicted%20in%20Figure%201.