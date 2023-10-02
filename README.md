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

## <u>TODO</u>

[ ] Understand white blood cells and the biology behind it
<br>
[ ] Understand the datasets. Are the pretrainable datasets combinable to form one dataset?
<br>
[ ] From each dataset figure out a rough outline for the inputs and outputs for models built on each & see how to make use of common weights (multi task learning for the pretraining datasets?)
<br>
[ ] See if there are any models which can be used as my base model to train the pretrainable network ( say vgg or resnet or something )
<br>
[ ] See what kind of models exist for this kind of problem/ For each dataset (UNet / multi task/ object detection?)
<br>
[ ] Come up with metrics we need to use
<br>
[ ] Come up with ways to show visualization ( eg attention heat maps, test results etc)
<br>
[ ] Finish coding the data pipelines in a generic reusable way
<br>
[ ] Train the pretrained model to the best of my ability and save the weights ( But figure out how to efficiently load them )
<br>
[ ] Whats the conclusion about scientific discoveries?
<br>