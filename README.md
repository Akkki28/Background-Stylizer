# Background-Stylizer
Built a model that stylizes the background of an image cropping out the person in it using Neural Style transfer and semantic segmentaion techniques
given two images,one content,one style, the model outputs the following
![videoGit](https://github.com/Akkki28/Background-Stylizer/assets/120105455/eb68cd25-711a-477e-b0f8-e09033285ce8)

# Neural Style Transfer
![](https://i.ytimg.com/vi/c3kL9yFGUOY/maxresdefault.jpg)
Neural style transfer (NST) refers to a class of software algorithms that manipulate digital images, or videos, in order to adopt the appearance or visual style of another image This project utilizes a TensorFlow Hub model for Neural Style Transfer, enabling efficient image manipulation through pre-trained weights and optimized code.

# Semantic Segmentation
![](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.labellerr.com%2Fblog%2Fsemantic-vs-instance-vs-panoptic-which-image-segmentation-technique-to-choose%2F&psig=AOvVaw10FkrMSfDbUnjNdjZcELG3&ust=1716300443547000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCID99MSznIYDFQAAAAAdAAAAABAE)
In digital image processing and computer vision, image segmentation is the process of partitioning a digital image into multiple image segments, also known as image regions or image objects.YOLO for object segmentation is employed to identify and isolate image regions for style transfer.
