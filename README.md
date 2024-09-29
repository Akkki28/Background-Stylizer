# Background-Stylizer
Built a model that stylizes the background of an image cropping out the person in it using Neural Style transfer and semantic segmentaion techniques
given two images,one content,one style, the model outputs the following
[Deployed Link](https://background-stylizer.streamlit.app/)

# Neural Style Transfer
![](https://i.ytimg.com/vi/c3kL9yFGUOY/maxresdefault.jpg)

Neural style transfer (NST) refers to a class of software algorithms that manipulate digital images, or videos, in order to adopt the appearance or visual style of another image This project utilizes a TensorFlow Hub model for Neural Style Transfer, enabling efficient image manipulation through pre-trained weights and optimized code.

# Semantic Segmentation
![](https://cdn.labellerr.com/semantic%20segmentation/Semantic%20segmentation.webp)

In digital image processing and computer vision, image segmentation is the process of partitioning a digital image into multiple image segments, also known as image regions or image objects.YOLO for object segmentation is employed to identify and isolate image regions for style transfer.

# Sample Output
![stylized_image (1)](https://github.com/user-attachments/assets/2541c028-69fd-427c-8000-4ecef37ff2ac)
![stylized_image (2)](https://github.com/user-attachments/assets/063e6424-6d48-48f4-b168-f8615623353e)
![stylized_image (3)](https://github.com/user-attachments/assets/0fa05426-8147-4b69-9d84-062525acee30)

# How to run
## Prerequisites
Ensure you have the following installed on your system:

- Python 3.7 or higher
- pip (Python package installer)

## Install Dependencies
```
pip install -r requirements.txt
```
## Run main.py
```
streamlit run main.py
```
## Run Website
You can also access the deployed version of the application [here](https://background-stylizer.streamlit.app/).
# Possible Improvements
- model may fail for multiple people hence build a mask and segment out all YOLO detection where label is person
- using different style transfer techniques(which would increase runtime) for better results

# References
- [Fast Style Transfer for Arbitrary Styles](https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization)
- [How to implement instance segmentation using YOLOv8 neural network](https://dev.to/andreygermanov/how-to-implement-instance-segmentation-using-yolov8-neural-network-3if9)
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
