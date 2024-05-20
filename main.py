import streamlit as st
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from functions import crop_center,load_and_preprocess_image,save_image
from PIL import Image
from ultralytics import YOLO
import cv2


hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

st.title("Background Stylizer")

with st.form(key='image_upload_form'):
    contentimg = st.file_uploader("Upload content image", type=["jpg", "jpeg", "png"], key='image1', help="An image of you to stylize")
    styleimg = st.file_uploader("Upload Style image", type=["jpg", "jpeg", "png"], key='image2', help="Image use to stylize your background")

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if contentimg is not None and styleimg is not None:
        content=Image.open(contentimg)
        style=Image.open(styleimg)
        content_img=load_and_preprocess_image(image=content)
        style_img=load_and_preprocess_image(image=style)
        outputs = hub_module(content_img, style_img)
        stylized_image = outputs[0]
        save_image(stylized_image, 'stylized_image.jpg')
        background = Image.open('stylized_image.jpg').convert('RGBA')
        resized_image = content.resize((256,256))
        model = YOLO("yolov8m-seg.pt")
        results = model.predict(resized_image)
        result = results[0]
        masks = result.masks
        mask1 = masks[0]
        polygon = mask1.xy[0]
        resized_array = np.array(resized_image)
        masky = np.zeros_like(resized_array[:, :, 0])
        cv2.fillPoly(masky, [polygon.astype(np.int32)], 255)
        overlay = cv2.bitwise_and(resized_array, resized_array, mask=masky)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        b, g, r = cv2.split(overlay)
        alpha = masky 
        overlay_rgba = cv2.merge((r, g, b, alpha))
        overlay_image = Image.fromarray(overlay_rgba).convert('RGBA')
        if background.size != overlay_image.size:
            raise ValueError("The images must have the same dimensions")
        combined = Image.alpha_composite(background, overlay_image)
        st.image(combined, caption='Combined Image')
    else:
        st.error("Please upload both images")
