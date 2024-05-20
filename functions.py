import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

def load_and_preprocess_image(image, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses an image from an array."""
    if isinstance(image, np.ndarray):
        img = image.astype(np.float32) / 255.0
    else: 
        img = np.array(image).astype(np.float32) / 255.0
    img = img[np.newaxis, ...]

    img = crop_center(img)

    if preserve_aspect_ratio:
        img = tf.image.resize_with_pad(img, image_size[0], image_size[1])
    else:
        img = tf.image.resize(img, image_size)

    return img

def save_image(image, filename):
    """Saves a TensorFlow image to a file."""
    image = tf.squeeze(image, axis=0)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        encoded_image = tf.io.encode_jpeg(image)
    elif filename.endswith('.png'):
        encoded_image = tf.io.encode_png(image)
    else:
        raise ValueError("Filename extension not supported. Use .jpg, .jpeg or .png")
    tf.io.write_file(filename, encoded_image)