from tensorflow.keras.models import load_model
import io
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import array_to_img
import whatimage
import pyheif
import tensorflow as tf
import matplotlib.cm as cm
from PIL import Image
import io
import base64

model = load_model('func_model_2.h5')


def decode_image(bytesIo):
    try:
        fmt = whatimage.identify_image(bytesIo)
        if fmt in ['heic','heif']:
            i = pyheif.read_heif(bytesIo)
            pi = Image.frombytes(mode=i.mode, size=i.size, data=i.data)
        else:
            pi = Image.open(io.BytesIO(bytesIo))
        return pi
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None
    
@staticmethod
def img_to_base64(img_array):
    img_pil = Image.fromarray(np.uint8(img_array))
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG')
    img_bytes = base64.b64encode(buffer.getvalue())
    return img_bytes.decode('utf-8')



@staticmethod
def get_img_array(img, size=(224, 224)):
    # `img` is a PIL image of size 224x224
    img = img.resize(size=size)
    # `array` is a float32 Numpy array of shape (224, 224, 3)
    array = img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    # array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
###
###
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer 
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def get_gradcam(img, heatmap, alpha=0.4):

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)
    return superimposed_img

@staticmethod
def get_result(image: Image.Image,alpha=0.4):
    img = decode_image(image)
    orignal_image = img_to_array(img.resize(size=(224,224)))
    img = np.expand_dims(orignal_image, axis = 0)
    img = img / 127.5 - 1.0    
    result = model.predict(img)
    # Remove last layer's softmax
    vis_model = model
    vis_model.layers[-1].activation = None

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img, vis_model, 'Conv_1')
    sup_img = get_gradcam(orignal_image,heatmap)
    return result, sup_img

