import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import argparse

def get_grad_cam_heatmap(model, img_array, class_index):
    # Get the model's last convolutional layer
    last_conv_layer = model.get_layer('conv_pw_13')
    
    # Create a model that maps the input image to the activations of the last conv layer and predictions
    grad_model = Model([model.inputs], [last_conv_layer.output, model.output])
    
    # Get the gradients of the loss with respect to the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    
    # Compute the guided gradients
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Compute the heatmap
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, guided_grads), axis=-1)
    
    # Normalize the heatmap to the range [0, 1]
    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def display_grad_cam(img_path, model, class_labels, target_size=(32, 32)):
    # Load and preprocess the image
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get the model's predictions
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    
    # Get the Grad-CAM heatmap
    heatmap = get_grad_cam_heatmap(model, img_array, class_index)
    
    # Rescale the heatmap to the size of the input image
    heatmap = cv2.resize(heatmap, (target_size[0], target_size[1]))
    heatmap = np.uint8(255 * heatmap)
    
    # Convert the heatmap to RGB
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (target_size[0], target_size[1]))
    
    # Overlay the heatmap on the original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Display the image with the Grad-CAM heatmap
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Predicted Class: {class_labels[class_index]}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualization for Fashion MNIST model.')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to the input image.')
    args = parser.parse_args()

    class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    model = load_model('fashion_mnist_mobilenet.h5')

    display_grad_cam(args.image_path, model, class_labels)
