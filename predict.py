import argparse
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from PIL import Image
import json
import warnings
import os 

# Ignore some warnings that are not necessary
warnings.filterwarnings('ignore')

#first we need to reload our model we want to use
def load_saved_model(saved_model_filepath):
    reload_model = tf.keras.models.load_model(saved_model_filepath,custom_objects={'KerasLayer':hub.KerasLayer},compile = False)
    
    return reload_model

# second, prepare our input images, we can change the image_size according to our input 
def process_image(numpy_i):
    image_size=224
    tensor_i=tf.convert_to_tensor(numpy_i,tf.float32)
    tensor_i = tf.image.resize(tensor_i, (image_size, image_size))
    tensor_i /= 255
    return tensor_i.numpy()

#third, implement a prediction function that returns best probabilites with their corresponding top classes

def predict(image_path, model, top_k): 
        im = Image.open(image_path)
        im_numpy = np.asarray(im)
        processed_image = process_image(im_numpy)
        processed_image = np.expand_dims(processed_image,axis = 0)
        prediction=model.predict(processed_image)
    
        classes, probs = tf.nn.top_k(prediction, k=top_k)
  
        return classes, probs

def loading_classes_names(classes_names_path):
    with open(classes_names_path, 'r') as f:
        return json.load(f)
     
                

    
   # fourth, use the args module to use our commands

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict a flower type')

    parser.add_argument('--input', action='store', dest='input_i', default='./test_images/orange_dahlia.jpg')
    parser.add_argument('--model', action='store', dest='model', default='./saved_model.h5')
    parser.add_argument('--top_k', action='store', dest='top_k', default=5, type=int)
    parser.add_argument('--class_names', action='store', dest="class_names", default='./label_map.json')

    args = parser.parse_args()
    
    # run the commands 
    image_path = args.input_i
    top_k = args.top_k
    classes_names_path = args.class_names
    my_model=args.model

    model = load_saved_model(my_model)
    classes, probs = predict(image_path, model, top_k)
    class_names = loading_classes_names(classes_names_path)
    
def print_the_info(top_k,classes_names,probs,classes):
    CLASSES = []
    PROBS = []
    for n in range(top_k):
            for class_i in classes.numpy():
                #print(classes_names[(class_i[n])]) #check the values
                CLASSES.append(class_names[str(class_i[n]+1)])
                print('Class: {}'.format((n+1)))            
                print(CLASSES[n])
            for probs_i in probs.numpy():
                #print(probs_i[n]) #check the values
                PROBS.append(probs_i[n])
                print('With a probababilty of:')            
                print(PROBS[n])
    print('---THE FLOWER IS:----')
    print(CLASSES[0])
                
                
print_the_info(top_k,classes_names_path,classes,probs)
 

