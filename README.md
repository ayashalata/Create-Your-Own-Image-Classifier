# Create-Your-Own-Image-Classifier

This is the second project of udacity nano degree for ML, the goal is to classify flowers according to their actual names using CNN. 

* `Project_Image_Classifier_Project.ipynb` the notebook file of the entire project.
* 'Project_Image_Classifier_Project.html' HTML file of the jupyter notebook.

* `Predict.py` impelmented application for the project, you can use terminal to check the results.


### How to use the application:

` predict.py --input 'the path of the image' --model 'the model path'  --top_k 'number of classes' --category_names 'path of classes names`
in your terminal window. 

You should be able to see the corresponding predictions probabilities with their label names according to the number of classes you've wrote in `top_k`, and in the last the application will print the best prediction of the flower, which should be its actual name. 

OR 

just type `predict.py` and  will run the saved model with 5 top classes for orange dahlia flower by default. 
