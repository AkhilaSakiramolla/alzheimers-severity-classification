# alzheimers-severity-classification
Developing machine learning and deep learning models to analyze MRI scans and classify them based on the severity of the Alzheimer's disease.

We used a Kaggle dataset (https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images), that was divided into four 
separate classes (Mild Demented, Moderate Demented, 
Non-Demented, Very Mild Demented) based on the 
severity of the disease. The dataset consists of 6400
images, each image with size of 208x176 pixels as we 
were able to get better performance compared to other 
combinations.

We have implemented three established machine learning models which are, Convolutional Neural Network, Transfer Learning and Random Forest, to provide with concrete results of classifying MRI scans 
to determine if the patient is detected with ‘Very Mild’, ‘Mild’, ‘Moderate’ level of dementia or is not detected with dementia. Our implementation gave the accuracy for
Convolutional Neural Network as 86%, the accuracy for ResNet as 84% and Random Forest as 75% approximately. In our work we concluded that Convolutional Neural Network model is the most efficient for this purpose and provides reliable results. 
