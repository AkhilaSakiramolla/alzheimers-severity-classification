# Alzheimers Severity classification

Developed machine learning and deep learning models to analyze MRI scans and classify them based on the severity of the Alzheimer's disease.

I used a Kaggle dataset (https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images), that was divided into four 
separate classes based on the severity of the disease:
  1) Mild Demented
  2) Moderate Demented
  3) Non-Demented
  4) Very Mild Demented

The dataset consists of 6400
images, each image with size of 208x176 pixels as I 
was able to get better performance compared to other 
combinations.

I have implemented three established machine learning models which are, Convolutional Neural Network, ResNet-152 and Random Forest, to provide with concrete results of classifying MRI scans to determine if the patient is detected with ‘Very Mild’, ‘Mild’, ‘Moderate’ level of dementia or is not detected with dementia. This implementation gave the accuracy for Convolutional Neural Network as 86%, for ResNet as 84% and Random Forest as 75% approximately. In this work, I concluded that Convolutional Neural Network model is the most efficient for this purpose and provides reliable results. 
