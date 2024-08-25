# Machine Learning with Python Certification - freeCodeCamp

This repository contains the solutions to the five projects I completed as part of the **Machine Learning with Python Certification** from freeCodeCamp. Each project showcases various machine learning techniques, from basic classifiers to neural networks.

## Project Directory Structure

- **Rock Paper Scissors**
  - Directory: `fcc-RockPaperScissors`
    - `RPS.py`: Main function for predicting the next move.
    - `RPS_game.py`: Contains the game logic.
    - `main.py`: Used to test the implementation of the game.
    - `test_module.py`: Unit tests for the project.

- **Cat and Dog Image Classifier**
  - File: `fcc_cat_dog.ipynb`
  - Uses TensorFlow and Keras to classify images of cats and dogs with a Convolutional Neural Network (CNN).

- **Book Recommendation Engine using KNN**
  - File: `fcc_book_recommendation_knn.ipynb`
  - Implements a book recommendation system using the K-Nearest Neighbors (KNN) algorithm.

- **Linear Regression Health Costs Calculator**
  - File: `fcc_predict_health_costs_with_regression.ipynb`
  - Predicts healthcare costs using a linear regression model.

- **Neural Network SMS Text Classifier**
  - File: `fcc_sms_text_classification.ipynb`
  - Classifies SMS messages as "spam" or "ham" using a neural network.

## Project Descriptions

### 1. Rock Paper Scissors
The goal of this project is to create a program that plays the game of Rock, Paper, Scissors. The program must compete against four different bots and win at least 60% of the matches. The implementation uses strategies that adapt based on the opponent's previous moves.

**Development:**
- Implement your strategy in `RPS.py`.
- Use `main.py` for testing and `test_module.py` for unit tests.

### 2. Cat and Dog Image Classifier
In this project, I built a Convolutional Neural Network (CNN) to classify images of cats and dogs. The model was trained on a dataset with 4,000 images (2,000 cats and 2,000 dogs) and evaluated on a validation set of 1,000 images.

**Key Points:**
- Achieved an accuracy of over 63%.
- Used data augmentation to reduce overfitting.

### 3. Book Recommendation Engine using KNN
This project involves building a book recommendation system using the K-Nearest Neighbors (KNN) algorithm. The model recommends books similar to a given book based on user ratings.

**Key Points:**
- Worked with the Book-Crossings dataset containing 1.1 million ratings.
- Implemented the `get_recommends` function that returns five similar books to a given title.

### 4. Linear Regression Health Costs Calculator
In this project, I implemented a linear regression model to predict healthcare costs based on various features such as age, BMI, and smoking status. The goal was to achieve a Mean Absolute Error (MAE) of under $3500.

**Key Points:**
- Processed and converted categorical data for regression.
- Achieved the desired MAE threshold.

### 5. Neural Network SMS Text Classifier
This project involves creating a neural network model to classify SMS messages as either "ham" (not spam) or "spam". The model uses text data and is trained to distinguish between the two classes.

**Key Points:**
- Used the SMS Spam Collection dataset.
- Implemented a `predict_message` function that returns the classification and probability score.

## Certification Details

**Certification:** Machine Learning with Python - freeCodeCamp  
**Completion Date:** August 22, 2024  
**Verification:** [freeCodeCamp Certification](https://freecodecamp.org./certification/DaVerm/machine-learning-with-python-v7)

## Other Projects

In addition to these certification projects, I have worked extensively on various machine learning, data science, and image classification projects using Python. Here are a few highlights:

- **Fine-tuning Meta Llama 2 7B for Financial Expertise**: Used AWS SageMaker to fine-tune a large language model for financial domain applications. [GitHub Repository](https://github.com/DakshVerma11/Financial-LLM-Fine-Tuning-AWS)
- **Image Classification with Data Augmentation**: Implemented image classification for cats vs. dogs using data augmentation and TensorFlow. [GitHub Repository](https://github.com/DakshVerma11/Cats_Vs_Dogs-CNN-Image_Classification)
- **Custom and Pretrained CNN Models for Flower Classification**: Developed models using custom CNN and pretrained models like MobileNet V2 and Inception V3 for flower image classification. [GitHub Repository](https://github.com/DakshVerma11/Flower-Image-Classification-CNN)
- **Fashion MNIST and MNIST Classification**: Worked on classifying clothing articles and digits using CNNs. [Fashion MNIST Repository](https://github.com/DakshVerma11/Fashion-MNIST-Classification) | [MNIST Repository](https://github.com/DakshVerma11/MNIST-Classification-Basic)
- **Time Series Forecasting with CNNs and LSTMs**: Applied deep learning techniques for time series forecasting. [GitHub Repository](https://github.com/DakshVerma11/Time-Series-Forecasting)
- **NLP with Text Generation**: Created a text generation model using the Kaggle Song Lyrics dataset. [GitHub Repository](https://github.com/DakshVerma11/First_NLP)

---

Feel free to explore the projects and their code in this repository. Contributions, suggestions, and discussions are welcome!

