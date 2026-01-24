MRI Image Analysis

Overview:
This project was created as a learning exploration into Machine Learning and Deep Learning techniques for analyzing brain MRI scans. Through this project, I learned how to:
- load and preprocess MRI images
- extract Intensity and GLCM features
- train ML models (SVM, Random Forest Classifier, Logistic Regression, Easy Ensemble Classifier)
- build deep learning architectures (FCNN, CNN, VGG, Autoencoder, and U-Net)
- perform image segmentation (Binary, Otsu, Watershed)
- evaluate model performance (Confusion Metrics) 

The goal was to understand the full workflow of MRI analysis. 

Project Contents:
This repository includes:
- images/ - MRI scans
- masks/ - tumor segmentation masks
- sub_dataset/ - extracted smaller pathces of images
- requirements.txt - Python dependencies
- Jupyter Notebooks for each stage:
    - dataset building
    - feature extraction
    - ML Models
    - Image Segmentation
    - FCNNs and CNNs
    - VGG
    - Autoencoder
    - U-Net segmentation
Each notebook focuses on a specific concept or model.

Models Implemented:
1. Machine Learning Models (Random Forest, SVM, Logisitic Regression)
   These models operate on features extracted from MRI images including texture, intensity statistics, etc.
   Logistic Regression:
   A classifier that models the probability of a class using the sigmoid function. It learns a weighted combination of input    features and creates a linear decision boundary.

   Support Vector Machine (SVM):
   It finds the optimal hyperplane that maximizes the margin between classes. With kernels like RBF, SVM can model nonlinear    boundaries.

   Random Forest:
   An ensemble of decision trees trained on random subsets of data and features. The final prediction is the majority vote.

   Easy Ensemble Classifier:
   It creates multiple balanced subsets of data and trains and AdaBoost model on each, finally combining them into a final      ensemble. It works well for severe class imbalance.

2. Fully Connected Neural Network (FCN)
A deep learning baseline where each neuron is connected to every neuron in the next layer.
How it works:
- Flattens the MRI image into a 1D vector
- Passes it through dense layers with nonlinear activations
- Learns global intensity patterns

3. Convolutional Neural Network (CNN)
A deep learning architecture designed specifically for image data. Convolutional layers slide filters across the image to detect local patterns such as edges, textures, and shapes.
How it works:
- Convolutions: extract spatial features
- Pooling: reduces spatial dimensions while keeping important information
- Fully connected layers: perform final classification

CNNs learn hierarchical representations — from low‑level edges to high‑level anatomical structures.

4. Autoencoder
An unsupervised neural network that learns a compressed representation of MRI images by reconstructing them.
How it works:
- Encoder: compresses the image into a low‑dimensional latent vector
- Bottleneck: stores the most essential information
- Decoder: reconstructs the original image from the latent vector
- Goal: minimize reconstruction error and learn meaningful features

5. VGG
A deep convolutional architecture that stacks many small 3\times 3 filters to learn increasingly complex features.
How it works:
- Small convolutions: capture edges, textures, and shapes
- Deep stacking: builds hierarchical feature representations
- Max pooling: reduces spatial size while preserving important patterns
- Dense layers: perform final classification

6. U-Net Segmentation
A U‑shaped architecture designed for precise pixel‑level segmentation of medical images.
How it works:
- Encoder (downsampling): learns context through repeated conv + pooling
- Bottleneck: captures high‑level abstract features
- Decoder (upsampling): restores spatial detail
- Skip connections: combine encoder and decoder features to preserve fine structure


     
