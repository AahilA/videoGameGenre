# Classifying VideoGame Genres from their posters

In this project we are trying to classify the genres of video games from their posters. We created 56 genres.

## Model

The model is a ResNet 34 network. It was implemented in PyTorch. 

## Data Scraping

## Training and Testing

The dataset we get is first converted to RGB format if they are not, with 3 layers, and are resized to size 256 * 256. We then split the dataset into two sets: 80% are the training set and 20% are the testing set. We then train the model using SGD with momentum 0.9 and L1 loss. We then trained with 100 epoches using the ResNet 34 network we described above. Training and testing accuracies are calculated by dividing all correct classifications of genres by the total amount of classifications. 


