# Classifying VideoGame Genres from their posters

In this project we are trying to classify the genres of video games from their posters.

## Model

The model is a ResNet 34 network. It was implemented in PyTorch. 

## Data Scraping

We scraped steam library for the popular tags, genres and images of their games. We reduced the number of tags and genres to ones we thought would be possible to deduce from the videogames posters. We filtered out irrelevant games, booster packs, bundles, etc. We were able to collect slightly over 35000 titles. We then saved the game data into a csv which mapped their image name, title of the game and a multi-hot encoding for their genres. 

## Training and Testing

We used the above data to create our own custom dataset, in which we preprocessed all the images by normalizing and resizing to 256 x 256. We also had to change the number of channels of a black and white image to 3 for RGB. We then split the dataset into two sets: 80% are the training set and 20% are the testing set. We then train the model using SGD with momentum 0.9, learning rate of 0.1, weight decay of 0.0001 and L1 loss. We then trained with 100 epoches using the ResNet 34 network we described above. Training and testing accuracies are calculated by dividing all correct classifications of genres by the total amount of classifications. 


