# Assignment: Deep Learning

## Problem 1 (20 points)

In this problem you work with the IMDB dataset, a set of 50,000 reviews from the Internet Movie Database.
You will perform the following experiments and compare the performance to the model implemented earlier in the lecture notes. Please refer to lecture notes for pre-processing steps.

1. We used two hidden layers in our deep NN model. Try using one and three hidden layers. Report how doing so affects validation and test accuracy.
2. We used 16 hidden neurons in our example. Try using 2 layers with more hidden units (32 and 64 units). Report how doing so affects validation and test accuracy.
3. We used binary-crossentropy for the loss function in our example. Try using the mse loss function and report how doing so affects validation and test accuracy.
4. Try using the tanh activation instead of relu and report how doing so affects validation and test accuracy.
Round your answers to 3 decimal places.

## Problem 2 (35 points)

In this problem you work with the boston housing dataset (available in R and Python) and predict the median price of homes in a given Boston suburb in the mid-1970s, given data points about the suburb at the time, such as the crime rate, the local property tax rate, etc..
1. Load and preprocess the data using Gaussian normalization.
2. Create a deep NN with two 64-unit dense layers. What are your metric and loss function when compiling the model?
3. Since the data set is small, you will use k-fold cross-validation instead of the hold-out. Use 4-fold cross-validation to fit the model for epochs = 500 and batch size = 1. Report the MAE on the 4 validation sets. Is there a lot of variation in MAE from one fold to another?
4. Plot the average of the per-epoch MAE scores for all folds vs the epoch number (Hint: use history). Based on this plot around which epoch should the training had stopped?
5. Train a final production model on all of the training data, with the ap- propriate number of epochs (from above), and report its performance on the test data.
