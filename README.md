# Hypothesis-Testing_Textual-Entailment
## Overview
The project is synonymous to Hypothesis testing. It is an application of natural language processing, where for a given pair of sentences, we verify whether the facts in the first sentence imply the facts in the second sentence. The first sentence is always considered to be true.This entailment between these two sentences can be Positive, Neutral, Negative.  
Positive can also be called entailment.  
Negative entailment can be called a contradiction.  
**Example:**  
**Evidence 1**: I was driving on a lonely road.  
**Hypothesis 1**:  A black car overtook and ran into a crowd of people.  
**Result 1**: Negative Entailment  
**Evidence 2**: I was running in the park.  
**Hypothesis 2**:  Daisy was sleeping on the couch.  
**Result 2**: Neutral Entailment  
**Evidence 3**:Meena and I were buying ice-cream from the stall.  
**Hypothesis 3**:  There were people at the ice-cream stall.  
**Result 3**: Positive Entailment  

## Dataset Used
* The Stanford Natural Language Inference(SNLI) Corpus: The SNLI corpus is a collection of 570k human-written Eng sentence pairs manually labeled for balanced classification with the labels supporting the task of textual entailment (RTE).
Example on SNLI website: [Link](https://nlp.stanford.edu/projects/snli/)
* GloVe: Global Vectors for Word Representation: GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. 
Example on GloVe website: [Link](https://nlp.stanford.edu/projects/glove/)
![image](https://user-images.githubusercontent.com/46564084/100749194-5aa4d000-340a-11eb-94b4-bfae54904ff4.png)

## Techstack used
* Python
* TensorFlow
* Numpy
* LSTM Cell
* Tqdm ( for progress bars)    

## Process of the project
* Step 1  
**Data collection and preparation**  
Import required packages, download GloVe dataset and SNLI dataset and unzip those files. Prepare the data by converting the vectors in GloVe dataset and converting them to a python dictionary.  
We also create a function sentencetoSequence which takes in a sentence and returns the GloVe vector representation of the entire sentence.   
* Step 2
**Define Constants and functions**
Here we mainly define all our constants that we would use in our LSTM layer. Those are max length of hypothesis sentence and evidence sentence, batch size, vector size, hidden size, learning rate, iteration counts for training, and probabilities.  
We also have two functions, first score_setup for returning the score in the form of [0.42,0.84,0.62], where the highest probability denotes which entailment it would be. 0 index is positive entailment, 1st index is neutral entailment and last is negative entailment. Another function is fit_to_size for resizing the input matrix to a given shape, it trims out the extra rows and columns.  
We have our main function split_data_into_scores: this function loads the SNLI data (subset of it) and appends all the hypothesis, evidence sentences’ glove vectors along with the correct labels and respective score computed by the previous function.  
* Step 3  
**LSTM layer, Accuracy, Loss functions & Optimizer**
We create classification_scores which is matrix multiplication of outputs from cells in RNN  and weight, and then add the bias. (like in neural networks)  
RNN_output is obtained from a bidirectional rnn. We also create  2 LSTM cells (front and back), and a dropoutWrapper, we also pre-initialize variables required for the tensorflow session, data is assigned later.  
Next we make an Accuracy variable scope and Loss variable scope by checking the mean of the number of correct labels and cross_entropy_with_logits functions respectively. Finally an optimizer is defined using Gradient Descent Optimizer using the loss calculated in Loss variable scope.  
* Step 4  
**Train and Test**   
Train: We initialize the tensorflow session, and also use tqdm for displaying the progress. Pick random values from the features list to create the feed_dictionary and the session is run with the optimizer declared before and a feed_dictionary that contains hyp, evi and labels   
Test: We send in a sample hypothesis and evidence sentence which calculate the prediction score and return the argmax the scores. Highest scores denote the correct entailment.  

## Final Results of the Project
![image](https://user-images.githubusercontent.com/46564084/100748858-e79b5980-3409-11eb-8d87-b9a31c09c7db.png)
I have used tqdm to show the progress bars while further optimizing the model. Minibatch loss is calculated using the gradient descent function, which further tries to increase the accuracy. We feed in a dictionary of all training data and correct labels, run the optimization.  
![image](https://user-images.githubusercontent.com/46564084/100748944-0568be80-340a-11eb-907a-1ec248a9afea.png)
Testing the model with an arbitrary example. Here hypothesis sentence follows the evidence sentence, hence this situation is possible and the result is thus Positive entailment.
There are more examples in the file above.

## Note
I have referred to various sites internet to code and also tried understanding the reasoning behind them.  






