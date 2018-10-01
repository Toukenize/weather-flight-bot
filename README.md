# The Challenge - Weather? Flight? Other?

In this project, I'm going to build a model that, given a input sentence, predicts the user's intention (weather intent, flight intent or other) and attempts to do semantic slot filling for the corresponding intent. 

Some examples are:
## Weather Intent
Input Sentence:	Weather forecast **Bangkok**.

Slot Filling:		O-O-**Location**

Intent:					Weather Intent

## Flight Intent
Input Sentence:	I would like to get a flight from **Singapore** to **Malaysia**.

Slot Filling:		O-O-O-O-O-O-O-O-**Origin**-O-**Destination**

Intent: 				Flight Intent

## Other (random sentence)

Input Sentence: Where can I get the best Ramen in Japan?

Slot Filling:		O-O-O-O-O-O-O-O-O

Intent:					Other Intent

The neural network model will be built using **Keras** with **Tensorflow** backend, the workflow and approach will be shared in further details in the following sections

# Finding Suitable Data to Model the Problem

## Flight Data

For flight data, a set of nicely annotated data on Air Travel Information Services (ATIS) collected in 1995 under the Advanced Research Projects Agency Spoken Language Systems (ARPA-SLS) technology development program is available on the [Linguistic Data Consortium](https://catalog.ldc.upenn.edu/LDC95S26) website. I used that as the baseline data for my neural network model.

In the original data set, named-entity-like labels are used on the utterances. Many of the labels used (e.g. B-flight_time, B-depart_data.day_name) are not necessary if I intend to train a model on detecting the flight intent, the semantics slots of 'origin' and 'destination'. Thus, I simplified the labels to ‘O’, ‘origin’ and ‘destination’. 

Also, for the intent, there are many categories of intent, I simplified the data by sieving out the ‘flight’ intents and classify the rest as ‘other’ intent.

As for the input sentences, I did not want my model to focus on numerical time as a feature. Therefore, I replaced all the time information (mainly digits) with the string ‘_D_’. 

In terms of words representation, I tried word encoding (one-hot sparse vector) and pre-trained word embedding (Global Vector, abbreviated as GloVe). Models trained on encoded inputs did not perform very well, and when the training epochs is increased, it overfitted the data and did poorly in the unseen data. The model trained on embedded inputs (using pre-trained GloVe embeddings that converts each word to a 300-dimension vector) saw a much better performance, and thus this approach was used for subsequent modelling.

Upon several iterations of modelling, I realize that the model could only identify cities that it was trained on (e.g. Boston, Denver). 

![image](https://user-images.githubusercontent.com/43180977/46293102-18329f00-c5c5-11e8-9ace-084b64724d2d.png)
###### Figure 1 – Outputs for similar sequences with seen cities (above) and unseen cities (below)

To tackle this, I modified the ATIS data by making use of a list of top 100 most visited cities (which consists of a variety of cities across the world) and randomly paired them with the original data (both flight intent and non-flight intent), to obtain a new set of data. The outcome was a pleasant surprise! 

![image](https://user-images.githubusercontent.com/43180977/46292306-2aabd900-c5c3-11e8-9779-b4295062990f.png)
###### Figure 2 – Outputs for similar sequences with seen (green) and unseen (red) cities

## Weather Data

For the weather data, there wasn’t any suitable data set that I could use. Thus, I prepared the training data on my own, by listing down different ways to ask about weather, and generated the training data by randomly pairing the weather questions and the top 100 cities used for the flight data earlier.

## Other Data

For the intent detection part of my task, I modelled it as a classification problem. I needed some data that falls under ‘other’ (i.e. neither flight nor weather), to properly train my model. I simply made use of my modified ATIS data that were classified as ‘other’ and replace the cities with the top 100 cities used for flight and weather data. This trains my model not to focus so much on the cities while predicting the user intent, as the sentences with the same cities can be ‘flight’, ‘weather’ or ‘other’ intent, depending on the context. 

One shortfall of these data is that they were part of the ATIS data, thus the domain is very much airline-specific. In other words, the trained model most probably won’t do well in classifying random non-airline-specific sentences which intent falls under the ‘other’ category. A way to deal with this is to gather more general utterances for the model training, but the data scraping would be time consuming, so I will leave this to another day. 

I took a shortcut by combining the output of both my intent model (predicting intent) and label model (predicting labels). When both models give me coherent outputs (e.g. predicted intent = flight, either origin or destination is predicted as part of the labels), then the outputs are accepted and displayed to the user. Otherwise, the program would indicate intent as ‘other’ and get the user to try another query.

## Modelling Decisions

My gut feeling was to try out LSTM neural network and convolutional neural network (CNN). The former excels in sequential data and saw success in natural language processing tasks, while the latter is the go-to choice for data with spatial relationship. I also played around with some other variations of the models, such as stacking LSTM with CNN (in hope of getting the best of both world) and encoder-decoder generative model (said to be state-of-the-art for language translation).

A total of 3,600 data (1,200 of each intent) were used to train and validate these models (30% validation split), with ‘softmax’ activation for the output layers (both intent and label models) and ‘categorical cross entropy’ as the loss function. To prevent overfitting, a combination of dropout layers and L2 regularizations were used. 

For both the intent and label model, the stacked LSTM and CNN model did the best, but with slightly different configuration, as shown below:

![image](https://user-images.githubusercontent.com/43180977/46292325-3a2b2200-c5c3-11e8-9222-a52f6ae5f76d.png)
###### Figure 3 – Intent Model Architecture (Left), Label Model Architecture (Right)

Both of my models scored more than 99% in their respective test and validation accuracies. In other words, the models probably maxed out in terms of performance, with respect to the small data set I fed in. Thus, I did not do any hyperparameter tuning for this project. 

One thing to note is that the encoder-decoder model did not do as well (90+% accuracy, with poor generalization), probably because I did not manage to find the right set of parameters and apply attention layer due to its model complexity. Given more time and labelled data, the encoder-decoder model should out-perform my current LSTM-CNN model.

## Potential Areas of Improvement - Limitations with GloVe Representation

While my models are somewhat able to generalize to unseen locations or cities, it depends largely on the GloVe of the input locations. For locations or cities with very different GloVe from the selected 100 cities, my models are unlikely to recognise these as locations or cities.

Also, GloVE are representations of single phrase. For locations or cities with multiple phrases (e.g. Ho Chi Minh City – 4 phrases), my models are unlikely to predict their labels correctly.

Some possible solutions are:

1.	make use of more advance word embeddings (e.g. Named-Entity-Recognition)
2.	store a comprehensive list of cities/ locations and check the input against this list to identify the cities/ locations prior to feeding it to the neural networks
3.	train a separate neural network to automatically recognise the cities/ locations

Despite the limitation, I think the model managed a good job that can handle both flight and weather intents reasonably well!

## How To Run the Model?

1. Clone the repository (or download and unzip them to the same directory)
2. Download the GloVe (get the glove-wiki-gigaword-300.txt version) from [NLP Stanford](https://nlp.stanford.edu/projects/glove/) or [my GDrive](https://drive.google.com/open?id=19Nt4a5l1U8Oa2YJV5K_yDuilY8OXltLB). Make sure you rename it to **glove-wiki-gigaword-300.txt** and move it to the same directory as the remaining files) or my program won't work.
3. Create a new environment (python 3.6) and install the **requirements.txt**. Follow these if you use Anaconda on windows:
   - *conda create --name newenv python=3.6*
   - *activate newenv*
   - *pip install -r requirements.txt*
4. Run the program **Intent_Detection_Model.py**.
   - *python -m Intent_Detection_Model.py*
5. Input your queries and you should see the magic!
