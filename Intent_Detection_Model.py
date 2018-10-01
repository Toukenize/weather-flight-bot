import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
from keras.models import load_model
from gensim.models import KeyedVectors
	
intent_dict = {0:'flight',1:'weather',2:'other'}
	
try:
    print('Loading Gloves (Global Vectors) for word embeddings.. \nThis will take up to 2 minutes due to the size of vocabulary (400,000 words!)')
    model = KeyedVectors.load_word2vec_format('glove-wiki-gigaword-300.txt')
    print('Sucessfully loaded Gloves!')

except FileNotFoundError:
    print('Unable to load file \'glove-wiki-gigaword-300.txt\'...\nPlease make sure the file is placed in the same directory and restart the program')

print('Loading NN models (Intent model & Label model)...')

try:
    intent_model = load_model('lstm_intent.h5')
    print('Successfully loaded Intent model!')

except (FileNotFoundError, OSError) as e:
    print('Unable to load file \'lstm_intent.h5\'\nPlease make sure the file is placed in the same directory and restart the program')
    
try:
    label_model = load_model('lstm_label.h5')
    print('Successfully loaded Label model!')
except (FileNotFoundError, OSError) as e:
    print('Unable to load file \'lstm_label.h5\'\nPlease make sure the file is placed in the same directory and restart the program')
	
def input_to_intent(words):
    
    embedded_input = np.zeros((1,50,300))

        
    for index, word in enumerate(words):
        try:
            embedded_input[0][index] = model.get_vector(word)
        except KeyError:
            embedded_input[0][index] = model.get_vector('unk')
    
    outputs = intent_model.predict(embedded_input)
    
    intent = intent_dict[np.argmax(outputs[0])]
    
    return intent

def input_to_label(words):    
            
    embedded_input = np.zeros((1,50,300))

    
    for index, word in enumerate(words):
        try:
            embedded_input[0][index] = model.get_vector(word)
        except KeyError:
            embedded_input[0][index] = model.get_vector('unk')
    
    output = label_model.predict(embedded_input)
    
    #print(output.shape)
    
    label_list = []
    
    destination_index, origin_index, location_index = [], [], []
    
    
    for index, label in enumerate(output[0]):
        
        label_id = np.argmax(label)
        
        #print(label,label_id)
        
        if label_id == 3:
            destination_index.append(index)
        elif label_id == 4:
            origin_index.append(index)
        elif label_id == 1:
            location_index.append(index)
            
    return destination_index, origin_index, location_index

def predict_user_input(x):
    x = x.strip().lower()
    
    words = x.split()
    
    
    intent = input_to_intent(words)
    destination_index, origin_index, location_index = input_to_label(words)
    
    #print('intent:',intent)
    #print('destination_index:', destination_index)
    #print('origin_index:', origin_index)
    #print('location_index:', location_index)
    
    
    if intent == 'flight' and (len(origin_index) > 0 or len(destination_index) > 0)	and len(location_index) == 0:
                             
        destination = ''
        origin = ''
        
        for i in destination_index:
            destination += words[i] + ' '
        
        for i in origin_index:
            origin += words[i] + ' '
            
        print('\n{')
        print('  \"intent\": \"flight_booking_intent\",')
        print('  \"slots\":')
        print('\t[{')
        print('\t  \"name\": \"origin\",')
        print('\t  \"value\": \"{}\",'.format(origin.strip()))
        print('\t},')
        print('\t{')
        print('\t  \"name\": \"destination\",')
        print('\t  \"value\": \"{}\",'.format(destination.strip()))
        print('\t}]')
        print('}\n')

    elif intent == 'weather' and len(destination_index) == 0 and len(origin_index) == 0 and len(location_index) > 0:
                             
        location = ''
      
        for i in location_index:
            location += words[i] + ' '
            
        print('\n{')
        print('  \"intent\": \"weather_intent\",')
        print('  \"slots\":')
        print('\t[{')
        print('\t  \"name\": \"city\",')
        print('\t  \"value\": \"{}\",'.format(location.strip()))
        print('\t}]')
        print('}\n')


        
    else:
        print('\nSeems like you are not asking about flight or weather...\n')

print('\n\nWelcome! Ask me anything about flight and weather!\n(Type \'quit\' to exit the program)\n')

user_input = ''

while user_input.lower() != 'quit':
    user_input = input('Query (to quit, type quit) :')
    if user_input.lower() != 'quit':
        predict_user_input(user_input)
    else:
        print('\nThank you, see you again!\n')
        quit()