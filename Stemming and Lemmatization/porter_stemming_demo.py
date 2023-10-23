import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')

'''
This code is a Python script that demonstrates how to remove punctuation, tokenize a sentence,
and perform stemming using the Natural Language Toolkit (nltk) library.

• First, the script imports the string and word_tokenize modules from the nltk library.

• It also defines an example sentence as a string variable.

• Next, the script removes the punctuation from the example sentence using the translate() method and the str.maketrans() function.

• This creates a new string variable called example_sentence_no_punct.

• After that, the script tokenizes the example_sentence_no_punct variable using the word_tokenize() method.

• This creates a list of word tokens called word_tokens.

• Finally, the script performs stemming on each word in the word_tokens list using the PorterStemmer algorithm from the nltk library.

• It then prints out each word and its corresponding stem using a formatted string.

• Overall, this code demonstrates how to use the nltk library to preprocess text data by removing punctuation, tokenizing sentences, 
and performing stemming.

'''


ps = PorterStemmer()

example_sentence = "Python programmers often tend like programming in python because it's like english. We call people who program in python pythonistas."

# Remove punctuation
example_sentence_no_punct = example_sentence.translate(str.maketrans("", "", string.punctuation))

# Create tokens
word_tokens = word_tokenize(example_sentence_no_punct)

# Perform stemming
print("{0:20}{1:20}".format("--Word--","--Stem--"))
for word in word_tokens:
    print ("{0:20}{1:20}".format(word, ps.stem(word)))

"""
RESULT:
--Word--            --Stem--            
Python              python              
programmers         programm            
often               often               
tend                tend                
like                like                
programming         program             
in                  in                  
python              python              
because             becaus              
its                 it                  
like                like                
english             english             
We                  we                  
call                call                
people              peopl               
who                 who                 
program             program             
in                  in                  
python              python              
pythonistas         pythonista  
"""