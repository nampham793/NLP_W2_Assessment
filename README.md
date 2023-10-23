# NLP_W2_Assessment
NamPNHSE171025

## Week 2 Assessment:
1 - **Compare Euclidean distance and Cosine similarity**. Give **a few examples** where you **should use Euclidean distance instead of Cosine similarity**. <br>
2 - **Compare Stemming and Lemmatization**. There are several different algorithms for stemming, including the Porter stemmer, Snowball stemmer, and the Lancaster stemmer. **Choose a stemming technique and find out how it works.**

## Table of contents:
  - [ Euclidean distance and Cosine similarity ](#content_1)
  - [ Stemming and Lemmatization ](#content_2)
    
<a name="content_1"></a>
## I. Euclidean distance and Cosine similarity:
Euclidean distance and cosine similarity are two different mathematical concepts used to measure the similarity or dissimilarity between vectors in a multi-dimensional space. They are commonly used in various fields, including machine learning, data mining, and information retrieval. 

### 1. Euclidean distance:
**Euclidean Distance** measures the "as-the-crow-flies" distance between two points in a Euclidean space, which is essentially the straight-line distance between two points. It is calculated using the Pythagorean theorem and can be expressed as:<br>
```math
||x-y||_n = \sqrt {\sum_{i=1}^{n} (x_i - y_i)^2}
```


Where `x_i` and `yi` are the coordinates of two points in n-dimensional space.

**When to Use Euclidean Distance Instead of Cosine Similarity**:

  - **_Semantic Meaning and Magnitude Matter:_** If you want to consider both the semantic meaning and the magnitude of word vectors, Euclidean distance may be more appropriate. For example, in word embeddings like Word2Vec or GloVe, words are represented as vectors with both direction (semantic meaning) and magnitude. Euclidean distance can capture differences in both meaning and intensity, which might be relevant in certain NLP applications.
  - **_Word Vector Aggregation:_** In some cases, you may need to aggregate word vectors to represent phrases or sentences. Euclidean distance is more suitable when combining vectors because it accounts for the magnitude of the vector components. For example, when comparing the similarity of two sentences represented as the sum of their word vectors, Euclidean distance can be used to consider the magnitude of the combined vectors.
  - **_Document Comparison with Term Frequency:_** If you are working with document comparison tasks based on term frequency vectors (e.g., TF-IDF vectors), Euclidean distance can be used to measure the difference between documents. It considers the magnitude of term frequencies, which can be meaningful in document retrieval and clustering.
  - **_Analyzing Vector Representations of Text:_** In situations where you analyze the vector representations of text data using techniques like Doc2Vec or FastText, Euclidean distance might be useful. These embeddings often have magnitude information in addition to semantic meaning, and Euclidean distance can help capture differences effectively.
  - **_Dimensionality Reduction:_** When you are performing dimensionality reduction on text data using techniques like Principal Component Analysis (PCA), Euclidean distance can be used to measure distances in the reduced space, which may have both semantic and magnitude information.

### 2. Cosine Similarity

**Cosine Similarity** measures the cosine of the angle between two non-zero vectors in a multi-dimensional space. It quantifies the similarity of direction between two vectors, irrespective of their magnitudes. The formula for cosine similarity is:

```math
\cos(\theta) = \dfrac {A \cdot B} {\left\| A\right\|\left\| B\right\|} 
```
Where A and B are vectors, and `||A||` and `||B||` represent the magnitudes of those vectors.

**When to Use Cosine Similarity Instead of Euclidean Distance**:
  - **_Text Document Comparison:_** Cosine similarity is widely used in natural language processing (NLP) for comparing and measuring the similarity between text documents. In NLP tasks, documents are often represented as high-dimensional vectors where each dimension represents a term frequency or term frequency-inverse document frequency (TF-IDF) value. Cosine similarity is effective in this context because it focuses on the direction of the vectors, making it suitable for capturing the semantic similarity between documents while ignoring their length or magnitude.
  - **_Information Retrieval:_** When building search engines or recommendation systems, Cosine similarity is commonly used to match user queries with documents or items in a database. It helps find relevant documents that share similar keywords or topics with the query.
  - **_Text Clustering:_** Cosine similarity is often used in document clustering tasks, where documents with similar content are grouped together. It allows for the identification of clusters based on the similarity of the terms and concepts used in the documents.
  - **_Content-Based Recommendation Systems:_** In recommendation systems, Cosine similarity is used to recommend items (e.g., books, movies, products) to users based on the similarity between their preferences and the item descriptions or attributes.
  - **_Sparse Data:_** When dealing with high-dimensional, sparse data, where most elements in vectors are zero (e.g., in text data with a large vocabulary), Cosine similarity is more suitable. It emphasizes the similarity of non-zero elements while efficiently ignoring the zero entries in the vectors.
  - **_Dimensionality Reduction:_** In dimensionality reduction techniques like Latent Semantic Analysis (LSA) or Latent Dirichlet Allocation (LDA), Cosine similarity is employed to measure the relationships between documents in a reduced-dimensional space.
  - **_Topic Modeling:_** Cosine similarity is used in topic modeling to identify documents that share common themes or topics based on their term distributions.
  - **_Recommendation Systems in Collaborative Filtering:_** While Cosine similarity is mainly associated with content-based recommendation systems, it is also used in collaborative filtering. In this context, it helps identify users with similar preferences by comparing their rating vectors.
  - **_Natural Language Understanding:_** When determining the similarity between phrases, sentences, or documents, Cosine similarity is valuable. It can help assess how closely related two pieces of text are in terms of their content or semantics.





<a name="content_2"></a>
## II. Stemming and Lemmatization: 
### 1. Stemming

**Stemming** is the process of reducing words to their base or root form by removing prefixes or suffixes. Stemming is a heuristic approach, and various algorithms like the Porter stemmer, Snowball stemmer, and Lancaster stemmer are available.

**Example using the Porter Stemmer**:
- Original Word: "jumping"
- Stemmed Word: "jump"

- Original Word: "flies"
- Stemmed Word: "fli"

- Original Word: "unhappiness"
- Stemmed Word: "unhappi"

### Demo Code Porter Stemmer
```python
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')

"""
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
"""

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
```
```bash
OUTPUT:
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
```


Stemming is useful for tasks where word variants can be treated as the same word to reduce dimensionality and improve text analysis and retrieval. However, it can sometimes result in non-dictionary words.

### 2. Lemmatization

**Lemmatization** is the process of reducing words to their base or dictionary form, known as the lemma. It ensures that the resulting word is a valid word in the language, considering linguistic knowledge and rules.

**Example**:
- Original Word: "jumping"
- Lemmatized Word: "jump"

- Original Word: "flies"
- Lemmatized Word: "fly"

- Original Word: "unhappiness"
- Lemmatized Word: "unhappiness"

Lemmatization is more accurate than stemming because it considers the context of words and grammatical rules. It is beneficial when maintaining the semantic meaning of words is crucial.

In summary, stemming is faster and less accurate, while lemmatization is slower but more accurate. The choice depends on the specific requirements of a natural language processing task.
