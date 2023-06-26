# Bag-of-Words Language Model

## **Intro to Bag-of-Words**

“A bag-of-words is all you need,” some NLPers have decreed.

The bag-of-words language model is a simple-yet-powerful tool to have up your sleeve when working on natural language processing (NLP). The model has many, many use cases including:

- determining topics in a song
- filtering spam from your inbox
- finding out if a tweet has positive or negative sentiment
- creating word clouds

Spam clasifier example:

```python
from spam_data import training_spam_docs, training_doc_tokens, training_labels
from sklearn.naive_bayes import MultinomialNB
from preprocessing import preprocess_text

# Add your email text to test_text between the triple quotes:
test_text = """
Our records indicate your Pension is under-performing to see higher growth and up to 25% cash release reply for a free review.
"""
test_tokens = preprocess_text(test_text)

def create_features_dictionary(document_tokens):
  features_dictionary = {}
  index = 0
  for token in document_tokens:
    if token not in features_dictionary:
      features_dictionary[token] = index
      index += 1
  return features_dictionary

def tokens_to_bow_vector(document_tokens, features_dictionary):
  bow_vector = [0] * len(features_dictionary)
  for token in document_tokens:
    if token in features_dictionary:
      feature_index = features_dictionary[token]
      bow_vector[feature_index] += 1
  return bow_vector

bow_sms_dictionary = create_features_dictionary(training_doc_tokens)
training_vectors = [tokens_to_bow_vector(training_doc, bow_sms_dictionary) for training_doc in training_spam_docs]
test_vectors = [tokens_to_bow_vector(test_tokens, bow_sms_dictionary)]

spam_classifier = MultinomialNB()
spam_classifier.fit(training_vectors, training_labels)

predictions = spam_classifier.predict(test_vectors)

print("Looks like a normal email!" if predictions[0] == 0 else "You've got spam!")
```

## **Bag-of-What?**

***Bag-of-words (BoW)*** is a statistical language model based on word count. Say what?

Let’s start with that first part: a ***statistical language model*** is a way for computers to make sense of language based on probability. For example, let’s say we have the text:

“Five fantastic fish flew off to find faraway functions. Maybe find another five fantastic fish?”

A statistical language model focused on the starting letter for words might take this text and predict that words are most likely to start with the letter “f” because 11 out of 15 words begin that way. A different statistical model that pays attention to word order might tell us that the word “fish” tends to follow the word “fantastic.”

Bag-of-words does not give a flying fish about word starts or word order though; its sole concern is ***word count*** — how many times each word appears in a document.

If you’re already familiar with statistical language models, you may also have heard BoW referred to as the ***unigram model***. It’s technically a special case of another statistical model, the *n*-gram model, with *n* (the number of words in a sequence) set to `1`.

If you have no idea what *n*-grams are, don’t worry — we’ll dive deeper into them in another lesson.

## **BoW Dictionaries**

One of the most common ways to implement the BoW model in Python is as a dictionary with each key set to a word and each value set to the number of times that word appears. Take the example below:

![https://content.codecademy.com/courses/NLP/bag-of-words.gif](https://content.codecademy.com/courses/NLP/bag-of-words.gif)

The words from the sentence go into the bag-of-words and come out as a dictionary of words with their corresponding counts. For statistical models, we call the text that we use to build the model our ***training data***. Usually, we need to prepare our text data by breaking it up into `documents` (shorter strings of text, generally sentences).

Example:

```python
from preprocessing import preprocess_text
# Define text_to_bow() below:
def text_to_bow(some_text):
  bow_dictionary = {}
  tokens = preprocess_text(some_text)
  for token in tokens:
    if token in bow_dictionary:
      bow_dictionary[token] += 1
    else:
      bow_dictionary[token] = 1  
  return bow_dictionary

print(text_to_bow("I love fantastic flying fish. These flying fish are just ok, so maybe I will find another few fantastic fish..."))
```

## **Introducing BoW Vectors**

Sometimes a dictionary just won’t fit the bill. Topic modelling applications, for example, require an implementation of bag-of-words that is a bit more mathematical: ***feature vectors***.

A feature vector is a numeric representation of an item’s important features. Each feature has its own column. If the feature exists for the item, you could represent that with a `1`. If the feature does not exist for that item, you could represent that with a `0`. A few monsters could be represented as vectors like so:

[Untitled Database](https://www.notion.so/6d27e090bfed4115847bb73b12020d1c?pvs=21)

For bag-of-words, instead of monsters you would have documents and the features would be different words. And we don’t just care if a word is present in a document; we want to know how many times it occurred! Turning text into a BoW vector is known as ***feature extraction*** or ***vectorization***.

But how do we know which vector index corresponds to which word? When building BoW vectors, we generally create a ***features dictionary*** of all vocabulary in our training data (usually several documents) mapped to indices.

For example, with “Five fantastic fish flew off to find faraway functions. Maybe find another five fantastic fish?” our dictionary might be:

```python
{'five': 0,
'fantastic': 1,
'fish': 2,
'fly': 3,
'off': 4,
'to': 5,
'find': 6,
'faraway': 7,
'function': 8,
'maybe': 9,
'another': 10}

```

Using this dictionary, we can convert new documents into vectors using a vectorization function. For example, we can take a brand new sentence “Another five fish find another faraway fish.” — ***test data*** — and convert it to a vector that looks like:

```python
[1, 0, 2, 0, 0, 0, 1, 1, 0, 0, 2]

```

The word ‘another’ appeared twice in the test data. If we look at the feature dictionary for ‘another’, we find that its index is `10`. So when we go back and look at our vector, we’d expect the number at index `10` to be `2`.

## **Building a Features Dictionary**

Now that you know what a bag-of-words vector looks like, you can create a function that builds them!

First, we need a way of generating a features dictionary from a list of training documents. We can build a Python function to do that for us…

```python
from preprocessing import preprocess_text
# Define create_features_dictionary() below:
def create_features_dictionary(documents):
  features_dictionary = {}
  merged = " ".join(documents)
  tokens = preprocess_text(merged)
  index = 0
  for token in tokens:
    if token not in features_dictionary:
      features_dictionary[token] = index
      index += 1
  return features_dictionary, tokens
```

## **Building a BoW Vector**

Nice work! Time to put that dictionary of vocabulary to good use and build a bag-of-words vector from a new document.

In Python, we can use a list to represent a vector. Each index in the list will correspond to a word and be set to its count.

![https://content.codecademy.com/courses/NLP/Building_vector.gif](https://content.codecademy.com/courses/NLP/Building_vector.gif)

```python
from preprocessing import preprocess_text
# Define text_to_bow_vector() below:
def text_to_bow_vector(some_text, features_dictionary):
  bow_vector = [0] * len(features_dictionary)
  tokens = preprocess_text(some_text)
  for token in tokens:
    feature_index = features_dictionary[token]
    bow_vector[feature_index] += 1
  return bow_vector, tokens
```

## **It's All in the Bag**

Phew! That was a lot of work.

It’s time to put `create_features_dictionary()` and `tokens_to_bow_vector()` together and use them in a spam filter we created that uses a Naive Bayes classifier. We’ve slightly modified the two functions for this use case, but they should still look familiar.

Let’s see `create_features_dictionary()` and `tokens_to_bow_vector()` in action with real test data, helping fend off spam!

```python
from spam_data import training_spam_docs, training_doc_tokens, training_labels, test_labels, test_spam_docs, training_docs, test_docs
from sklearn.naive_bayes import MultinomialNB

def create_features_dictionary(document_tokens):
  features_dictionary = {}
  index = 0
  for token in document_tokens:
    if token not in features_dictionary:
      features_dictionary[token] = index
      index += 1
  return features_dictionary

def tokens_to_bow_vector(document_tokens, features_dictionary):
  bow_vector = [0] * len(features_dictionary)
  for token in document_tokens:
    if token in features_dictionary:
      feature_index = features_dictionary[token]
      bow_vector[feature_index] += 1
  return bow_vector

# Define bow_sms_dictionary:
bow_sms_dictionary = create_features_dictionary(training_doc_tokens) 

# Define training_vectors:
training_vectors = [tokens_to_bow_vector(training_doc, bow_sms_dictionary) for training_doc in training_spam_docs]
# Define test_vectors:

test_vectors = [tokens_to_bow_vector(test_doc, bow_sms_dictionary) for test_doc in test_spam_docs]

spam_classifier = MultinomialNB()

def spam_or_not(label):
  return "spam" if label else "not spam"

# Uncomment the code below when you're done:
spam_classifier.fit(training_vectors, training_labels)

predictions = spam_classifier.score(test_vectors, test_labels)

print("The predictions for the test data were {0}% accurate.\n\nFor example, '{1}' was classified as {2}.\n\nMeanwhile, '{3}' was classified as {4}.".format(predictions * 100, test_docs[0], spam_or_not(test_labels[0]), test_docs[10], spam_or_not(test_labels[10])))
```

## **Spam A Lot No More**

Amazing work! As is the case with many tasks in Python, there’s already a library that can do all of that work for you.

For `text_to_bow()`, you can approximate the functionality with the `collections` module’s `Counter()` function:

```python
from collections import Counter

tokens = ['another', 'five', 'fish', 'find', 'another', 'faraway', 'fish']
print(Counter(tokens))

# Counter({'fish': 2, 'another': 2, 'find': 1, 'five': 1, 'faraway': 1})

```

For vectorization, you can use `CountVectorizer` from the machine learning library `scikit-learn`. You can use `fit()` to train the features dictionary and then `transform()` to transform text into a vector:

```python
from sklearn.feature_extraction.text import CountVectorizer

training_documents = ["Five fantastic fish flew off to find faraway functions.", "Maybe find another five fantastic fish?", "Find my fish with a function please!"]
test_text = ["Another five fish find another faraway fish."]
bow_vectorizer = CountVectorizer()
bow_vectorizer.fit(training_documents)
bow_vector = bow_vectorizer.transform(test_text)
print(bow_vector.toarray())
# [[2 0 1 1 2 1 0 0 0 0 0 0 0 0 0]]

```

Result:

```python
from spam_data import training_spam_docs, training_doc_tokens, training_labels, test_labels, test_spam_docs, training_docs, test_docs
from sklearn.naive_bayes import MultinomialNB
# Import CountVectorizer from sklearn:
from sklearn.feature_extraction.text import CountVectorizer

# Define bow_vectorizer:
bow_vectorizer = CountVectorizer()

# Define training_vectors:
training_vectors = bow_vectorizer.fit_transform(training_docs)
# Define test_vectors:
test_vectors = bow_vectorizer.transform(test_docs)

spam_classifier = MultinomialNB()

def spam_or_not(label):
  return "spam" if label else "not spam"

# Uncomment the code below when you're done:
spam_classifier.fit(training_vectors, training_labels)

predictions = spam_classifier.score(test_vectors, test_labels)

print("The predictions for the test data were {0}% accurate.\n\nFor example, '{1}' was classified as {2}.\n\nMeanwhile, '{3}' was classified as {4}.".format(predictions * 100, test_docs[7], spam_or_not(test_labels[7]), test_docs[15], spam_or_not(test_labels[15])))
```

## **BoW Wow**

As you can see, bag-of-words is pretty useful! BoW also has several advantages over other language models. For one, it’s an easier model to get started with and a few Python libraries already have built-in support for it.

Because bag-of-words relies on single words, rather than sequences of words, there are more examples of each unit of language in the training corpus. More examples means the model has less ***data sparsity*** (i.e., it has more training knowledge to draw from) than other statistical models.

Imagine you want to make a shirt to sell to people. If you have the shirt exactly tailored to someone’s body, it probably won’t fit that many people. But if you make a shirt that is just a giant bag with arm holes, you know that no one will buy it. What do you do? You loosely fit the shirt to someone’s body, leaving some extra room for different body shapes.

***Overfitting*** (adapting a model too strongly to training data, akin to our highly tailored shirt) is a common problem for statistical language models. While BoW still suffers from overfitting in terms of vocabulary, it overfits less than other statistical models, allowing for more flexibility in grammar and word choice.

The combination of low data sparsity and less overfitting makes the bag-of-words model more reliable with smaller training data sets than other statistical models.

## **BoW Ow**

Alas, there is a trade-off for all the brilliance BoW brings to the table.

Unless you want sentences that look like “the a but for the”, BoW is NOT a great primary model for text prediction. If that sort of “sentence” isn’t your bag, it’s because bag-of-words has high ***perplexity***, meaning that it’s not a very accurate model for language prediction. The probability of the following word is always just the most frequently used words.

If your BoW model finds “good” frequently occurring in a text sample, you might assume there’s a positive sentiment being communicated in that text… but if you look at the original text you may find that in fact every “good” was preceded by a “not.”

Hmm, that would have been helpful to know. The BoW model’s word tokens lack context, which can make a word’s intended meaning unclear.

Perhaps you are wondering, “What happens if the model comes across a new word that wasn’t in the training data?” As mentioned, like all statistical models, BoW suffers from overfitting when it comes to vocabulary.

There are several ways that NLP developers have tackled this issue. A common approach is through ***language smoothing*** in which some probability is siphoned from the known words and given to unknown words.

## **Review of Bag-of-Words**

You made it! And you’ve learned plenty about the bag-of-words language model along the way:

- Bag-of-words (BoW) — also referred to as the unigram model — is a statistical language model based on word count.
- There are loads of real-world applications for BoW.
- BoW can be implemented as a Python dictionary with each key set to a word and each value set to the number of times that word appears in a text.
- For BoW, training data is the text that is used to build a BoW model.
- BoW test data is the new text that is converted to a BoW vector using a trained features dictionary.
- A feature vector is a numeric depiction of an item’s salient features.
- Feature extraction (or vectorization) is the process of turning text into a BoW vector.
- A features dictionary is a mapping of each unique word in the training data to a unique index. This is used to build out BoW vectors.
- BoW has less data sparsity than other statistical models. It also suffers less from overfitting.
- BoW has higher perplexity than other models, making it less ideal for language prediction.
- One solution to overfitting is language smoothing, in which a bit of probability is taken from known words and allotted to unknown words.