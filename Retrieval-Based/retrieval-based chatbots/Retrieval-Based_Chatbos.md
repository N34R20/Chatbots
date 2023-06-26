# Retrieval-Based Chatbots

## **Introduction to Retrieval-based Chatbots**

Most modern technology users are well-aware that the voice assistant on their phone is an intelligent chatbot mimicking a human conversation. Less obvious are the thousands of other chatbots that facilitate shopping online, accessing your bank account, and a multitude of other customer-service interactions. In sheer number, **retrieval-based chatbots** are the most popular chatbot implementation in use today. This popularity is due in large part to the strength of retrieval-based bots in **closed-domain** conversations — conversations that are clearly limited in scope.

Most chatbot systems, including those that are retrieval-based, depend on three main tasks in order to convincingly complete a conversation:

- **Intent Classification**: When presented with user input, a system must classify the intent of the message. Is a message requesting information on the material of a pair of pants for sale? Or is the message related to the estimated shipping date of the clothing item?
- **Entity Recognition**: Entities are often the proper nouns of a message, like the day of the week when an item will ship, or the name of a specific item of clothing. A chatbot system must be able to recognize which entities a user message revolves around, and reference those entities in a response.
- **Response Selection**: Retrieval-based chatbot systems are unique in their reliance on a collection of predefined responses to user messages. Once a system has an understanding of both the intent and main entities of the user message, the program can retrieve the best-fit response from this collection.

The diagram in the workspace area is a representation of a standard chatbot system architecture, which includes four main steps:

- When a chatbot receives a user message, **intent classification** and **entity recognition** programs are immediately run in tandem.
- The results from both of these tasks are then fed into a candidate response generator. In a retrieval-based chatbot system, **response generation** relies on a database of predefined responses. Using the results from entity recognition and intent classification tasks, a set of possibly similar chatbot responses are retrieved from the database.
- After the selection of a set of candidate responses, a final **response selection** program ranks and selects the “best fit” response to be returned to the user.

![Untitled](Retrieval-Based%20Chatbots%208880f7bae2674c16b8adf1a1375a60ea/Untitled.png)

## **Intent with Bag-of-Words**

One way we can begin parsing a user’s message to a retrieval-based chatbot is to consider a user’s intent word by word. Bag-of-Words (BoW) is one of the most common models used to understand human language at the individual word level. You might remember that BoW results in a mapping of a word to its frequency in a text.

The `collections` module’s `Counter()` concisely builds this word-to-frequency mapping:

```python
Counter(preprocess("Wheels on the bus go round and round, round and round, round and round."))

print(word_counts)
#Counter({'round': 6, 'wheels': 1, 'bus': 1, 'go': 1})

```

We can use the results of this mapping to construct a measure of the intent of a user’s message. Then we’ll use this measure to retrieve the most similar answer from our collection of predefined chatbot responses.

However, there are a number of different ways in which we can define sentence similarity. There is no guarantee that one definition will be best, or even that any two definitions will suggest the same response!

A simple BoW model is best-fit for a situation where the order of words does not contain much information about the intent of a user’s message. In these situations, the unique collection of words in a message often reveals a lot of information about the user’s concern and provides a simple, yet powerful metric to quantify similarity.

## **Intent with TF-IDF**

While the number of shared words is an intuitive way to think about the similarity between two documents, a number of other methods are also commonly used to determine the best match between the user input and a pre-defined response.

One notable measure, **term frequency–inverse document frequency** (**tf-idf**), puts emphasis on the relative frequency in which a term occurs within a possible response and the user message. You can dive into the calculation in [our lesson on TF-IDF](https://www.codecademy.com/paths/build-chatbots-with-python/tracks/retrieval-based-chatbots/modules/retrieval-based-chatbots/lessons/retrieval-based-chatbots/exercises/link), but generally, tf-idf is best suited for domains where the most important terms in an input or response are mentioned repeatedly.

In Python, the `sklearn` package has a handy `TfidfVectorizer()` object that can be initialized as follows:

```python
vectorizer = TfidfVectorizer()

```

We can then fit the tf-idf model with the `.fit_transform()` method and input a list of string objects. Using the vectorized results of this fitted model, we can compute the cosine similarity of the user message and a possible response with the aptly named `cosine_similarity()` function:

```python
# fit model
tfidf_vectors = vectorizer.fit_transform(input_list)

# compute cosine similarity
cosine_sim = cosine_similarity(user_message_vector, response_vector)

```

Most retrieval-based chatbots use multiple measures in order to rank the similarity between a user’s input and a number of possible responses. Oftentimes, different measures produce different similarity rankings.

The selection of a similarity measure is one of the most important decisions chatbot architects have to make while building a system. We should always consider the relative strengths and weaknesses of different metrics.

Let’s use tf-idf similarity to determine the best match between `user_message` and a set of possible responses from the earlier exercise; how might this impact which pre-defined response we determine to have the most similar intent with the user message?

```python
#Example:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_text 

response_a = "Every dress style is cut from a polyester blend for a strechy fit."
response_b = "The 'Elosie' dress runs large. I suggest you take your regular size or smaller."
response_c = "The 'Elosie' dress comes in green, lavender, and orange."
user_message = "Hello! What is the fit of the 'Elosie' dress? My shoulders are broad, so I often size up for a comfortable fit. Do dress sizes run large or small?"

documents = [response_a,response_b,response_c,user_message]

# preprocess responses and user_message:
processed_docs = [preprocess_text(doc) for doc in documents]

# create tfidf vectorizer:
vectorizer = TfidfVectorizer()
# fit and transform vectorizer on processed docs:
tfidf_vectors = vectorizer.fit_transform(processed_docs)

# compute cosine similarity betweeen the user message tf-idf vector and the different response tf-idf vectors:
cosine_similarities = cosine_similarity(tfidf_vectors[-1], tfidf_vectors)

# get the index of the most similar response to the user message:
similar_response_index = cosine_similarities.argsort()[0][-2]

best_response = documents[similar_response_index]
print(best_response)
```

## **Entity Recognition with POS Tagging**

After determining the best method for the classification of a user’s intent, chatbot architects set upon the task of recognizing entities within a user’s message. Similar to other tasks in chatbot systems, there are a number of methods that can be used to locate and interpret the entities found in a user message — it is up to the system architect (you!) to critically evaluate methods and select those that are best-fit for a chatbot’s specific domain.

**Part of speech** (**POS**) tagging is commonly used to identify entities within a user message, as most entities are nouns (refresh your understanding of English parts of speech in our [Parsing with Regex lesson](https://www.codecademy.com/courses/language-parsing/lessons/nlp-regex-parsing-intro/exercises/introduction)). `nltk`’s `pos_tag()` function can rapidly tag a tokenized sentence and return a list of tuple objects for use in entity recognition tasks. A sentence, like “Jack and Jill went up the hill,” when tokenized and supplied in a call to `pos_tag()`, outputs a list of tuple objects:

```python
tagged_message = [('Jack', 'NNP'), ('and', 'CC'), ('Jill', 'NNP'), ('went', 'VBD'), ('up', 'RP'), ('the', 'DT'), ('hill', 'NN')]

```

We can extract words tagged as a noun, represented in `nltk`‘s [tagging schema](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) as a collection of tags beginning with “NN,” by checking if the string “NN” occurs in the token tag, and then appending the token string to a list of nouns if True:

```python
for token in tagged_message:
  if "NN" in token[1]:
     message_nouns.append(token[0])

```

Once we have a list of the entities in a user message, we’re well on our way to developing a realistic chatbot response!

## **Entity Recognition with Word Embeddings**

A conversation is a two-way street — both participants must actively listen to one another and adapt their responses based on the information given by the other. While POS tagging allows us to extract key entities in a user message, it does not provide context that allows a chatbot to believably integrate an entity reference into a predefined response.

A retrieval-based chatbot’s collection of predefined responses is very similar to an empty

[Madlibs story](https://en.wikipedia.org/wiki/Mad_Libs)

. Each response is a complete sentence, but with many key words replaced with blank spots. Unlike Madlibs, these blanks will be labeled with a reference to a broad kind of entity. For instance, a predefined response for a weather report chatbot might look like:

![https://content.codecademy.com/programs/chatbots/retrieval-based-chatbots/retrieval_based_chatbot_madlib.png](https://content.codecademy.com/programs/chatbots/retrieval-based-chatbots/retrieval_based_chatbot_madlib.png)

In order to produce a coherent response, the chatbot must insert entities from a user message into these blank spots. Chatbot architects often use word embedding models, like word2vec, to rank the similarity of user-provided entities and the broad category associated with a response “blank spot” (refresh your understanding of word2vec in our [word embeddings](https://www.codecademy.com/paths/build-chatbots-with-python/tracks/retrieval-based-chatbots/modules/retrieval-based-chatbots/lessons/retrieval-based-chatbots/exercises/link) lesson). The `spacy` package provides a pre-trained word2vec model which can be called on a string of entities and the responses category like this:

```python
# load word2vec model
word2vec = spacy.load('en')

# call model on data
tokens = word2vec("wednesday, dog, boots, flower")
response_category = word2vec("weekday")

```

After the model has been applied, we can use spacy’s `.similarity()` method to compute the cosine similarity between user-provided entities and a response category:

```python
output_list = list()
for token in tokens:
    output_list.append(f"{token.text} {response_category.text} {token.similarity(response_category)}")

# output:
# wednesday weekday 0.453354920245737
# dog weekday 0.21911001129423147
# boots weekday 0.15690609198936833
# flower weekday 0.17118961389940174

```

The resulting output above shows that the word2vec model found “wednesday” to have the greatest similarity to “weekday.” This should match what you would expect! A chatbot system can select the token with the highest similarity score and insert it into the “blank spot” in a predefined response in order to continue a coherent conversation with a user.

```python
#Example:
import spacy
word2vec = spacy.load('en')

message_nouns = ['shirts', 'weekend', 'package']

tokens = word2vec(" ".join(message_nouns))
category = word2vec("clothes")

def compute_similarity(tokens, category):
  output_list = list()
  #your code here
  for token in tokens:
    output_list.append(f"{token.text} {category.text} {token.similarity(category)}")
  return output_list

print(compute_similarity(tokens, category))

blank_spot = message_nouns[0]
bot_response = f"Hey! I just checked my records, your shipment containing {blank_spot} is en route. Expect it within the next two days!"
print(bot_response)
```

## **Response Selection**

Great work! We’ve covered all the steps necessary for the construction of a chatbot response selection system — now let’s put our skills to work. Using BoW for intent selection, and word2vec for entity recognition, let’s automate the selection of the best-fit response from a collection of candidate responses.

All of the functions defined across previous exercises have been imported into the workspace. In addition, there are four provided data sources:

- The `user_message`, a question from a user engaging with a weather chatbot.
- Three possible `response` objects, with an entity `blank_spot` representing an “Illinois” category.

We’ve already preprocessed the `user_message` and response objects, and saved the results as BoW models. The results of calling our `compare_overlap()` function on each candidate response have been saved in `similarity_list`.

Similarly, a word2vec model has already been fit on our data, and the results of the model have been saved in `word2vec_result`. It’s up to you to combine the results from both models to retrieve a response object.

```python
from user_functions import preprocess, compare_overlap, pos_tag, extract_nouns, word2vec, compute_similarity
from collections import Counter

user_message = "Good morning... will it rain in Chicago later this week?"

blank_spot = "illinois city"
response_a = "The average temperature this weekend in {} with be 88 degrees. Bring your sunglasses!"
response_b = "Forget about your umbrella; there is no rain forecasted in {} this weekend."
response_c = "This weekend, a warm front from the southeast will keep skies near {} clear."
responses= [response_a, response_b, response_c]

#preprocess documents
bow_user_message = Counter(preprocess(user_message))
processed_responses = [Counter(preprocess(response)) for response in responses]

#determine intent with BoW
similarity_list = [compare_overlap(doc, bow_user_message) for doc in processed_responses]

response_index = similarity_list.index(max(similarity_list))

#extract entities with word2vec 
tagged_user_message = pos_tag(preprocess(user_message))
message_nouns = extract_nouns(tagged_user_message)

#execute word2vec below
tokens = word2vec(" ".join(message_nouns))
category = word2vec(blank_spot)
word2vec_result = compute_similarity(tokens, category)

print(word2vec_result)
entity = word2vec_result[2][0]
print(entity)
#select final response below

final_response = responses[response_index].format(entity)
print(final_response)
```

## **Building a Retrieval System**

Now that we have covered the three underlying tasks of a chatbot system — intent classification, entity extraction, and response selection — it’s time for us to pull all these tasks together into a full conversational system! We’ll do so by integrating the functions we have written throughout this lesson into three methods and organize them into a `class`.

This organization will allow us to call our program from the bash terminal and pass in our own questions as the `user_message`. We have already included code that initializes and starts a conversation with our bot. It’s up to you to construct the `.find_intent_match()`, `.find_entities()`, and `.respond()` methods so that our bot can really speak!

```python
from user_functions import preprocess, compare_overlap, pos_tag, extract_nouns, compute_similarity
from data import responses, blank_spot
from collections import Counter
import spacy
word2vec = spacy.load('en')

class ChatBot:
  def find_intent_match(self, responses, user_message):
    bow_user_message = Counter(preprocess(user_message))
    processed_responses = [Counter(preprocess(response)) for response in responses]
    # define similarity_list here:
    similarity_list = [compare_overlap(response, bow_user_message) for response in processed_responses]
    # define response_index here:
    response_index = similarity_list.index(max(similarity_list))
    return responses[response_index]

    
  def find_entities(self, user_message):
    tagged_user_message = pos_tag(preprocess(user_message))
    message_nouns = extract_nouns(tagged_user_message)
    
    # execute word2vec model here:
    tokens = word2vec(" ".join(message_nouns))
    category = word2vec(blank_spot)
    word2vec_result = compute_similarity(tokens, category)
    word2vec_result.sort(key=lambda x: x[2])
    return word2vec_result[-1][0]

  # define .respond() here:
  def respond(self, user_message):
     best_response = self.find_intent_match(responses, user_message)
     entity = self.find_entities(user_message)
     print(best_response.format(entity))    
    
  def chat(self):
    user_message = input("Hi, I'm Stratus. Ask me about your local weather!\n")
    self.respond(user_message)

# create ChatBot() instance:
bot = ChatBot()
# call .chat() method:
bot.chat()
```

## **Lesson Review**

Congratulations! You’ve learned to build your own retrieval-based chatbot, and have the tools necessary to develop a bot that can be used in any closed-domain. Let’s review what we’ve covered in this lesson:

- Retrieval-based chatbots are used in **closed-domain scenarios** and rely on a collection of predefined responses to a user message. A retrieval-based bot completes three main tasks: **intent classification**, **entity recognition**, and **response selection**.
- There are a number of ways to determine which response is the best fit for a given user message. One of the most important decisions a chatbot architect makes is the selection of a similarity metric.
- **Bag-of-Words** (**BoW**) models are commonly used to compute intent similarity measures based on word overlap.
- **Term frequency–inverse document frequency** (**tf-idf**) is another common similarity metric which incorporates the relative frequency of terms across the collection of possible responses. The `sklearn` package provides a `TfidfVectorizer()` object that we can use to fit tf-idf models.
- Entity recognition tasks often extract proper nouns from a user message using **Part of Speech** (**POS**) tagging. POS tagging can be performed with `nltk`’s `.pos_tag()` method.
- It’s often helpful to imagine pre-defined chatbot responses as a kind of MadLibs story. We can use word embeddings models, like the one implemented in the `spacy` package, to insert entities into response objects based on their cosine similarity with abstract, “blank-spot” concepts.
- The final response selection relies on results from both intent classification and entity recognition tasks in order to produce a coherent response to the user message.