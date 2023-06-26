# Word Embeddings

## **Introduction**

There is a famous saying by German writer Johann Wolfgang von Goethe:

```
"Tell me with whom you consort and I will tell you who you are..."

```

Goethe’s assumption is that the people you spend time with each and every day are a reflection of who you are as a person. Would you agree?

Now, what if we extend that same idea from people to words? Linguist John Rupert Firth took this step and came up with his own saying:

```
"You shall know a word by the company it keeps."

```

This idea that a word’s meaning can be understood by its context, or the words that surround it, is the basis for word embeddings. A *word embedding* is a representation of a word as a numeric vector, enabling us to compare and contrast how words are used and identify words that occur in similar contexts.

The applications of word embeddings include:

- entity recognition in chatbots
- sentiment analysis
- syntax parsing

Have little to no experience with vectors? Do not fear! We’ll break down word embeddings and how they relate to vectors, step-by-step, in the next few exercises with the help of the [spaCy package](https://spacy.io/). Let’s get started!

## **Vectors**

Vectors can be many things in many different fields, but ultimately they are containers of information. Depending on the size, or the dimension, of a vector, it can hold varying amounts of data.

The simplest case is a 1-dimensional vector, which stores a single number. Say we want to represent the length of a word with a vector. We can do so as follows:

```
"cat" ----> [3]
"scrabble" ----> [8]
"antidisestablishmentarianism" ----> [28]

```

Instead of looking at these three words with our own eyes, we can compare the vectors that represent them by plotting the vectors on a number line.

![https://content.codecademy.com/programs/chatbots/word-embeddings/vectors-one-dimension.png](https://content.codecademy.com/programs/chatbots/word-embeddings/vectors-one-dimension.png)

We can clearly see that the “cat” vector is much smaller than the “scrabble” vector, which is much smaller than the “antidisestablishmentarianism” vector.

Now let’s say we also want to record the number of vowels in our words, in addition to the number of letters. We can do so using a 2-dimensional vector, where the first entry is the length of the word, and the second entry is the number of vowels:

```
"cat" ----> [3, 1]
"scrabble" ----> [8, 2]
"antidisestablishmentarianism" ----> [28, 11]

```

To help visualize these vectors, we can plot them on a two-dimensional grid, where the x-axis is the number of letters, and the y-axis is the number of vowels.

![https://content.codecademy.com/programs/chatbots/word-embeddings/vectors-two-dimensions.png](https://content.codecademy.com/programs/chatbots/word-embeddings/vectors-two-dimensions.png)

Here we can see that the vectors for “cat” and “scrabble” point to a more similar area of the grid than the vector for “antidisestablishmentarianism”. So we could argue that “cat” and “scrabble” are closer together.

While we have shown here only 1-dimensional and 2-dimensional vectors, we are able to have vectors in any number of dimensions. Even 1,000! The tricky part, however, is visualizing them.

Vectors are useful since they help us summarize information about an object using numbers. Then, using the number representation, we can make comparisons between the vector representations of different objects!

This idea is central to how word embeddings map words into vectors.

We can easily represent vectors in Python using NumPy arrays. To create a vector containing the odd numbers from `1` to `9`, we can use NumPy’s `.array()` method:

```python
odd_vector = np.array([1, 3, 5, 7, 9])

```

## **What is a Word Embedding?**

Now that you have an understanding of vectors, let’s jump back to word embeddings. Word embeddings are vector representations of a word.

They allow us to take all the information that is stored in a word, like its meaning and its part of speech, and convert it into a numeric form that is more understandable to a computer.

For example, we could look at a word embedding for the word “peace”.

```
[5.2907305, -4.20267, 1.6989858, -1.422668, -1.500128, ...]

```

Here “peace” is represented by a 96-dimension vector, with just the first five dimensions shown. Each dimension of the vector is capturing some information about how the word “peace” is used. We can also look at a word embedding for the word “war”:

```
[7.2966490, -0.52887750, 0.97479630, -2.9508233, -3.3934135, ...]

```

By converting the words “war” and “peace” into their numeric vector representations, we are able to have a computer more easily compare the vectors and understand their similarities and differences.

We can load a basic English word embedding model using spaCy as follows:

```python
nlp = spacy.load('en')

```

Note: the convention is to load spaCy models into a variable named `nlp`.

To get the vector representation of a word, we call the model with the desired word as an argument and can use the `.vector` attribute.

```python
nlp('love').vector

```

But how do we compare these vectors? And how do we arrive at these numeric representations?

## **Distance**

The key at the heart of word embeddings is distance. Before we explain why, let’s dive into how the distance between vectors can be measured.

There are a variety of ways to find the distance between vectors, and here we will cover three. The first is called Manhattan distance.

In Manhattan distance, also known as city block distance, distance is defined as the sum of the differences across each individual dimension of the vectors. Consider the vectors `[1,2,3]` and `[2,4,6]`. We can calculate the Manhattan distance between them as shown below:

$$
manhattan\ distance\ =\ \left | 1-2 \right |+\left | 2-4 \right| +\left | 3-6 \right|=1+2+3=6
$$

Another common distance metric is called the Euclidean distance, also known as straight line distance. With this distance metric, we take the square root of the sum of the squares of the differences in each dimension.

$$
euclidean\ distance\ =\sqrt{(1-2)^{2})+(2-4)^{2})+(3-6)^{2})}=\sqrt{14}\approx 3.74

$$

The final distance we will consider is the cosine distance. Cosine distance is concerned with the angle between two vectors, rather than by looking at the distance between the points, or ends, of the vectors. Two vectors that point in the same direction have no angle between them, and have a cosine distance of `0`. Two vectors that point in opposite directions, on the other hand, have a cosine distance of `1`. We would show you the calculation, but we don’t want to scare you away! For the mathematically adventurous, [you can read up on the calculation here](https://en.wikipedia.org/wiki/Cosine_similarity#Definition).

We can easily calculate the Manhattan, Euclidean, and cosine distances between vectors using helper functions from [SciPy](https://docs.scipy.org/doc/scipy-0.14.0/reference/index.html):

```python
from scipy.spatial.distance import cityblock, euclidean, cosine

vector_a = np.array([1,2,3])
vector_b = np.array([2,4,6])

# Manhattan distance:
manhattan_d = cityblock(vector_a,vector_b) # 6

# Euclidean distance:
euclidean_d = euclidean(vector_a,vector_b) # 3.74

# Cosine distance:
cosine_d = cosine(vector_a,vector_b) # 0.0

```

When working with vectors that have a large number of dimensions, such as word embeddings, the distances calculated by Manhattan and Euclidean distance can become rather large. Thus, calculations using cosine distance are preferred!

## **Word Embeddings Are All About Distance**

The idea behind word embeddings is a theory known as the distributional hypothesis. This hypothesis states that words that co-occur in the same contexts tend to have similar meanings. With word embeddings, we map words that exist with the same context to similar places in our vector space (math-speak for the area in which our vectors exist).

The numeric values that are assigned to the vector representation of a word are not important in their own right, but gather meaning from how similar or not words are to each other.

Thus the cosine distance between words with similar contexts will be small, and the cosine distance between words that have very different contexts will be large.

The literal values of a word’s embedding have no actual meaning. We gain value in word embeddings from comparing the different word vectors and seeing how similar or different they are. Encoded in these vectors, however, is latent information about how they are used.

```python
import spacy
from scipy.spatial.distance import cosine
from processing import most_common_words, vector_list

# print word and vector representation at index 347
print(most_common_words[347], vector_list[347])

# define find_closest_words
def find_closest_words(word_list, vector_list, word_to_check):
    return sorted(word_list,
                  key=lambda x: cosine(vector_list[word_list.index(word_to_check)], vector_list[word_list.index(x)]))[:10]

# find closest words to food
close_to_food = find_closest_words(most_common_words, vector_list, "food")
print(close_to_food)
```

## **Word2vec**

You might be asking yourself a question now. How did we arrive at the vector values that define a word vector? And how do we ensure that the values chosen place the vectors for words with similar context close together and the vectors for words with different usages far apart?

Step in word2vec! *Word2vec* is a statistical learning algorithm that develops word embeddings from a corpus of text. Word2vec uses one of two different model architectures to come up with the values that define a collection of word embeddings.

One method is to use the continuous bag-of-words (CBOW) representation of a piece of text. The word2vec model goes through each word in the training corpus, in order, and tries to predict what word comes at each position based on applying bag-of-words to the words that surround the word in question. In this approach, the order of the words does not matter!

The other method word2vec can use to create word embeddings is continuous skip-grams. Skip-grams function similarly to n-grams, except instead of looking at groupings of n-consecutive words in a text, we can look at sequences of words that are separated by some specified distance between them.

For example, consider the sentence `"The squids jump out of the suitcase"`. The 1-skip-2-grams includes all the bigrams (2-grams) as well as the following subsequences:

```
(The, jump), (squids, out), (jump, of), (out, the), (of, suitcase)

```

When using continuous skip-grams, the order of context **is** taken into consideration! Because of this, the time it takes to train the word embeddings is slower than when using continuous bag-of-words. The results, however, are often much better!

With either the continuous bag-of-words or continuous skip-grams representations as training data, word2vec then uses a shallow, 2-layer neural network to come up with the values that place words with a similar context in vectors near each other and words with different contexts in vectors far apart from each other.

Let’s take a closer look to see how continuous bag-of-words and continuous skip-grams work!

## **Gensim**

Depending on the corpus of text we select to train a word embedding model, different word embeddings will be created according to the context of the words in the given corpus. The larger and more generic a corpus, however, the more generalizable the word embeddings become.

When we want to train our own word2vec model on a corpus of text, we can use the [gensim package](https://radimrehurek.com/gensim/)!

In previous exercises, we have been using pre-trained word embedding models stored in spaCy. These models were trained, using word2vec, on blog posts and news articles collected by the [Linguistic Data Consortium](https://catalog.ldc.upenn.edu/LDC2013T19) at the University of Pennsylvania. With gensim, however, we are able to build our *own* word embeddings on any corpus of text we like.

To easily train a word2vec model on our own corpus of text, we can use gensim’s `Word2Vec()` function.

```python
model = gensim.models.Word2Vec(corpus, size=100, window=5, min_count=1, workers=2, sg=1)

```

- `corpus` is a list of lists, where each inner list is a document in the corpus and each element in the inner lists is a word token
- `size` determines how many dimensions our word embeddings will include. Word embeddings often have upwards of 1,000 dimensions! Here we will create vectors of 100-dimensions to keep things simple.
- don’t worry about the rest of the keyword arguments here!

To view the entire vocabulary used to train the word embedding model, we can use the `.wv.vocab.items()` method.

```python
vocabulary_of_model = list(model.wv.vocab.items())

```

When we train a word2vec model on a smaller corpus of text, we pick up on the unique ways in which words of the text are used.

For example, if we were using scripts from the television show Friends as a training corpus, the model would pick up on the unique ways in which words are used in the show. While the generalized vectors in a spaCy model might not place the vectors for “Ross” and “Rachel” close together, a gensim word embedding model trained on Friends’ scripts would place the vectors for words like “Ross” and “Rachel”, two characters that have a continuous on and off-again relationship throughout the show, very close together!

To easily find which vectors gensim placed close together in its word embedding model, we can use the `.most_similar()` method.

```python
model.most_similar("my_word_here", topn=100)

```

- `"my_word_here"` is the target word token we want to find most similar words to
- `topn` is a keyword argument that indicates how many similar word vectors we want returned

One last gensim method we will explore is a rather fun one: `.doesnt_match()`.

```python
model.doesnt_match(["asia", "mars", "pluto"])

```

- when given a list of terms in the vocabulary as an argument, `.doesnt_match()` returns which term is furthest from the others.

Let’s play around with gensim word2vec models to explore the word embeddings defined on our own corpus of training data!

## **Review**

Lost in a multidimensional vector space after this lesson? We hope not! We have covered a lot here, so let’s take some time to recap.

- *Vectors* are containers of information, and they can have anywhere from 1-dimension to hundreds or thousands of dimensions
- *Word embeddings* are vector representations of a word, where words with similar contexts are represented with vectors that are closer together
- *spaCy* is a package that enables us to view and use pre-trained word embedding models
- The distance between vectors can be calculated in many ways, and the best way for measuring the distance between higher dimensional vectors is *cosine distance*
- *Word2Vec* is a shallow neural network model that can build word embeddings using either continuous bag-of-words or continuous skip-grams
- *Gensim* is a package that allows us to create and train word embedding models using any corpus of text