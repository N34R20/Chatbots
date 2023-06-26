# Text Preprocessing

## **Introduction**

Text preprocessing is an approach for cleaning and preparing text data for use in a specific context. Developers use it in almost all natural language processing (NLP) pipelines, including voice recognition software, search engine lookup, and machine learning model training. It is an essential step because text data can vary. From its format (website, text message, voice recognition) to the people who create the text (language, dialect), there are plenty of things that can introduce noise into your data.

The ultimate goal of cleaning and preparing text data is to reduce the text to only the words that you need for your NLP goals.

In this lesson, you will learn strategies for preparing text data. While this list is not exhaustive, we will cover a few common approaches for cleaning and processing text data. They include:

- Using Regex & NLTK libraries
- Noise Removal – Removing unnecessary characters and formatting
- Tokenization – break multi-word strings into smaller components
- Normalization – a catch-all term for processing data; this includes stemming and lemmatization

## **Noise Removal**

Text cleaning is a technique that developers use in a variety of domains. Depending on the goal of your project and where you get your data from, you may want to remove unwanted information, such as:

- Punctuation and accents
- Special characters
- Numeric digits
- Leading, ending, and vertical whitespace
- HTML formatting

The type of noise that you need to remove from text usually depends on its source. For example, you could access data via the Twitter API, scraping a webpage, or voice recognition software. Fortunately, you can use the `.sub()` method in Python’s regular expression (`re`) library for most of your noise removal needs.

The `.sub()` method has three required arguments:

1. `pattern` – a regular expression that is searched for in the input string. There must be an `r` preceding the string to indicate it is a raw string, which treats backslashes as literal characters.
2. `replacement_text` – text that replaces all matches in the input string
3. `input` – the input string that will be edited by the `.sub()` method

The method returns a string with all instances of the `pattern` replaced by the `replacement_text`. Let’s see a few examples of using this method to remove and replace text from a string.

### **Examples**

First, let’s consider how to remove HTML `<p>` tags from a string:

```python
import re

text = "<p>    This is a paragraph</p>"

result = re.sub(r'<.?p>', '', text)

print(result)
#    This is a paragraph

```

Notice, we replace the tags with an empty string `''`. This is a common approach for removing text.

---

Next, let’s remove the whitespace from the beginning of the text. The whitespace consists of four spaces.

```python
import re

text = "    This is a paragraph"

result = re.sub(r'\s{4}', '', text)

print(result)
# This is a paragraph

```

Take a look at Codecademy’s [Parsing with Regular Expressions](https://www.codecademy.com/paths/natural-language-processing/tracks/nlp-language-parsing/modules/nlp-language-parsing/lessons/nlp-regex-parsing-intro/exercises/introduction) lesson if you want to learn more regular expression syntax and tricks.

```python
import re

headline_one = '<h1>Nation\'s Top Pseudoscientists Harness High-Energy Quartz Crystal Capable Of Reversing Effects Of Being Gemini</h1>'

tweet = '@fat_meats, veggies are better than you think.'

headline_no_tag = re.sub(r'<.?h.>', '', headline_one)

tweet_no_at = re.sub(r'@', '', tweet)

try:
  print(headline_no_tag)
except:
  print('No variable called `headline_no_tag`')
try:
  print(tweet_no_at)
except:
  print('No variable called `tweet_no_at`')
```

## **Tokenization**

For many natural language processing tasks, we need access to each word in a string. To access each word, we first have to break the text into smaller components. The method for breaking text into smaller components is called *tokenization* and the individual components are called *tokens*.

A few common operations that require tokenization include:

- Finding how many words or sentences appear in text
- Determining how many times a specific word or phrase exists
- Accounting for which terms are likely to co-occur

While tokens are usually individual words or terms, they can also be sentences or other size pieces of text.

To tokenize individual words, we can use `nltk`‘s `word_tokenize()` function. The function accepts a string and returns a list of words:

```python
from nltk.tokenize import word_tokenize

text = "Tokenize this text"
tokenized = word_tokenize(text)

print(tokenized)
# ["Tokenize", "this", "text"]

```

---

To tokenize at the sentence level, we can use `sent_tokenize()` from the same module.

```python
from nltk.tokenize import sent_tokenize

text = "Tokenize this sentence. Also, tokenize this sentence."
tokenized = sent_tokenize(text)

print(tokenized)
# ['Tokenize this sentence.', 'Also, tokenize this sentence.']

```

## **Normalization**

Tokenization and noise removal are staples of almost all text pre-processing pipelines. However, some data may require further processing through text normalization. Text *normalization* is a catch-all term for various text pre-processing tasks. In the next few exercises, we’ll cover a few of them:

- Upper or lowercasing
- Stopword removal
- *Stemming* – bluntly removing prefixes and suffixes from a word
- *Lemmatization* – replacing a single-word token with its root

The simplest of these approaches is to change the case of a string. We can use Python’s built-in String methods to make a string all uppercase or lowercase:

```python
my_string = 'tHiS HaS a MiX oF cAsEs'

print(my_string.upper())
# 'THIS HAS A MIX OF CASES'

print(my_string.lower())
# 'this has a mix of cases'

```

## **Stopword Removal**

Stopwords are words that we remove during preprocessing when we don’t care about sentence structure. They are usually the most common words in a language and don’t provide any information about the tone of a statement. They include words such as “a”, “an”, and “the”.

NLTK provides a built-in library with these words. You can import them using the following statement:

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

```

We create a set with the stop words so we can check if the words are in a list below.

Now that we have the words saved to `stop_words`, we can use tokenization and a list comprehension to remove them from a sentence:

```python
nbc_statement = "NBC was founded in 1926 making it the oldest major broadcast network in the USA"

word_tokens = word_tokenize(nbc_statement)
# tokenize nbc_statement

statement_no_stop = [word for word in word_tokens if word not in stop_words]

print(statement_no_stop)
# ['NBC', 'founded', '1926', 'making', 'oldest', 'major', 'broadcast', 'network', 'USA']

```

In this code, we first tokenized our string, `nbc_statement`, then used a list comprehension to return a list with all of the stopwords removed.

## **Stemming**

In natural language processing, *stemming* is the text preprocessing normalization task concerned with bluntly removing word affixes (prefixes and suffixes). For example, stemming would cast the word “going” to “go”. This is a common method used by search engines to improve matching between user input and website hits.

NLTK has a built-in stemmer called PorterStemmer. You can use it with a list comprehension to stem each word in a tokenized list of words.

First, you must import and initialize the stemmer:

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

```

Now that we have our stemmer, we can apply it to each word in a list using a list comprehension:

```python
tokenized = ['NBC', 'was', 'founded', 'in', '1926', '.', 'This', 'makes', 'NBC', 'the', 'oldest', 'major', 'broadcast', 'network', '.']

stemmed = [stemmer.stem(token) for token in tokenized]

print(stemmed)
# ['nbc', 'wa', 'found', 'in', '1926', '.', 'thi', 'make', 'nbc', 'the', 'oldest', 'major', 'broadcast', 'network', '.']

```

Notice, the words like ‘was’ and ‘founded’ became ‘wa’ and ‘found’, respectively. The fact that these words have been reduced is useful for many language processing applications. However, you need to be careful when stemming strings, because words can often be converted to something unrecognizable.

```python
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

populated_island = 'Java is an Indonesian island in the Pacific Ocean. It is the most populated island in the world, with over 140 million people.'

stemmer = PorterStemmer()

island_tokenized = word_tokenize(populated_island)

stemmed = [stemmer.stem(token) for token in island_tokenized]

try:
  print('A stemmer exists:')
  print(stemmer)
except:
  print('Expected a variable called `stemmer`')
try:
  print('Words Tokenized:')
  print(island_tokenized)
except:
  print('Expected a variable called `island_tokenized`')
try:
  print('Stemmed Words:')
  print(stemmed)
except:
  print('Expected a variable called `stemmed`')
```

## **Lemmatization**

*Lemmatization* is a method for casting words to their root forms. This is a more involved process than stemming, because it requires the method to know the part of speech for each word. Since lemmatization requires the part of speech, it is a less efficient approach than stemming.

In the next exercise, we will consider how to tag each word with a part of speech. In the meantime, let’s see how to use NLTK’s lemmatize operation.

We can use NLTK’s `WordNetLemmatizer` to lemmatize text:

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

```

Once we have the `lemmatizer` initialized, we can use a list comprehension to apply the lemmatize operation to each word in a list:

```python
tokenized = ["NBC", "was", "founded", "in", "1926"]

lemmatized = [lemmatizer.lemmatize(token) for token in tokenized]

print(lemmatized)
# ["NBC", "wa", "founded", "in", "1926"]

```

The result saved to `lemmatized` contains `'wa'`, while the rest of the words remain the same. Not too useful. This happened because `lemmatize()` treats every word as a noun. To take advantage of the power of lemmatization, we need to tag each word in our text with the most likely part of speech. We’ll do that in the next exercise.

```python
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

populated_island = 'Indonesia was founded in 1945. It contains the most populated island in the world, Java, with over 140 million people.'

lemmatizer = WordNetLemmatizer()

tokenized_string = word_tokenize(populated_island)

lemmatized_words = [lemmatizer.lemmatize(token) for token in tokenized_string]

try:
  print(f'A lemmatizer exists: {lemmatizer}')
except:
  print('Expected a variable called `lemmatizer`')
try:
  print(f'Words Tokenized: {tokenized_string}')
except:
  print('Expected a variable called `tokenized_string`')
try:
  print(f'Lemmatized Words: {lemmatized_words}')
except:
  print('Expected a variable called `lemmatized_words`')
```

## **Part-of-Speech Tagging**

To improve the performance of lemmatization, we need to find the part of speech for each word in our string. In **script.py**, to the right, we created a part-of-speech tagging function. The function accepts a word, then returns the most common part of speech for that word. Let’s break down the steps:

### **1. Import wordnet and Counter**

```python
from nltk.corpus import wordnet
from collections import Counter

```

- `wordnet` is a database that we use for contextualizing words
- `Counter` is a container that stores elements as dictionary keys

### **2. Get synonyms**

Inside of our function, we use the `wordnet.synsets()` function to get a set of synonyms for the word:

```python
def get_part_of_speech(word):
  probable_part_of_speech = wordnet.synsets(word)

```

The returned synonyms come with their part of speech.

### **3. Use synonyms to determine the most likely part of speech**

Next, we create a `Counter()` object and set each value to the count of the number of synonyms that fall into each part of speech:

```python
pos_counts["n"] = len(  [ item for item in probable_part_of_speech if item.pos()=="n"]  )
...

```

This line counts the number of nouns in the synonym set.

### **4. Return the most common part of speech**

Now that we have a count for each part of speech, we can use the `.most_common()` counter method to find and return the most likely part of speech:

```python
most_likely_part_of_speech = pos_counts.most_common(1)[0][0]

```

---

Now that we can find the most probable part of speech for a given word, we can pass this into our lemmatizer when we find the root for each word. Let’s take a look at how we would do this for a tokenized string:

```python
tokenized = ["How", "old", "is", "the", "country", "Indonesia"]

lemmatized = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized]

print(lemmatized)
# ['How', 'old', 'be', 'the', 'country', 'Indonesia']
# Previously: ['How', 'old', 'is', 'the', 'country', 'Indonesia']

```

Because we passed in the part of speech, “is” was cast to its root, “be.” This means that words like “was” and “were” will be cast to “be”.

```python
import nltk
from nltk.corpus import wordnet
from collections import Counter

def get_part_of_speech(word):
  probable_part_of_speech = wordnet.synsets(word)
  
  pos_counts = Counter()

  pos_counts["n"] = len(  [ item for item in probable_part_of_speech if item.pos()=="n"]  )
  pos_counts["v"] = len(  [ item for item in probable_part_of_speech if item.pos()=="v"]  )
  pos_counts["a"] = len(  [ item for item in probable_part_of_speech if item.pos()=="a"]  )
  pos_counts["r"] = len(  [ item for item in probable_part_of_speech if item.pos()=="r"]  )
  
  most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
  return most_likely_part_of_speech
```

```python
rom nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from part_of_speech import get_part_of_speech

lemmatizer = WordNetLemmatizer()

populated_island = 'Indonesia was founded in 1945. It contains the most populated island in the world, Java, with over 140 million people.'

tokenized_string = word_tokenize(populated_island)

lemmatized_pos = [lemmatizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized_string]

try:
  print(f'The lemmatized words are: {lemmatized_pos}')
except:
  print('Expected a variable called `lemmatized_pos`')
```

Result:

`The lemmatized words are: ['Indonesia', 'be', 'found', 'in', '1945', '.', 'It', 'contain', 'the', 'most', 'populate', 'island', 'in', 'the', 'world', ',', 'Java', ',', 'with', 'over', '140', 'million', 'people', '.']`