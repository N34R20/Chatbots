# Generative ChatBots

## **Introduction to Generative Chatbots**

We’ve discussed how seq2seq models were designed for machine translation, but generating dialog can also be accomplished with seq2seq. Rather than converting input in one language to output in another language, we can convert dialog input into a likely corresponding response.

In fact, the only major difference between the code used for a machine translation program and a generative chatbot is the dataset we use to train our model!

What about the three components of closed-domain systems: intent classification, entity recognition, and response selection? When building an open-domain chatbot, intent classification is much harder and an infinite number of intents are possible. Entity recognition is just ignored in favor of the trained “black box” model.

While closed-domain architecture is focused on response selection from a set of predefined responses, generative architecture allows us to perform unbounded text generation. Instead of selecting full sentences, the open-domain model generates word by word or character by character, allowing for new combinations of language.

In the seq2seq decoding function, the decoder generates several possible output tokens and the one with the highest probability (according to the model) gets selected.

## **Choosing the Right Dataset**

One of the trickiest challenges in building a deep learning chatbot program is choosing a dataset to use for training.

Even though the chatbot is more open-domain than other architectures we’ve come across, we still need to consider the context of the training data. Many sources of “open-domain” data which hopefully will capture unbiased, broad human conversation, actually have their own biases which will affect our chatbot. If we use customer service data, Twitter data, or Slack data, we’re setting our potential conversations up in a particular way.

Does it matter to us if our model is trained on “authentic” dialog? If we prefer that the model use complete sentences, then TV and movie scripts could be great resources.

Depending on your use case for the chatbot, you may also need to consider the license associated with the dataset.

Of course, there are ethical considerations as well here. If our training chat data is biased, bigoted, or rude, then our chatbot will learn to be so too.

Once we’ve selected and preprocessed a set of chat data, we can build and train the seq2seq model using the chosen dataset.

You may recall a previous lesson on seq2seq models involving a rudimentary machine translation program on the Codecademy platform. Similarly, we will be building a generative chatbot on the platform that won’t be passing any Turing tests. Again, you will have the opportunity to build a far more robust chatbot on your own device by increasing the quantity of chat data used (or changing the source entirely), increasing the size and capacity of the model, and allowing for more training time.

## **Setting Up the Bot**

Just as we built a chatbot class to handle the methods for our rule-based and retrieval-based chatbots, we’ll build a chatbot class for our generative chatbot.

Inside, we’ll add a greeting method and a set of exit commands, just like we did for our closed-domain chatbots.

However, in this case, we’ll also import the seq2seq model we’ve built and trained on chat data for you, as well as other information we’ll need to generate a response.

As it happens, many cutting-edge chatbots blend a combination of rule-based, retrieval-based, and generative approaches in order to easily handle some intents using predefined responses and offload other inputs to a natural language generation system.

## **Generating Chatbot Responses**

As you may have noticed, a fundamental change from one chatbot architecture to the next is how the method that handles conversation works. In rule-based and retrieval-based systems, this method checks for various user intents that will trigger corresponding responses. In the case of generative chatbots, the seq2seq test function we built for the machine translation will do most of the heavy lifting for us!

For our chatbot we’ve renamed `decode_sequence()` to `.generate_response()`. As a reminder, this is where response generation and selection take place:

1. The encoder model encodes the user input
2. The encoder model generates an embedding (the last hidden state values)
3. The embedding is passed from the encoder to the decoder
4. The decoder generates an output matrix of possible words and their probabilities
5. We use NumPy to help us choose the most probable word (according to our model)
6. Our chosen word gets translated back from a NumPy matrix into human language and added to the output sentence

## **Handling User Input**

Hmm… why can’t our chatbot chat? Right now our `.generate_response()` method only works with preprocessed data that’s been converted into a NumPy matrix of one-hot vectors. That won’t do for our chatbot; we don’t just want to use test data for our output. We want the `.generate_response()` method to accept new user input.

Luckily, we can address this by building a method that translates user input into a NumPy matrix. Then we can call that method inside `.generate_response()` on our user input.

We said it before, and we’ll say it again: we’re adding deep learning in now, so running the code may take a bit more time again.

```python
import numpy as np
import re
from seq2seq import encoder_model, decoder_model, num_decoder_tokens, num_encoder_tokens, input_features_dict, target_features_dict, reverse_target_features_dict, max_decoder_seq_length, max_encoder_seq_length

class ChatBot:
  
  negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")

  exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")
  
  def start_chat(self):
    user_response = input("Hi, I'm a chatbot trained on dialog from The Princess Bride. Would you like to chat with me?\n")
    
    if user_response in self.negative_responses:
      print("Ok, have a great day!")
      return
    
    self.chat(user_response)
  
  def chat(self, reply):
    while not self.make_exit(reply):
      reply = input(self.generate_response(reply))
    
  # define .string_to_matrix() below:
  def string_to_matrix(self, user_input):
    tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
    user_input_matrix = np.zeros(
(1, max_encoder_seq_length, num_encoder_tokens),
dtype='float32')
    for timestep, token in enumerate(tokens):
      user_input_matrix[0, timestep, input_features_dict[token]] = 1.
    return user_input_matrix
  
  def generate_response(self, user_input):
    # change user_input into a NumPy matrix:
    input_matrix = self.string_to_matrix(user_input)
    # update argument for .predict():
    states_value = encoder_model.predict(input_matrix)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_features_dict['<START>']] = 1.
    
    chatbot_response = ''

    stop_condition = False
    while not stop_condition:
      output_tokens, hidden_state, cell_state = decoder_model.predict(
        [target_seq] + states_value)

      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      sampled_token = reverse_target_features_dict[sampled_token_index]
      chatbot_response += " " + sampled_token
      
      if (sampled_token == '<END>' or len(chatbot_response) > max_decoder_seq_length):
        stop_condition = True
        
      target_seq = np.zeros((1, 1, num_decoder_tokens))
      target_seq[0, 0, sampled_token_index] = 1.
      
      states_value = [hidden_state, cell_state]
      
    return chatbot_response
  
  def make_exit(self, reply):
    for exit_command in self.exit_commands:
      if exit_command in reply:
        print("Ok, have a great day!")
        return True
      
    return False
  
chatty_mcchatface = ChatBot()
# call .generate_response():
chatty_mcchatface.generate_response("What?")
```

## **Handling Unknown Words**

Nice work! Our chatbot now knows how to accept user input. But there is a pretty large caveat here: our chatbot only knows the vocabulary from our training data. What if a user uses a word that the chatbot has never seen before?

With our current code, we’ll get a `KeyError`:

This is because in `.string_to_matrix()` we are looking for `token` in `input_features_dict`:

```python
for timestep, token in enumerate(tokens):
  user_input_matrix[0, timestep, input_features_dict[token]] = 1.

```

Currently, if the token doesn’t exist in the `input_features_dict` dictionary (which keeps track of all words in the training data), our program has no way of handling it.

Here are a few popular approaches to tackle unknown words:

- Tell the chatbot to ignore them, which is the simplest fix for smaller datasets, but can never generate those words as output. (Can you imagine scenarios when this could be a problem?)
- Pause the chat process and have the chatbot ask what the entire utterance means. This requires the user to rephrase the entire utterance. This causes issues when working with a fairly limited dataset, since we may end up with the chatbot repeatedly asking the user to rephrase each input statement.
- Add in a step for the chabot to register any unknown word as a `'<UNK>'` token. This is generally more complicated than the other two solutions. It would require that the training data is built out with `'<UNK>'` tokens and requires several extra manual steps.

## **Review**

Hooray! While our chatbot is far from perfect, it does produce innovative babble, allowing us to create a truly open-domain conversation. Just like with a machine translation model, we want to accommodate new sentence structures and creativity, something we can do with a generative model for our chatbot.

Of course, even with a better trained model, there are some issues that we need to consider.

First, there are ethical considerations that are much harder to discern and track when working with deep learning models. It’s very easy to accidentally train a model to reproduce latent biases within the training data.

We’re also making a pretty big assumption right now in our chatbot architecture: the chatbot responses only depend on the previous turn of user input. The seq2seq model we have here won’t account for any previous dialog. For example, our chatbot has no way to handle a simple back-and-forth like this:

User: “Do you have any siblings?”

Chatbot: “Yes, I do”

User: “How many?”

Chatbot: 🤯

In addition to topics that have been previously covered, the entire context of the chat is missing. Our chatbot doesn’t know anything about the user, their likes, their dislikes, etc.

As it happens, handling context and previously covered topics is an active area of research in NLP. Some proposed solutions include:

- training the model to hang onto some previous number of dialog turns
- keeping track of the decoder’s hidden state across dialog turns
- personalizing models by including user context during training or adding user context as it is included in the user input

```python
import numpy as np
import re
from seq2seq import encoder_model, decoder_model, num_decoder_tokens, num_encoder_tokens, input_features_dict, target_features_dict, reverse_target_features_dict, max_decoder_seq_length, max_encoder_seq_length

class ChatBot:
  
  negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")

  exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")
  
  def start_chat(self):
    user_response = input("Hi, I'm a chatbot trained on dialog from The Princess Bride. Would you like to chat with me?\n")
    
    if user_response in self.negative_responses:
      print("Ok, have a great day!")
      return
    
    self.chat(user_response)
  
  def chat(self, reply):
    while not self.make_exit(reply):
      reply = input(self.generate_response(reply))
    
  def string_to_matrix(self, user_input):
    tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
    user_input_matrix = np.zeros(
      (1, max_encoder_seq_length, num_encoder_tokens),
      dtype='float32')
    for timestep, token in enumerate(tokens):
      if token in input_features_dict:
        user_input_matrix[0, timestep, input_features_dict[token]] = 1.
    return user_input_matrix
  
  def generate_response(self, user_input):
    input_matrix = self.string_to_matrix(user_input)
    states_value = encoder_model.predict(input_matrix)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_features_dict['<START>']] = 1.
    
    chatbot_response = ''

    stop_condition = False
    while not stop_condition:
      output_tokens, hidden_state, cell_state = decoder_model.predict(
        [target_seq] + states_value)
      
      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      sampled_token = reverse_target_features_dict[sampled_token_index]
      
      chatbot_response += " " + sampled_token
      
      if (sampled_token == '<END>' or len(chatbot_response) > max_decoder_seq_length):
        stop_condition = True
        
      target_seq = np.zeros((1, 1, num_decoder_tokens))
      target_seq[0, 0, sampled_token_index] = 1.
      
      states_value = [hidden_state, cell_state]
      
    chatbot_response = chatbot_response.replace("<START>", "").replace("<END>", "")
      
    return chatbot_response
  
  def make_exit(self, reply):
    for exit_command in self.exit_commands:
      if exit_command in reply:
        print("Ok, have a great day!")
        return True
      
    return False
  
chatty_mcchatface = ChatBot()
chatty_mcchatface.start_chat()
```