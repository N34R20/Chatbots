# Rule-Based Chat-bots

## ****Introduction****

Have you ever sought customer support help? Maybe you needed to pay a bill or make an inquiry about your bank account. There’s a good chance the first interaction you had was with a chatbot. Many companies, including airlines, [use rule-based chatbots](https://digiday.com/uk/klm-uses-artificial-intelligences-customer-service/) to handle customer support requests, like reporting flight delays and providing customers with boarding passes.

*Rule-based* chatbots use regular expression patterns to match user input to human-like responses that simulate a conversation with a real person. For example, if a customer wanted to check the status of their KLM flight, they may have an interaction like this:

```
- USER: Hey, do you know if flight 3984 is on time?
- KLM CHATBOT: Unfortunately, flight 3984 is delayed 30 minutes. Can I help you with anything else?
- USER: Nope, that's too bad.
- KLM CHATBOT: Okay, sorry for the inconvenience. Have a good day.

```

In the conversation above, the chatbot matched each statement by the user, and responded with something that made sense.

Many chatbots, including the airline example above, are called *dialog agents*, or *closed domain* chatbots because they are limited to conversations on a specific subject, such as checking flight status or getting a boarding pass.

In this lesson, you will learn how to build a closed domain, rule-based chatbot. We will introduce you to important rule-based concepts such as utterances, intents, and entities.

## **Greeting the User**

The first step for any rule-based chatbot is greeting the user and asking them how the chatbot can help. In this exercise, we will walk you through the following four steps to accomplish just that:

1. Get the name of the user
2. Ask the user, by name, if they need help
3. Exit the conversation if the user does not want help
4. Return the user’s help request if they want help

Let’s walk through the code for each of these steps.

### **1. Get the user’s name**

Throughout this lesson, we will use the `input` function to solicit responses from a user. For example, to greet a user and ask for their name, we would write:

```python
name = input("Hi, I'm a customer support representative. What is your name? ")

```

The user’s response will be saved to the variable, `name`.

### **2. Ask the user if they need help**

Now that you have the user’s name, you can insert it into the following question to ask if they need help:

```python
will_help = input(f"Okay {name}, what can I help you with? ")

```

Asking the user if they need help, with their name, is a personal touch and mimics how a human customer support representative would be trained to respond.

### **3. Exit the conversation if the user does not want help**

In **script.py**, there is a `SupportBot` attribute called `negative_responses`, which contains a tuple of words. We can use this tuple in a conditional to check if the user does not want help:

```python
if will_help in self.negative_responses:
    print("Ok, have a great day!")
    return

```

In this code, if the user responds “nothing” to the question, “What can I help you with?”, the chatbot will wish the user to have a good day and exit.

### **4. Return the user’s help request if they want help**

Finally, if the user wants help, return their response.

```python
return will_help

```

In a later exercise, we will match the text saved to `will_help` to a given intent.

## **Handling the Conversation**

The start of most customer support conversations is formulaic. In the last exercise, we welcomed the user, then asked for their name and if they wanted help. Once you’re beyond this initial welcome, the conversation can start to go in many different directions. Usually, chatbots have a central method to handle the back-and-forth of a conversation.

In **script.py**, we created a method called `.handle_conversation()` that will be our central method to continue responding for as long as a user asks questions.

Because we start our conversation with the `.welcome()` method, the first step is to call our conversation handling method:

```python
def welcome(self):
  ...
  self.handle_conversation(will_help)

```

To handle the indefinite length of a conversation, we use a `while` loop. The `while` loop can check if the user wants to continue the conversation after each response. Let’s say a user can only exit the conversation by responding with “stop.” Our `while` loop would look something like:

```python
def handle_conversation(self, reply):
  while not (reply == "stop"):
    reply = input("How can I help you?")

```

At the moment, this code will respond “How can I help you?” to every user response but “stop” – not too useful. However, the `while` loop, as you’ll see over the next few exercises, provides a lot of flexibility for processing user input and responding with something that makes sense.

## **Exiting the Conversation**

In the last exercise, you made a while loop to respond to a user until they replied “stop.” Saying the word “stop” isn’t the only, nor is it the typical, way to end a conversation with someone. Instead, you may say something like:

- I have to leave, bye.
- I need to go. I’ll come back later.

To account for the variety of statements someone could make to exit a conversation, we created a method called `.make_exit()`. The purpose of this method is to check whether the user wants to leave the conversation. If they do, it returns `True`. If not, it returns `False`.

The `.make_exit()` method accepts one argument, the user’s response. Within the method, we check whether the user’s response contains a word that is commonly used to exit a conversation, such as “bye” or “exit.” In **script.py**, we save a few of these words in a tuple called `exit_commands`.

```python
def make_exit(self, reply):
  for exit_command in self.exit_commands:
    if exit_command in reply:
      print("Nice talking with you!")
      return True

  return False

```

In the code above, we iterate over each word in `self.exit_commands`. If there is a match between the user’s input and a word from `self.exit_commands`, the method prints “Ok, have a nice day!” and returns `True`. If there is not a match, then the method returns `False`.

We can incorporate this method call into each loop using the following code:

```python
while not make_exit(reply):
  reply = input("something ")

```

If `.make_exit()` returns `True`, then the loop will stop executing, and the conversation will end.

## **Interpreting User Responses I**

At this point, we’ve built entry and exit paths to our rule-based chatbot. If someone were to interact with it, they would be greeted as if they were talking to a human, and they would be able to stop the conversation with a simple statement, like “goodbye.”

Now, we can shift our focus to the conversation. In addition to greeting and exiting, our chatbot will be able to do two other actions:

- Allow a user to pay their bill
- Tell the user how they can pay their bill

We often refer to these actions as *intents*. Before we can trigger an intent, the chatbot must interpret the user’s statement. We often refer to the user’s statement as an *utterance*. In **script.py**, we added a class variable called `matching phrases`, which is a dictionary with the following:

```python
self.matching_phrases = {'how_to_pay_bill': [r'.*how.*pay bills.*', r'.*how.*pay my bill.*']}

```

This dictionary contains a list with two strings for regular expression matching:

- `r'.*how.*pay bill.*'`
- `r'.*how.*pay my bill.*'`

In Exercise 7, we will use this `'how_to_pay_bill'` key-value pair to match user inputs like:

- “how can I pay my bill?”
- “how can I pay bills?”

## **Interpreting User Responses II**

At this point, our chatbot has a mapping from regular expression patterns that represent user utterance to chatbot intents. In this exercise, we’ll use the regular expression library’s `.match()` method to check if a user’s utterance matches one of these patterns.

In **script.py**, we added a method called `.match_reply()` that we will use to match user utterances. Let’s walk step-by-step through the code for matching user utterances:

### **1. Iterate over each item in the dictionary**

```python
def match_reply(self, reply):
  for key, values in self.matching_phrases.items():
    for regex_pattern in values:
      ...

```

In the code above, the first `for` loop iterates over each item in the `self.matching_phrases` dictionary. Inside of this, there is another `for` loop that we use to iterate over the matching patterns in the current list of regex patterns.

### **2. Check if user utterance matches a regular expression pattern**

```python
def match_reply(self, reply):
  for key, values in self.matching_phrases.items():
    for regex_pattern in values:

      found_match = re.match(regex_pattern, reply)
      ...

```

In the code above, we use the `re.match()` function to check if the current regular expression pattern matches the user utterance.

### **3. Respond if a match was made**

```python
def match_reply(self, reply):
  for key, values in self.matching_phrases.items():
    for regex_pattern in values:
      found_match = re.match(regex_pattern, reply)

      if found_match:
        reply = input("Great! I found a matching regular expression. Isn't that cool?")
        return reply
      ...

```

In the code above, we use a conditional to check if `found_match` is `True`. In a future exercise, we will trigger an intent inside of this conditional. Here, we use an `input()` statement to ask another question and then return the `reply`.

### **4. Respond if a match was not made**

```python
def match_reply(self, reply):
  for key, values in self.matching_phrases.items():
    for regex_pattern in values:
      found_match = re.match(regex_pattern, reply)
      if found_match:
        reply = input("Great! I found a matching regular expression. Isn't that cool?")
        return reply

    return input("Can you please ask your questions a different way? ")

```

If the `found_match` variable is `False`, then we return the response to the question, “Can you please ask your questions a different way?”

### **5. Call `.match_reply()` after every user response**

```python
def handle_conversation(self, reply):
  while not make_exit(reply):
    reply = match_reply(reply)

```

Finally, inside the `while` loop of `.handle_conversation()`, we need to call `.match_reply()` so that we check the user’s utterance every time we get a response.

## **Intents**

Now that the chatbot is set up to match user input, we can trigger a desirable response. An intent often maps to a function or method. For example, if you wanted to say to a virtual assistant, “Hey, play music,” it may match the user’s utterance to a function called `play_music()`.

In **script.py**, we added two methods to handle our intent actions: `self.how_to_pay_bill_intent()` and `self.pay_bill_intent()`. We can use a conditional statement inside `self.match_reply()` to trigger a method mapped to an intent if a match is found:

```python
def match_reply(self, reply):
    for key, values in self.matching_phrases.items():
      for regex_pattern in values:
        found_match = re.match(regex_pattern, reply)
        if found_match and key == 'how_to_pay_bill':
          return self.how_to_pay_bill_intent()

    return input("I did not understand you. Can you please ask your question again?")

```

In the above code, we call `self.how_to_pay_bill_intent()` if there is a match and the `key` is equal to `"how_to_pay_bill"`. Additionally, we return output from `self.how_to_pay_bill_intent()` .

Code Example:

```python
import re
import random

class SupportBot:
  negative_responses = ("nothing", "don't", "stop", "sorry")

  exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later")

  def __init__(self):
    self.matching_phrases = {'how_to_pay_bill': [r'.*how.*pay bills.*', r'.*how.*pay my bill.*'], r'pay_bill': [r'.*want.*pay my bill.*', r'.*need.*pay my bill.*']}

  def welcome(self):
    name = input("Hi, I'm a customer support representative. Welcome to Codecademy Bank. Before we can help you, I need some information from you. What is your first name and last name? ")
    
    will_help = input(f"Ok {name}, what can I help you with? ")
    
    if will_help in self.negative_responses:
      print("Ok, have a great day!")
      return
    
    self.handle_conversation(will_help)
  
  def handle_conversation(self, reply):
    while not self.make_exit(reply):
      reply = self.match_reply(reply)
      
  def make_exit(self, reply):
    for exit_command in self.exit_commands:
      if exit_command in reply:
        print("Ok, have a great day!")
        return True
      
    return False
  
  def match_reply(self, reply):
    for key, values in self.matching_phrases.items():
      for regex_pattern in values:
        found_match = re.match(regex_pattern, reply)
        if found_match and key == 'how_to_pay_bill':
          return self.how_to_pay_bill_intent()
        elif found_match and key == 'pay_bill':
          return self.pay_bill_intent()
        
    return input("I did not understand you. Can you please ask your question again?")
  
  def how_to_pay_bill_intent(self):
    return input("You can pay your bill a couple of ways. 1) online at bill.codecademybank.com or 2) you can pay your bill right now with me. Can I help you with anything else?")
  
  def pay_bill_intent(self):
    return input("inside self.pay_bill_intent()")
  
# Create a SupportBot instance
SupportConversation = SupportBot()
# Call the .welcome() method
SupportConversation.welcome()
```

## **Entities**

To improve the functionality of a chatbot, we can also parse the user’s response and grab important information. For example, if you wanted to play a song from a virtual assistant, you could say, “Hey, play Beethoven’s Fifth Symphony.” In this example, Beethoven’s Fifth Symphony would be passed into an intent of the virtual assistant that plays music. We call the information that a chatbot passes from a user statement to a triggered intent an *entity* – also referred to as a *slot*.

In the chatbot we’ve built in this lesson, we need to collect the user’s account number when we call the `.pay_bill_intent()` method to credit their account. We can do this by adding a regular expression, like the one below, to the `pay_bill` list in `self.matching_phrases`:

```python
regex = r'.*want.*pay.*my.*bill.*account.*number.*is (\d+)'

```

If an utterance matches the above statement the `(\d+)`, called a capture group, will grab the numeric value that follows the pattern. Notice, there is a space between “is” and `"(\d+)"`, rather than a `.*` pattern. With the regular expression above, we can use the following to grab and print an account number:

```python
reply = "i want to pay my bill. my account number is 888999333"
found_match = re.match(regex, reply)
print(found_match.groups()[0]) # Prints: '888999333'

```

In the above code, the `found_match` variable contains the account number. We can grab this number using the `.groups()` method. Because there is only one group, we can use `found_match.groups()[0]` to grab the first, and only, group.

Once we have the group, we can pass it into `.pay_bill_intent()`:

```python
def match_reply(self):
														  ...
  if found_match and (key == 'pay_bill'):
    return self.pay_bill_intent(found_match.groups()[0])

```

In the above code, we pass the account number, as an entity, into the `self.pay_bill_intent()` method if the `found_match` variable contains the account number. Otherwise, we call the method without passing the account number.

## **Review**

In this lesson, you learned how to build a rule-based chatbot that is capable of serving a couple of hypothetical user needs. The chatbot that we developed is small and does not offer a lot of functionality. However, you can extend and improve this chatbot to help with additional tasks by adding:

- Regular expressions for utterance matching
- Intents for additional functionality
- Slots to improve information passing

In addition to these features, chatbots can be extended to do far more sophisticated tasks by hooking them up to databases or using them to trigger a call to a human support representative. While these concepts are outside the scope of this lesson, it’s worth understanding that a chatbot can be more useful when fit into more complex software.