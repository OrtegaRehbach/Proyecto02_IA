import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4')

# -----------------------------------------------------------------------------------------
# Initialize globals
# -----------------------------------------------------------------------------------------

DIVIDER = '-'*50

# Min probability for an email to be considered spam
MIN_SPAM_PROB = 0.8

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Initialize an empty dictionary to store word counts
word_counts = defaultdict(lambda: {'ham': 0, 'spam': 0})

# Initialize counters for spam and ham emails
spam_count = 0
ham_count = 0

# -----------------------------------------------------------------------------------------
# Function definitions
# -----------------------------------------------------------------------------------------

def is_not_stopword(token: str):
    return token.lower() not in stop_words

def is_not_punctuation(token: str):
    return all([c not in string.punctuation for c in token])

def is_not_digits(token: str):
    return not token.isnumeric()

def stem_word(token: str):
    return stemmer.stem(token)

def lemmatize_word(token: str):
    return lemmatizer.lemmatize(token)

def remove_punctuation(token: str):
    new_token = "".join([c if is_not_punctuation(c) else "" for c in token])
    return new_token

def update_word_counts(tokens: list, t_class: str):
    # Add words to word_dict
    for token in set(tokens):
        word_counts[token][t_class] += 1

def process_text(text: str):
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Process tokens
    tokens_without_punctuation = filter(is_not_punctuation, tokens)
    tokens_without_numbers = filter(is_not_digits, tokens_without_punctuation)
    filtered_tokens = filter(is_not_stopword, tokens_without_numbers)
    stemmed_tokens = map(stem_word, filter(is_not_stopword, filtered_tokens))
    lemmatized_tokens = map(lemmatize_word, map(stem_word, filter(is_not_stopword, stemmed_tokens)))
    # Return processed tokens
    return [t for t in list(lemmatized_tokens)]
    
def count_spam_and_ham(data: pd.DataFrame):
    global spam_count, ham_count  # Declare as global variables
    # Count the number of spam and ham emails
    for t_class in data["class"]:
        if t_class == "spam":
            spam_count += 1
        elif t_class == "ham":
            ham_count += 1

def p_word(word: str):
    p_word_is_spam = word_counts[word]["spam"] / spam_count
    p_word_is_ham = word_counts[word]["ham"] / ham_count
    return p_word_is_spam / (p_word_is_ham + p_word_is_spam)

def p_text_is_spam_given_words(text: str, verbose=False):
    t1 = 0
    t2 = 0
    caught_words = []
    processed_tokens = process_text(text)   # Tokenize and clean the text
    for i in range(len(processed_tokens)):
        token = processed_tokens[i]
        if verbose: print("Checking token:", token)
        if token not in word_counts.keys():     # Ignore tokens not found in word_counts dict
            if verbose: print("Ignored token:", token)
            continue
        if t1 == 0 and t2 == 0: # Set the first values of t1 and t2 (only for the first word)
            t1 = p_word(token)
            t2 = 1 - t1
            if verbose: print(f"Initial values set: t1={t1} t2={t2}")
        else:   # For every other word, multiply the probabilities
            t1 *= p_word(token)
            t2 *= 1 - t1
            if verbose: print(f"Values set: t1={t1} t2={t2}")
        caught_words.append(token)
    if verbose: print(f"Final values: t1={t1} t2={t2}")
    if t1 == t2 == 0: # If no words were caught, return 0 and an empty list
        return 0.0, []
    # Return the overall probability and the list of caught words
    probability = t1 / (t1 + t2)
    return probability, caught_words

# -----------------------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------------------

print("Loading data...")
data = pd.read_csv("./res/spam-2.csv", encoding = 'latin1')
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)  # drop unavailable attributes
data = data.rename(columns ={"v1":"class", "v2":"text"})                # rename columns
print("Done!")

print("Processing data...")
# Divide the data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size = 0.2, random_state = 101, stratify=data['class'])

# Construct the train data
train_data = pd.DataFrame(X_train)
train_data.loc[:, "class"] = y_train
# Count amount of spam and ham emails
count_spam_and_ham(train_data)

# Process the data (remove punctuation, lemmatize, stem, ...)
for text, t_class in zip(train_data["text"], train_data["class"]):
    processed_tokens = process_text(text)
    update_word_counts(processed_tokens, t_class)   # Update the word_counts dict
print("Done!")

print("Calculating performance metrics...")
# Calculate probabilties for test data
y_predicted = []
for x in X_test:
    probability, words = p_text_is_spam_given_words(x, verbose=False)
    y_predicted.append("spam" if probability >= MIN_SPAM_PROB else "ham")
print("Done!")

conf_matrix = confusion_matrix(y_test, y_predicted)
accuracy = accuracy_score(y_test, y_predicted)
precision_spam = precision_score(y_test, y_predicted, pos_label='spam')
precision_ham = precision_score(y_test, y_predicted, pos_label='ham')
recall_spam = recall_score(y_test, y_predicted, pos_label='spam')
recall_ham = recall_score(y_test, y_predicted, pos_label='ham')
f1_spam = f1_score(y_test, y_predicted, pos_label='spam')
f1_ham = f1_score(y_test, y_predicted, pos_label='ham')

# Calculate performance metrics
print(DIVIDER)
print("Performance Metrics:")
print(DIVIDER)
print("Confusion_matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision (SPAM):", precision_spam)
print("Precision (HAM):", precision_ham)
print("Recall (SPAM):", recall_spam)
print("Recall (HAM):", recall_ham)
print("F1-Score (SPAM):", f1_spam)
print("F1-Score (HAM):", f1_ham)

# text2 = "WINNER!! As a valued network customer you have been selected to receivea �900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
# # text2 = "I HAVE A DATE ON SUNDAY WITH WILL!!"

# print(text2)
# print("Checking text...")
# p_text2_is_spam, words = p_text_is_spam_given_words(text2)
# print(f"Text is spam: {p_text2_is_spam:.5f}")
# print("The following words were caught:", words)

verbose_printing = True
exit = False
while not exit:
    print(DIVIDER)
    print("Spam & Ham Classifier")
    print(f"Verbose: {verbose_printing}")
    print(DIVIDER)
    print("1. Enter text")
    print("2. Toggle verbose printing")
    print("3. Exit")
    print(DIVIDER)
    option = input("Enter option: ")
    print(DIVIDER)
    if option == "1":
        text = input("Enter text: ")
        print(DIVIDER)
        print("Calculating probability...")
        p_text_is_spam, words = p_text_is_spam_given_words(text, verbose=verbose_printing)
        print("Done!")
        print(DIVIDER)
        print("Text was found to be {}".format("SPAM" if p_text_is_spam >= MIN_SPAM_PROB else "HAM"))
        print(f"With a {p_text_is_spam:.2%} probability of being SPAM")
        print("Caught words:", words)
        if verbose_printing:
            for word in words:
                print(f"'{word}' -> Probability: {p_word(word)}")
    elif option == "2":
        verbose_printing = not verbose_printing
        print(f"Toggled verobse printing to '{verbose_printing}'")
    elif option == "3":
        exit = True
    else:
        print(f"'{option}' is not a valid option")