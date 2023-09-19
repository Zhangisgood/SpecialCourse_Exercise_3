import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

with open('mobydick.txt', 'r', encoding='utf-8') as file:
    text = file.read()

tokens = word_tokenize(text)

filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]

pos_tags = nltk.pos_tag(filtered_tokens)

pos_freq = FreqDist(tag for (word, tag) in pos_tags)
common_pos = pos_freq.most_common(5)
print("Top 5 most common parts of speech and their frequencies:")
for pos, freq in common_pos:
    print(f"{pos}: {freq}")

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word, pos='v') for (word, pos) in pos_tags[:20]]
print("Lemmatized top 20 tokens:")
for token in lemmatized_tokens:
    print(token)

plt.figure()
pos_freq.plot(30, cumulative=False)
plt.show()
