from nltk.corpus import wordnet
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def assign_product_to_class(class_descriptions, description_of_product):
    comparison_list = []
    description_of_product = list(set(description_of_product))
    description_of_product = [word for word in description_of_product if word not in stopwords.words('english')]
    for className in class_descriptions.keys():
        comparison_per_class = []
        for word1 in class_descriptions[className]:
            word_from_list1 = wordnet.synsets(word1)
            for word2 in description_of_product:
                word_from_list2 = wordnet.synsets(word2)
                if word_from_list1 and word_from_list2:
                    s = word_from_list1[0].wup_similarity(word_from_list2[0])
                    comparison_per_class.append(s)
        comparison_per_class = [item for item in comparison_per_class if item != None]
        list_of_similar_values = sorted(comparison_per_class, reverse=True)[:5]
        comparison_list.append([np.mean(list_of_similar_values), className])
    return sorted(comparison_list, reverse=True)

stemmer = PorterStemmer()
tknzr = TweetTokenizer()

classDescriptions = {
    "Camera & Photo": ["lens", "camera", "photo", "camcorder", "photography", "image", "film", "digital", "monitor", "record"],
    "Bedding & Bath": ["bed", "bath", "sheet", "towel", "shower", "tube", "bathroom", "bedroom", "pillow", "mattress", "sleep"],
    "Exercise & Fitness": ["exercise", "fitness", "sport", "games", "weight", "train", "resistance", "soccer", "tennis", "golf", "yoga", "basketball", "fit"]
}
for i in classDescriptions.keys():
    classDescriptions[i] = [stemmer.stem(word) for word in classDescriptions[i]]


file = pd.read_csv("./test_set2.csv", delimiter=";", encoding='latin-1')


list_of_products = list(zip(file["Product_id"].tolist(), file["Description"], file["Category"]))
list_of_products_ready = [list(elem) for elem in list_of_products]

real_label = []
prediction = []

for i in range(len(list_of_products_ready)):
    # Tokenize the sentence
    tokenized_words = tknzr.tokenize(list_of_products_ready[i][1])
    list_of_products_ready[i].pop(1)
    # Stem the words
    stemed_words = [stemmer.stem(plural) for plural in tokenized_words]
    # Tag the morphology of the word
    tagged_words = pos_tag(stemed_words)
    # Only select the NN and NNP
    only_nouns = [word for word, pos in tagged_words if pos == 'NN' or pos == 'NNP']
    # Append the resulting words
    list_of_products_ready[i].append(only_nouns)

    # Start classification
    similatiry_to_classes = assign_product_to_class(classDescriptions, list_of_products_ready[i][2])
    list_of_products_ready[i].insert(2, similatiry_to_classes[0][1])

    real_label.append(list_of_products_ready[i][1])
    prediction.append(list_of_products_ready[i][2])
    print(list_of_products_ready[i])


print(confusion_matrix(real_label, prediction))

print(classification_report(real_label, prediction, target_names=["Exercise & Fitness", "Camera & Photo", "Bedding & Bath"]))