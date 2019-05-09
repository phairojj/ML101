import deepcut

neg_text = [(line.strip(), '1') for line in open("data/neg.txt", 'r', encoding="utf-8")]
print(type(neg_text),neg_text)
neutral_text = [(line.strip(), '2') for line in open("data/neutral.txt", 'r', encoding="utf-8")]
pos_text = [(line.strip(), '3') for line in open("data/pos.txt", 'r', encoding="utf-8")]

for a,b in neg_text + neutral_text + pos_text:
    print(a,"-", b)
# Combine All Data Together
full_data = [(deepcut.tokenize(sentence), sentiment) for (sentence, sentiment) in neg_text + neutral_text + pos_text]

#print(full_data)