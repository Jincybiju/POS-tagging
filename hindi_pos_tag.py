from nltk.tag import tnt
from nltk.corpus import indian
import nltk
import os
import re
from googletrans import Translator
from nltk.tree import Tree

translator = Translator()
sentence_id = 0


temp_f = open("input2.txt", "r+", encoding="utf-8-sig")
text = temp_f.read()
print("Input :: \n ", text)




def train_hindi_model(model_path):
    train_data = indian.tagged_sents(model_path)
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    return tnt_pos_tagger


model = train_hindi_model(os.getcwd() + '/data/hindi.pos')
new_tagged = (model.tag(nltk.word_tokenize(text)))
print("\n\n tagged list BEFORE TRAINING model")
print(new_tagged)


def get_sentId(model_path):
    ids = re.compile('<Sentence\sid=\d+>')
    with open(model_path, "r+", encoding ="utf8") as temp_f:
        content = temp_f.readlines()
        for i in content:
            id_found = (ids.findall(i))
            if id_found:
                id_found = str(id_found).replace("['<Sentence id=", "").replace(">']", "")
                id = int(id_found)
    id = id + 1
    return id


# Function to tag words
def tag_words(model, text):
    tagged = (model.tag(nltk.word_tokenize(text)))
    return tagged


# Function for extracting keywords
def get_keywords(pos):
    grammar = r"""NP:{<NN.*>}"""
    chunkParser = nltk.RegexpParser(grammar)
    chunked = chunkParser.parse(pos)
    continuous_chunk = set()
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.add(named_entity)
                current_chunk = []
            else:
                continue
    return (continuous_chunk)



model = train_hindi_model(os.getcwd() + '/data/hindi.pos')
tagged_words = tag_words(model, text)

sentence_id = (get_sentId(os.getcwd() + '/data/hindi.pos'))
print("\n\nSentence ID \n\n")
print(sentence_id)


result_list = []
for nep_word, tag in tagged_words:
    if tag == "Unk":
        x = translator.translate(nep_word)
        if x is not None:
            str1 = str(x)
            new_str = str1.split()
            for j in new_str:
                if re.search('^text=', j, re.I):
                    word = j.replace("text=", ",").replace(",", "")
                    word = str(word)
                    # pos=nltk.pos_tag(word)
                    pos = nltk.tag.pos_tag([word])
                    # print (i, pos)
                    for en_word, tag in pos:
                        result = nep_word + "_" + (tag) + " "
                        result_list.append(result)

    else:
        result = nep_word + "_" + (tag) + " "
        result_list.append(result)

writing_word = str("\n<Sentence id=") + str(sentence_id) + ">\n"
output = writing_word + "".join(result_list) + "\n</Sentence>\n</Corpora>"


with open(os.getcwd() + '/data/hindi.pos', "r+", encoding="utf8") as f1:
    for line in f1.readlines():
        f1.write(line.replace("</Corpora>", ""))
    f1.write(output)


"""
Retrain the model
"""
model = train_hindi_model(os.getcwd() + '/data/hindi.pos')
final_tagged = tag_words(model, text)
print("\n\n Final Tagged list after Retraining the model \n\n")
print(final_tagged)
print("\n\n Finish...Writing to File....")


#Write Final Output to File
with open(os.getcwd()+"/result/S_POS_output.txt", "a", encoding="utf8") as output_file:
     output_file.write("\n[INPUT]\n")
     output_file.write(text)
     output_file.write("\n[OUTPUT]\n")
     output_file.write(str(final_tagged))


print("\n\nDone....\n\n")


print(get_keywords(final_tagged))
