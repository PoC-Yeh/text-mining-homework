import pandas as pd
import numpy as np
import operator

#text
with open("/Users/AllisonYeh/TMU/text_mining/PA-2/TMBD_news_files_2000.txt", 'r', encoding='UTF-8') as c:
    text = c.readlines()
 
#punctuations
with open("/Users/AllisonYeh/TMU/text_mining/PA-2/punctuation.txt", 'r', encoding='UTF-8') as c:
    pun = list(c.read())
punctuation = [p for p in pun if pun.index(p) %2 == 0]

#stopwords
with open("/Users/AllisonYeh/TMU/text_mining/PA-2/stopword_chinese.txt", 'r', encoding='UTF-8') as c:
    stop = c.readlines()
stop_word = [s.strip() for s in stop]




#calculate TF, DF
doc_term = {}   #{1:{termA: 2, termB: 1,...}, 2:{termF: 6, termK: 10}...}
all_term = {}   #{term1: [TF, DF], term2: [TF, DF]...}


for count in range(1, 2001):
    test = text[count - 1].replace("\u3000", "").split(" ")[:-1]    #replace"\u3000" with ""
    test[0] = test[0].split("\t")[1]
    new_test = [t for t in test if len(t) > 0]

    f1 = list(filter(lambda x: x not in punctuation, new_test))   #text without punctuation
    f2 = list(filter(lambda x: x not in stop_word, f1))     #text without stopword

    doc_term[count] = {}

    for i in f2:
        if i in list(doc_term[count].keys()):
            doc_term[count][i] += 1
        else:
            doc_term[count][i] = 1

    for x in list(doc_term[count].keys()):
        if x in list(all_term.keys()):
            all_term[x][0] += doc_term[count][x]   #all_term[termA] = [TF, DF]
            all_term[x][1] += 1
        else:
            all_term[x] = [0, 1]
            all_term[x][0] += doc_term[count][x]
        
#print(doc_term)
#print(all_term)




#Transform dict into dataframe, and use dataframe to calculate IDF and Term Weight
df_all_term = pd.DataFrame(all_term,index=['TF', 'DF']).T
df_all_term['IDF'] = np.log10(2000/df_all_term['DF'])
df_all_term["weight"] = df_all_term["TF"] * df_all_term["IDF"]
print(df_all_term)

#also turn doc_term into dataframe
doc_df = pd.DataFrame(doc_term)
doc_df.fillna(0, inplace = True)  #not appear (which resulted as NaN when turned into dataframe)-> 0
doc_df[doc_df > 0] = 1  #appear -> 1

df_join = df_all_term.join(doc_df)


#calculate the vector of each paragraph
for i in range(1, 2001):
    col_name = "vector_{}".format(i)
    df_join[col_name] = df_join[i] * df_join["weight"]
    #df_join["vector_1"] = df_join[1] * df_join["weight"]


#calculate similarity between each paragraph and paragrahp no.56
similarity = {}

for i in range(1, 2001):
    vector_count = "vector_{}".format(i)
    numerator = sum(df_join["vector_56"] * df_join[vector_count]) 
    denominator_1 = (sum(df_join["vector_56"] ** 2)) ** 0.5 
    denominator_2 = (sum(df_join[vector_count] ** 2)) ** 0.5
    
    similar = numerator / (denominator_1 * denominator_2)
    similarity[i] = similar
    

sort_similarity = sorted(similarity.items(), key=operator.itemgetter(1), reverse=True)




#output txt
result_list = ["ID Similarity\n"]
for i in sort_similarity:
    inside_list = []
    inside_list.append(str(i[0]))
    inside_list.append(str(i[1]))
    new = " ".join(inside_list)+ "\n"
    result_list.append(new)
print(result_list[:11])


f = open('similarity_result.txt', 'w', encoding = 'UTF-8') 
for i in result_list:
    f.write(i)
f.close()
