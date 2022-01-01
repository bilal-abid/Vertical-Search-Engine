#Task 1: Vertical Search Engine

	# Queue = [seed];  data_file = pd.DataFrame(columns=["df_paper_title", "df_paper_URL", "df_paper_date",
    #                                                    "df_paper_Author","df_Author_URL",
    #                                                    "df_Co_Author"]);data_Queue = {} #this is the queue which initially contains the given seed URL

#Step1: Building a web crawler

import itertools
import threading
import time
import sys
def sleep():
	done = False
	#here is the animation
	def animate():
		for c in itertools.cycle(['|', '/', '-', '\\']):
			if done:
				break
			sys.stdout.write('\rloading ' + c)
			sys.stdout.flush()
			time.sleep(0.1)
		sys.stdout.write('\rDone!     ')

	t = threading.Thread(target=animate)
	t.start()

	#long process here
	time.sleep(2)
	done = True

import requests
from bs4 import BeautifulSoup
import pandas as pd

def mycrawler(seed, maxcount):
	Queue = [seed];  data_file = pd.DataFrame(columns=["df_paper_title","df_paper_URL",
                                                       "df_paper_Author","df_Author_URL"]);data_Queue = {} #this is the queue which initially contains the given seed URL
	count = 0; #i =0;
	while(Queue!=[] and count < maxcount):
            count +=1
            url = Queue.pop(0)
            print("fetching " + url)
            #print(Queue)

            code = requests.get(url)
            plain = code.text
            s = BeautifulSoup(plain, "html.parser")
            #print(s.prettify())
            results = s.find_all('div', class_='result-container')
            #print(results)
            for x in range(1, 14):
                pagination = 'https://pureportal.coventry.ac.uk/en/organisations/school-of-life-sciences/publications/?page=' + str(x)
                #print(pagination)
                Queue.append(pagination)
            print("Total Publications: ",len(results)*x)
            for result in results:
                #print('helo')
                title = result.find('h3', class_='title')
                print("=================================================================================================================")
                # tit = title.string
                # print(tit)
                print(title.string)
                data_Queue['df_paper_title'] = title.string#.strip()
                title_URL = title.find('a', class_='link')
                Actual_URL = title_URL.get('href')
                if (Actual_URL != None and Actual_URL != '/'):
                    Actual_URL = Actual_URL#.strip() ##The strip() method removes any leading (spaces at the beginning) and trailing (spaces at the end) characters (space is the default leading character to remove)
                    print(Actual_URL)
                    data_Queue['df_paper_URL'] = Actual_URL.strip()
                    title_Date = result.find('span', class_='date') or result.find('span', class_='prize_shortrenderer_date')
                    print(title_Date.string)
                    # data_Queue['df_paper_date'] = title_Date.string#.strip()
                    Authors = result.find_all('a', class_='link person')
                    for Author in Authors:
                        Author_URL = Author.get('href')
                        if (Author_URL != None and Author_URL != '/'):
                            Author_URL = Author_URL.strip()  ##The strip() method removes any leading (spaces at the beginning) and trailing (spaces at the end) characters (space is the default leading character to remove)
                            print("Author:", Author.string, "URL:",Author_URL)
                            data_Queue['df_paper_Author'] = Author.string#.strip()
                            data_Queue['df_Author_URL'] = Author_URL#.strip()
                            # data_Queue['df_Co_Author'] = Author.next_sibling#.strip()
                            data_file = data_file.append(data_Queue, ignore_index=True)
                            Queue.append(Author_URL)
                print("=================================================================================================================")
            print("BFS Queue----------->", '\n',Queue)
            data_file.to_csv("./Data_Extracted_file.csv", index=False)
            print("Data_Extracted_file.csv saved")
            print("**********************************************************************************************************************************************")
            sleep()

seed = 'https://pureportal.coventry.ac.uk/en/organisations/school-of-life-sciences/publications/'
mycrawler(seed,13) #use any number, instead of 10, to control the number of pages crawled
print("Loading Data File----->")
#sleep()
#print('\n')
dataset_File = pd.read_csv("Data_Extracted_file.csv")
print(dataset_File)
#print(type(dataset_File))


#Step2: Construction of Incidence Matrix Index

#1- Pre-processing on the data (remove stopwords, and lowercase all )
import pandas as pd

dataset_File = pd.read_csv("Data_Extracted_file.csv")
f = open('Data_Extracted_file.csv', 'r',encoding='UTF8')
docs = f.read().split("\n")  #docs = f.read().split() for spliting further into single words
print("Before Pre-processing---------->",'\n')
print(docs)

import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords

sw = stopwords.words('english')
#print(sw)

#nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()
filtered_docs = []
for doc in docs:
    tokens = word_tokenize(doc)#;print(tokens)
    tmp = ""
    for w in tokens:
        if w not in sw:
            tmp += ps.stem(w) + " "#;print("tmp-----",tmp);#print("w-----",w)
    filtered_docs.append(tmp)
#sleep()
print('\n\n\n\n',"After Pre-processing---------->",'\n',filtered_docs)
#print(type(filtered_docs))

#2- Incidence Matrix Index
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer #Bag of Model
vectorizer = CountVectorizer()                              #Bag of Model
X = vectorizer.fit_transform(filtered_docs)
# #sleep()
#print('\n\n\n',X)
print('\n\n\n',X.todense())
X = X.T.toarray()
#print('\n\n\n',X)

df = pd.DataFrame(X, index=vectorizer.get_feature_names_out())
print('\n\n', df)
df.to_csv("./Incidence_Index_file.csv", index=False)




#Step3: Fully working querry processor
def get_Query(q, df):
	filtered_test_docs = []
	for doc in q:
		tokens = word_tokenize(doc)
		tmp = ""
		for w in tokens:
			if w not in sw:
				tmp += ps.stem(w) + " "
		filtered_test_docs.append(tmp)

	print('\n','\n',"User query:", filtered_test_docs, '\n')
	print("Results with the highest cosine similarity values: ")
	# Convert the query become a vector
	q = [filtered_test_docs]
	q_vec = vectorizer.transform(filtered_test_docs).toarray().reshape(df.shape[0], )
	sim = {}
	# Calculate the similarity
	for i in range(10):
		sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)

	# Sort the values
	sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
	# Print the articles and their similarity values
	for k, v in sim_sorted:
		if v != 0.0:
			print("Similarity Value:", v)
			print(docs[k])
			print()


#sample queries to run
#Physical activity in patients with chronic obstructive pulmonary disease.
#Acute Hormonal Response to Kettlebell Swing Exercise Differs Depending on Load, Even When Total Work Is Normalized
#Actual and perceived motor competence mediate the relationship between physical fitness and technical skill performance in young soccer players
#Age-related changes in isolated mouse skeletal muscle function are dependent on sex, muscle, and contractility mode

# Add The Query
q1 = ['Actual and perceived motor competence mediate the relationship between physical fitness and technical skill performance in young soccer players']
# Call the function
get_Query(q1, df)