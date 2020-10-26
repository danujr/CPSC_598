import tkinter
from tkinter import *
from transformers import *
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pandas as pd

embedder = SentenceTransformer('bert-base-nli-mean-tokens')


def getDataset():
    df1 = pd.read_csv('Colab_DatasetCosentyx.csv', header=0, encoding='unicode_escape', engine='python',index_col=False)
    corpus = df1["Questions"][1:53].tolist()
    corpus_embeddings = embedder.encode(corpus)
    return corpus_embeddings,corpus



def getResponse(queries, query_embeddings):
    corpus_embeddings,corpus = getDataset()
    closest_n = 5
    l = []
    r = []
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nMost similar sentences in corpus:\n")

        for idx, distance in results[0:closest_n]:
            print(corpus[idx].strip(), "(Score: %.2f)" % ((1 - distance) * 100))
            l.append(corpus[idx].strip())
            #r.append((1 - distance) * 100)
            r.append("(Score: %.2f)" % ((1 - distance) * 100))

    return list(zip(l, r))


def predict_class(msg):
    vector_embedd = embedder.encode(msg)
    return vector_embedd


def chatbot_response(msg):

    query_embeddings = predict_class(msg)
    print(type(query_embeddings))
    print(msg)
    res = getResponse(msg, query_embeddings)
    return res

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        closest_n = 5

        res = chatbot_response(msg)

        #ChatLog.insert(END, "Bot: " + '\n\n'.join(map(str, res)) +'\n\n')
        ChatLog.insert(END, "Bot: "+ '\n\n'.join([str(elem) for elem in res]) + '\n\n')


        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Cosentyx Bot")
base.geometry("550x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="15", width="250", font="Arial",)
ChatLog.insert('insert', "Please ask question about Cosentyx!! \n\n")
ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send)

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x = 535, y=10, height=396)
ChatLog.place(x=6 , y=6, height = 386, width = 530)
EntryBox.place(x=128, y=401, height=90, width=390)
SendButton.place(x=6, y=401, height=90)



base.mainloop()