from tkinter import *
from tkinter import Label
from tkinter import ttk
from tkinter.ttk import *
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from pandas import DataFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

import tkinter
import nltk
import pickle
import random
import tkinter as tk
import json
import sentiment_mod as s
import matplotlib.pyplot as plt


class Window(Frame):

        def __init__(self, master=None):
        
            Frame.__init__(self, master)
            load = Image.open("img/background.jpg")
            load=load.resize((1100,600),Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)
            img = Label(self, image=render)
            img.image = render
            img.place(x=0, y=0)
            self.master = master
            self.init_window()
            
        def init_window(self):

            self.master.title("Political Affairs Analysis")
            self.pack(fill=BOTH, expand=1)

            T=Text(height=2, width=15)
            T.place(x=720,y=100)
            


            def callback():
                

                def hi():
                    print("hiii")

                win = tk.Toplevel()
                win.title("Fetched tweets")
                win.geometry("1100x600")
                #win.configure(bg='#036CB5')
                lod = Image.open("img/background_main.jpg")
                lod=lod.resize((1400,1000),Image.ANTIALIAS)
                render = ImageTk.PhotoImage(lod)
                imag = Label(win,image=render)
                imag.image = render
                imag.place(x=0, y=0)
                
                
                menubar = Menu(win)
                
                graphmenu=Menu(menubar)
                menubar.add_cascade(label="Graph", menu=graphmenu)
               
                
                graphmenu.add_command(label="Graph1",font = ('Courier', 14), command=hi)
                graphmenu.add_command(label="Graph2",font = ('Courier', 14), command=hi)
                win.config(menu=menubar)

               

                def tr():
                    print("called")
                    s = tk.Scrollbar(win)
                    T1 = tkinter.Text(win, height=150, width=100, font=("Courier", 14))
                    T1.focus_set()
                    s.pack(side=tk.RIGHT, fill=tk.Y)
                    T1.pack(fill=tk.Y)
                    s.config(command=T1.yview)
                    T1.config(yscrollcommand=s.set)
                    enc = 'iso-8859-15'
                    file = open("/text/tweets.txt", 'r', encoding=enc)
                    data = file.read()
                    file.close()
                    print(data)
                    T1.insert(tk.END,data)
                    T1.config(state=DISABLED)
                  
                def showwBar():
                    mGui=Tk()
                    mGui.geometry('470x460')
                    mGui.title('Comparison')

                    featuresetss = open("/pickle_files/featuresets.pickle","rb")
                    featuresets = pickle.load(featuresetss)
                    featuresetss.close()
                    random.shuffle(featuresets)
                    print(len(featuresets))

                    testing_set = featuresets[250:]

                    s = ttk.Style()
                    s.theme_use('clam')

                    ONB = open("/pickle_files/originalnaivebayes5k.pickle","rb")
                    classifier = pickle.load(ONB)
                    ONB.close()

                    ltex = Label(mGui, background="#e6e6e6", font=("Bookman Old Style", 12), text="Original Naive Bayes accuracy percent:")
                    ltex.pack()
                    dab = nltk.classify.accuracy(classifier, testing_set)*100
                    ltex = Label(mGui,background="#e6e6e6",font=("Courier", 12), text=dab)
                    ltex.pack()
                    s.configure("green.Horizontal.TProgressbar", foreground='white', background='green')
                    mpb = ttk.Progressbar(mGui,style="green.Horizontal.TProgressbar",orient ="horizontal",length = 200, mode ="determinate")  
                    mpb.pack()
                    mpb["maximum"] = 100
                    mpb["value"] = dab


                    
                    Logistic = open("/pickle_files/LogisticRegression_classifier5k.pickle","rb")
                    LogisticRegression_classifier = pickle.load(Logistic)
                    Logistic.close()

                    ltex = Label(mGui,background="#e6e6e6",font=("Bookman Old Style", 12), text="LogisticRegression_classifier accuracy percent:")
                    ltex.pack() 
                    dab = nltk.classify.accuracy(LogisticRegression_classifier, testing_set)*100
                    ltex = Label(mGui,background="#e6e6e6",font=("Courier", 12), text=dab)
                    ltex.pack()
                    s.configure("purple.Horizontal.TProgressbar", foreground='white', background='purple')
                    mpb = ttk.Progressbar(mGui,style="purple.Horizontal.TProgressbar",orient ="horizontal",length = 200, mode ="determinate")
                    mpb.pack()
                    mpb["maximum"] = 100
                    mpb["value"] = dab
                    mGui.mainloop()


                def showPie():
                    mGui=Tk()
                    mGui.geometry('650x700')
                    mGui.title('Comparison')
                    count_positive=0
                    count_negative=0
                    count=[]
                    pullData=open("text/sentiment_value.txt","r").read()
                    lines=pullData.split("\n")
                   
                    for l in lines:
                        if "positive" in l:
                            count_positive+=1
                        elif "negative" in l:
                            count_negative+=1
                    count.append(count_positive)
                    count.append(count_negative)
                    print("Count ",count_positive+count_negative)
                    data1={'Opinions':['Positive','Negative'],
                           'No of Responses':[count_positive, count_negative]
                          }
                    print(data1)
                    df=DataFrame(data1,columns=['Opinions','No of Responses'])
                    
                    df=df[['Opinions','No of Responses']].groupby("Opinions").sum()
                    print(df)
                
                    figure=plt.Figure(figsize=(6,5),dpi=100)
                    ax1=figure.add_subplot(111)
                    bar=FigureCanvasTkAgg(figure,mGui)
                    bar.get_tk_widget().pack(side=tk.LEFT,fill=tk.BOTH)
                    df.plot(kind='bar',legend=True,ax=ax1)
                    ax1.set_title("Public opinions")
                    
                    
                
                def showPrediction():
                    
                    mGui=Tk()
                    mGui.geometry('650x700')
                    mGui.title('Prediction')
                    count_positive=0
                    count_negative=0
                    count=[]
                    pullData=open("text/sentiment_value.txt","r").read()
                    lines=pullData.split("\n")
                   
                    for l in lines:
                        if "positive" in l:
                            count_positive+=1
                        elif "negative" in l:
                            count_negative+=1
                    count.append(count_positive)
                    count.append(count_negative)
                    print("Count ",count_positive+count_negative)
                    
                    avg_opinions = (count_positive + count_negative) / 2 
                    
                    data1={'Opinions':['Overall Response'],
                           'No of Responses':[avg_opinions]
                          }
                    print(data1)
                    df=DataFrame(data1,columns=['Opinions','No of Responses'])
                    
                    df=df[['Opinions','No of Responses']].groupby("Opinions").sum()
                    print(df)
                
                    figure=plt.Figure(figsize=(6,5),dpi=100)
                    ax1=figure.add_subplot(111)
                    bar=FigureCanvasTkAgg(figure,mGui)
                    bar.get_tk_widget().pack(side=tk.LEFT,fill=tk.BOTH)
                    df.plot(kind='bar',legend=True,ax=ax1)
                    ax1.set_title("Overall Prediction")


                ip=T.get("1.0","end-1c") 
                B1 = tkinter.Button(win, text ="Show tweets", command=tr)
                B1.place(x = 5, y = 5, height=20, width=100)
                
                B2 = tkinter.Button(win, text ="Show Accuracy", command=showwBar)
                B2.place(x = 5, y = 35, height=20, width=100)
                
                B3 = tkinter.Button(win, text ="Show Graph", command=showPie)
                B3.place(x = 5, y = 55, height=20, width=100)
                
                B4 = tkinter.Button(win, text ="Prediction", command=showPrediction)
                B4.place(x = 5, y = 75, height=20, width=100)
                

                ckey="XXXX"
                csecret="XXXX"
                atoken="XXXX"
                asecret="XXXX"
              
                    
                    
                
                try:
                    class listener(StreamListener):
                        
                        tweet_count = 0
                        
                        def __init__(self, max_tweets):
                            self.max_tweets = max_tweets

                        def on_data(self, data):
      
                            self.tweet_count += 1
                            print(self.tweet_count)
                            print(self.max_tweets)
        
                            if self.tweet_count >= self.max_tweets:
                                return False
                            
                            all_data=json.loads(data)
                            tweet=all_data["text"]
                            char_list = [tweet[j] for j in range(len(tweet)) if ord(tweet[j]) in range(65536)]

                            tweet=''
                            for j in char_list:
                                tweet=tweet+j

                            sentiment_value, confidence = s.sentiment(tweet)
                            if confidence *100>=80:
                                tweetsOutput=open("text/tweets.txt","a")
                                sentimentOutput=open("text/sentiment_value.txt","a")
                                sentimentOutput.write(sentiment_value)
                                sentimentOutput.write("\n")

                                tweetsOutput.write(sentiment_value)
                                tweetsOutput.write("\n")
                                tweetsOutput.write(tweet)
                                tweetsOutput.write("\n\n")
                                tweetsOutput.close()
                                sentimentOutput.close()
                            print (tweet,sentiment_value,confidence)
                            
                            return True

                        def on_error(self, status):
                            print("In status: ",status)

                    auth = OAuthHandler(ckey, csecret)
                    auth.set_access_token(atoken, asecret)

                    twitterStream = Stream(auth, listener(100))
                    twitterStream.filter(track=[ip])

                except:
                    return (True)
            B = tkinter.Button(text ="Submit",command=callback, bg='#66d9ff')
            B.place(x = 850, y = 100, height=40, width=60)

            menu = Menu(self.master)
            self.master.config(menu=menu)


root = Tk()

root.geometry("1100x600")

app = Window(root)

root.mainloop()
