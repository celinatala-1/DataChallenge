import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import json
import tkinter as tk
from tkinter import messagebox
from tkinter import*
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.style
matplotlib.style.use('ggplot')
import seaborn as sn
import numpy as np


#PROGRAM: I am running a regression that utilizes the RandomForestClassifier from sklearn. 
#Essentially I used the DALI_Data-Anon.json data file and trained the computer to recognize
#certain patterns. Then according to user input, the computer will computer with a certain accuracy
#their gender and phoneType. I also included two other functions, one called "importance" 
#under the "Graph" menu that plots the importance of our x variables. The other "correlation" allows
#users to choose two random variable and it will plot their correlation. 

#The object containing data we need from Json
objects={}

#Opening the Json files and parsing them appropriately 
with open("/Users/celinatala/Desktop/DataChallenge/DALI_Data-Anon.json") as json_file:
    data =json.load(json_file)
    for variable in data:
        for key in variable.keys():
            if key not in objects.keys():
                objects[key]=[]
            objects[key].append(variable[key])

#We have to recode the dependent variable (gender/phonetype) into binary groups in order to run a logistic regression
genders=objects['gender']
i=0
for gender in genders:
    if gender == "Male":
        genders[i] = 1
    else:
        genders[i] = 0
    i+=1

phonetype=objects['phoneType']
for i in range (0, len(phonetype)):
    if phonetype[i] == "iOS":
        phonetype[i] = 1
    else:
        phonetype[i] = 0

# #Changing the year list into int
year = objects['year']
for i in range (0, len(year)):
    year[i] = year[i][1:]

year=list(map(int, year))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#We are starting to run our logistic regression now using the following variables
df = pd.DataFrame(objects, columns = ['gender', 'phoneType', 'year', 'happiness', 'stressed', 'heightInches', 'sleepPerNight', 'affiliated', 'numOfLanguages'])

##since heightInches has a missing value, we will fill it in with the average of the other heights
df['heightInches'].fillna(df['heightInches'].mean(), inplace = True)

#Setting our x and y variables
X = df[['year', 'happiness', 'stressed', 'heightInches', 'sleepPerNight', 'affiliated', 'numOfLanguages']]
y = df['gender']
y2 = df['phoneType']

#Training using 75% of our data (leaving 25% for testing)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
X_train,X_test,y2_train,y2_test = train_test_split(X,y2,test_size=0.25,random_state=0)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

clf2 = RandomForestClassifier(n_estimators=100)
clf2.fit(X_train, y2_train)
y_pred2 = clf2.predict(X_test)

# #The GUI
root = tk.Tk()
label_width = 20

canvas1 = tk.Canvas(root, width = 500, height = 400)
canvas1.pack()

#Only allow integers
def intOnly(s, min, max):
    try:
        int(s) 
        if int(s) < int(min) or int(s) > int(max):
            return False
        return True
    except ValueError:
        return False

#callback functions to check user input
def userinput(input, min, max):
    if intOnly(input, min, max) or input is "":
        return True
    else:
        tk.messagebox.showerror("Error", "Only input numbers or input out of range ")
        return False

reg = root.register(userinput)

#Now we will create the different buttons/entries for user input
#Year
label1 = tk.Label(root, text = '   YEAR: ', width = label_width, anchor = "w")
canvas1.create_window(150, 100, window=label1)

YEARS = [("'19", 19), ("'20", 20), ("'21", 21), ("'22", 22)]

y = tk.IntVar()

width = 250
for text, mode in YEARS:
    b = tk.Radiobutton(root, text = text, variable = y, value = mode)
    canvas1.create_window(width, 100, window=b)
    width+=50

#Happiness Level
label1 = tk.Label(root, text='   HAPPINESS (Out of 5): ', width =label_width, anchor = "w")
canvas1.create_window(150, 130, window=label1)

entry1 = tk.Entry(root, validate = 'key', validatecommand = (reg, '%S', 0, 5))
canvas1.create_window(320, 130, window=entry1)

#Stressed
label2 = tk.Label(root, text='   STRESS (Out of 10): ', width =label_width, anchor = "w")
canvas1.create_window(150, 160, window=label2)

entry2 = tk.Entry(root, validate = 'key', validatecommand = (reg, '%S', 0, 10))
canvas1.create_window(320, 160, window=entry2)

#Height
label3 = tk.Label(root, text='   HEIGHT(in inches): ', width =label_width, anchor = "w")
canvas1.create_window(150, 190, window=label3)

entry3 = tk.Entry(root, validate = 'key', validatecommand = (reg, '%S', 0, 250))
canvas1.create_window(320, 190, window=entry3)

#Sleep
label4 = tk.Label(root, text='   SLEEP(hours per night): ', width =label_width, anchor = "w")
canvas1.create_window(150, 220, window=label4)

entry4 = tk.Entry(root, validate = 'key', validatecommand = (reg, '%S', 0, 24))
canvas1.create_window(320, 220, window=entry4)

#Affiliated
v = tk.IntVar()
v.set(1)
label5 = tk.Label(root, text='   AFFILIATED: ', width =label_width, anchor = "w")
canvas1.create_window(150, 250, window=label5)

radio1 = tk.Radiobutton(root, text = "yes", variable =v, value = 1)
canvas1.create_window(250, 250, window=radio1)

radio2 = tk.Radiobutton(root, text = "no", variable = v, value = 0)
canvas1.create_window(320, 250, window=radio2)

#Languages
label6 = tk.Label(root, text='   LANGUAGES: ', width =label_width, anchor = "w")
canvas1.create_window(150, 280, window=label6)

entry6 = tk.Entry(root, validate = 'key', validatecommand = (reg, '%S', -1, 100))
canvas1.create_window(320, 280, window=entry6)

def predict():
    global year
    year = y.get()
    
    global happy
    happy = float(entry1.get())
    
    global stress 
    stress = float(entry2.get())
    
    global height 
    height = float(entry3.get())
    
    global sleep 
    sleep = float(entry4.get())
    
    global affiliated
    affiliated = v.get()
    
    global languages
    languages = float(entry6.get())
    
    print(languages)
    
    gender = clf.predict([[year, happy, stress, height, sleep, affiliated, languages]])
    if gender == 0:
        g_result = "female"
    else:
        g_result = "male"
    phoneType = clf2.predict([[year, happy, stress, height, sleep, affiliated, languages]])
    if phoneType == 0:
        p_result = "Android"
    else:
        p_result = "iOS"

    label1_prediction = tk.Label(root, text = "You are probably a %s (with an accuracy of %.3f)" %(g_result, metrics.accuracy_score(y_test, y_pred)))
    canvas1.create_window(320, 350, window=label1_prediction)
    
    label2_prediction = tk.Label(root, text = "You are probably have a %s (with an accuracy of %.3f)" %(p_result, metrics.accuracy_score(y2_test, y_pred2)))
    canvas1.create_window(320, 380, window=label2_prediction)
    
button1 = tk.Button(root, text = '  Predict     ', command = predict)
canvas1.create_window(320, 320, window=button1)


def randomCorrelations():
    
    df2 = pd.DataFrame(objects)
    df2['heightInches'].fillna(df['heightInches'].mean(), inplace = True)

    top = Toplevel(root)

    canvas2 = tk.Canvas(top, width = 500, height = 400)
    canvas2.pack()
    
    label1 = tk.Label(top, text = "Choose two variables to see a correlation between them")
    canvas2.create_window(250, 50, window=label1)
    
    # THe options we want to use
    optionsList = list(df2.keys())
    oMenuWidth = len(max(optionsList, key=len))
    x = tk.StringVar()
    x.set(optionsList[0])
    y = tk.StringVar()
    y.set(optionsList[0])
    
    label2 = tk.Label(top, text = "Var1: ")
    label2.config(width = 90)
    canvas2.create_window(190, 100, window=label2)
    xList = tk.OptionMenu(top, x, *optionsList)
    xList.config(width = oMenuWidth)
    canvas2.create_window(350, 100, window=xList)
    
    label3 = tk.Label(top, text = "Var2: ")
    canvas2.create_window(190, 125, window=label3)
    yList = tk.OptionMenu(top, y, *optionsList)
    yList.config(width = oMenuWidth)
    canvas2.create_window(350, 125, window=yList)
    
    button = tk.Button(top, text = "Correlate", command = lambda: Correlate(df2[x.get()], df2[y.get()], x.get(), y.get()))
    canvas2.create_window(300, 200, window=button)
    
    top.mainloop()
    
def Correlate(x, y, xAxis, yAxis):
    r = np.corrcoef(x, y)
    plt.scatter(x, y)
    plt.title("A random correlation between two variables")
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.show()

        
menubar = Menu(root)
graphmenu = Menu(menubar, tearoff=0)
graphmenu.add_command(label = "Correlations ", command = randomCorrelations)
menubar.add_cascade(label = "graphs", menu=graphmenu)
root.config(menu=menubar)
        
root.mainloop()

    
            