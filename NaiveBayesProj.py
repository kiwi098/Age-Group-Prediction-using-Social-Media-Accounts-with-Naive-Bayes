import pandas
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import GUIProj
from PyQt5 import QtGui
import sys

# Load the CSV file using the pandas.read_csv method
def get_dataset():
    global dataset
    dataset = pandas.read_csv("SocialMediaAccountsSurvey1.csv")
    print(dataset) # Print the content of the CSV file

def encode_dataset():
    global dataset
    encode = LabelEncoder()
    # YES = 1 and NO = 0
    dataset['Facebook'] = encode.fit_transform(dataset['Facebook'])
    dataset['Instagram'] = encode.fit_transform(dataset['Instagram'])
    dataset['Twitter'] = encode.fit_transform(dataset['Twitter'])
    dataset['Tiktok'] = encode.fit_transform(dataset['Tiktok'])
    dataset['Youtube'] = encode.fit_transform(dataset['Youtube'])
    dataset['Spotify/YT Music'] = encode.fit_transform(dataset['Spotify/YT Music'])
    dataset['Shopee/Lazada'] = encode.fit_transform(dataset['Shopee/Lazada'])
    dataset['Tinder/Bumble'] = encode.fit_transform(dataset['Tinder/Bumble'])
    dataset['Discord'] = encode.fit_transform(dataset['Discord'])
    dataset['Skype'] = encode.fit_transform(dataset['Skype'])
    dataset['LinkedIn'] = encode.fit_transform(dataset['LinkedIn'])
    dataset['Pinterest'] = encode.fit_transform(dataset['Pinterest'])
    dataset['Reddit'] = encode.fit_transform(dataset['Reddit'])

    # Teen = 3; Adult = 0; Middle Age Adult = 1; Senior Adult = 2
    dataset['AGE'] = encode.fit_transform(dataset['AGE'])

    print(dataset)
    
def traintestsplit_dataset():
    global dataset
    global ui

    testsize = ui.doublespinbox.value()
    randomstate = ui.spinbox.value()
    
    # Defining the FEATURES and TARGET variables
    features = ["Facebook", "Instagram", "Twitter", "Tiktok", "Youtube", "Spotify/YT Music", "Shopee/Lazada",
                "Tinder/Bumble", "Discord", "Skype", "LinkedIn", "Pinterest", "Reddit"]
    target = "AGE"
    
    global features_train
    global features_test
    global target_train
    global target_test
    
    features_train, features_test, target_train, target_test = train_test_split(
        dataset[features], dataset[target], 
        test_size = testsize, random_state = randomstate)
        
    # Displaying the SPLIT Datasets
    print('\tTRAINING FEATURES\n', features_train)
    print('count: ', len(features_train))
    print('\tTESTING FEATURES\n', features_test)
    print('count: ', len(features_test))
    print('\tTRAINING TARGET\n', target_train)
    print('count: ', len(target_train))
    print('\TESTING TARGET\n', target_test)
    print('count: ', len(target_test))
    ui.label_training_output.setText(str(len(features_train)))
    ui.label_testing_output.setText(str(len(features_test)))
    ui.textedit_terminal_output.setText("FEATURES TRAIN \n" + str(features_train) + "\n FEATURES TEST \n" + str(features_test)
                                        + "\n TARGET TRAIN \n" + str(target_train) + "\n TARGET TEST \n" + str(target_test))

def NaiveBayes_dataset():
    global model
    global ui
    model = CategoricalNB()

    model.fit(features_train.to_numpy(), target_train.to_numpy())
    testpred = model.predict(features_test.to_numpy())
    accuracy = accuracy_score(target_test.to_numpy(), testpred)
    accuracy_output = accuracy*100
    print("\nModel Accuracy = ",accuracy_output,"%")
    ui.label_accuracy_output.setText(str("%.2f" % accuracy_output) + "%")

def input_predict():
    global ui
    facebook = int(ui.checkbox_facebook.isChecked())
    instagram = int(ui.checkbox_instagram.isChecked())
    twitter = int(ui.checkbox_twitter.isChecked())
    tiktok = int(ui.checkbox_tiktok.isChecked())
    youtube = int(ui.checkbox_youtube.isChecked())
    spotify = int(ui.checkbox_spotify_ytmusic.isChecked())
    lazada = int(ui.checkbox_shopee_lazada.isChecked())
    tinder = int(ui.checkbox_tinder_bumble.isChecked())
    discord = int(ui.checkbox_discord.isChecked())
    skype = int(ui.checkbox_skype.isChecked())
    linkedin = int(ui.checkbox_linkedin.isChecked())
    pinterest = int(ui.checkbox_pinterest.isChecked())
    reddit = int(ui.checkbox_reddit.isChecked())

    answer = model.predict([[facebook,instagram,twitter,tiktok,youtube,spotify,lazada
                            ,tinder,discord,skype,linkedin,pinterest,reddit]])

    if answer == 3:
        print("\nTeen")
        ui.label_prediction_picture.setPixmap(QtGui.QPixmap('Teen.png'))
    elif answer == 0:
        print("\nAdult")
        ui.label_prediction_picture.setPixmap(QtGui.QPixmap('Adult.png'))
    elif answer == 1:
        print("\nMiddle Age Adult")
        ui.label_prediction_picture.setPixmap(QtGui.QPixmap('Middle_Age_Adult.png'))
    elif answer == 2:
        print("\nSenior")
        ui.label_prediction_picture.setPixmap(QtGui.QPixmap('Senior_Adult.png'))

def machinelearning_sequence():
    get_dataset()
    encode_dataset()
    traintestsplit_dataset()
    NaiveBayes_dataset()
    input_predict()

def input_clear():
    global ui
    ui.checkbox_facebook.setChecked(0)
    ui.checkbox_instagram.setChecked(0)
    ui.checkbox_twitter.setChecked(0)
    ui.checkbox_tiktok.setChecked(0)
    ui.checkbox_youtube.setChecked(0)
    ui.checkbox_spotify_ytmusic.setChecked(0)
    ui.checkbox_shopee_lazada.setChecked(0)
    ui.checkbox_tinder_bumble.setChecked(0)
    ui.checkbox_discord.setChecked(0)
    ui.checkbox_skype.setChecked(0)
    ui.checkbox_linkedin.setChecked(0)
    ui.checkbox_pinterest.setChecked(0)
    ui.checkbox_reddit.setChecked(0)

    ui.doublespinbox.setValue(0.20)
    ui.spinbox.setValue(17)

    ui.label_prediction_picture.clear()
    ui.textedit_terminal_output.clear()
    ui.label_accuracy_output.clear()

    ui.label_training_output.clear()
    ui.label_testing_output.clear()

    ui.textedit_terminal_output.clear()

app = GUIProj.QtWidgets.QApplication(sys.argv)
MainWindow = GUIProj.QtWidgets.QMainWindow()
ui = GUIProj.Ui_MainWindow()
ui.setupUi(MainWindow)

ui.button_predict.clicked.connect(machinelearning_sequence)
ui.button_clear.clicked.connect(input_clear)

MainWindow.show()
sys.exit(app.exec_())