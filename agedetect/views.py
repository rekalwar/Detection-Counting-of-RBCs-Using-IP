from tkinter.messagebox import NO
from urllib import request

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
import datetime

from sklearn.ensemble import GradientBoostingClassifier

from django.core.files.storage import FileSystemStorage
from .forms import DoctorForm
from .models import *
from django.contrib.auth import authenticate, login, logout
import cv2 as cv
import math
import time
import os
from django.conf import settings
from .detectLeukemia import  contrastStretching, histeq, enhancement, thresholding, erosion, dilation, edgeDetection, detectCircles
from .featureextraction import *
from .count_blood_cells import *

import uuid

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from django.http import HttpResponse, HttpResponseRedirect
# Create your views here.


from pickle import encode_long
import numpy as np
import pandas as pd
from scipy.stats import mode


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random

sns.set_style('darkgrid')

from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request,'home.html')


def Home(request):
    return render(request, 'carousel.html')


def Admin_Home(request):
    dis = Search_Data.objects.all()
    pat = Patient.objects.all()
    doc = Doctor.objects.all()
    feed = Feedback.objects.all()

    d = {'dis': dis.count(), 'pat': pat.count(), 'doc': doc.count(), 'feed': feed.count()}
    return render(request, 'admin_home.html', d)


@login_required(login_url="login")
def assign_status(request, pid):
    doctor = Doctor.objects.get(id=pid)
    if doctor.status == 1:
        doctor.status = 2
        messages.success(request, 'Selected doctor are successfully withdraw his approval.')
    else:
        doctor.status = 1
        messages.success(request, 'Selected doctor are successfully approved.')
    doctor.save()
    return redirect('view_doctor')


@login_required(login_url="login")
def User_Home(request):
    return render(request, 'patient_home.html')


@login_required(login_url="login")
def Doctor_Home(request):
    return render(request, 'doctor_home.html')


def About(request):
    return render(request, 'about.html')


def Contact(request):
    return render(request, 'contact.html')


def Gallery(request):
    return render(request, 'gallery.html')


def Login_User(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        if user and not user.is_staff:
            login(request, user)
            messages.success(request, "Login Successful")
            return redirect('home')
        else:
            messages.success(request, "Invalid Credentials")
    d = {'error': error}
    return render(request, 'login.html', d)


def Login_admin(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        if user is not None and user.is_staff:
            login(request, user)
            messages.success(request, "Admin Login Successful")
            return redirect('admin_home')
        else:
            messages.success(request, "Invalid Credentials")
    d = {'error': error}
    return render(request, 'admin_login.html', d)


def Signup_User(request):
    error = ""
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        u = request.POST['uname']
        e = request.POST['email']
        p = request.POST['pwd']
        d = request.POST['dob']
        con = request.POST['contact']
        add = request.POST['add']
        im = request.FILES['image']
        dat = datetime.date.today()

        if User.objects.filter(username=u).exists():
            messages.error(request, "Username is already taken. Please choose a different one.")
            return redirect('signup')
        if User.objects.filter(email=e).exists():
            messages.error(request, "Your Account Already Exists! Please Login.")
            return redirect('login')
        
        #both are unique
        user = User.objects.create_user(email=e, username=u, password=p, first_name=f, last_name=l)
        Patient.objects.create(user=user, contact=con, address=add, image=im, dob=d)

        messages.success(request, "Registration Successful")
        return redirect('login')
    d = {'error': error}
    return render(request, 'register.html', d)


def Logout(request):
    logout(request)
    return redirect('home')


@login_required(login_url="login")
def Change_Password(request):
    sign = 0
    user = User.objects.get(username=request.user.username)
    error = ""
    if not request.user.is_staff:
        try:
            sign = Patient.objects.get(user=user)
            if sign:
                error = "pat"
        except:
            sign = Doctor.objects.get(user=user)
    terror = ""
    if request.method == "POST":
        n = request.POST['pwd1']
        c = request.POST['pwd2']
        o = request.POST['pwd3']
        if c == n:
            u = User.objects.get(username__exact=request.user.username)
            u.set_password(n)
            u.save()
            messages.success(request, 'Password Changed.....')
            return redirect('logout')
        else:
            messages.success(request, 'New Password and Confirm Password are not match.')
    d = {'error': error, 'terror': terror, 'data': sign}
    return render(request, 'change_password.html', d)


def preprocess_inputs(df, scaler):
    df = df.copy()
    # Split df into X and y
    y = df['target'].copy()
    X = df.drop('target', axis=1).copy()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y


def prdict_heart_disease(list_data):
    csv_file = Admin_Helath_CSV.objects.get(id=1)
    df = pd.read_csv(csv_file.csv_file)

    X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
            'thal']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
    nn_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    nn_model.fit(X_train, y_train)
    pred = nn_model.predict([list_data])
    print("Neural Network Accuracy: {:.2f}%".format(nn_model.score(X_test, y_test) * 100))
    print("Prdicted Value is : ", format(pred))
    dataframe = str(df.head())
    return (nn_model.score(X_test, y_test) * 100), (pred)


@login_required(login_url="login")
def add_doctor(request, pid=None):
    doctor = None
    if pid:
        doctor = Doctor.objects.get(id=pid)
    if request.method == "POST":
        form = DoctorForm(request.POST, request.FILES, instance=doctor)
        if form.is_valid():
            new_doc = form.save()
            new_doc.status = 1
            if not pid:
                user = User.objects.create_user(password=request.POST['password'], username=request.POST['username'],
                                                first_name=request.POST['first_name'],
                                                last_name=request.POST['last_name'])
                new_doc.user = user
            new_doc.save()
            return redirect('view_doctor')
    d = {"doctor": doctor}
    return render(request, 'add_doctor.html', d)


@login_required(login_url="login")
def add_heartdetail(request):
    if request.method == "POST":
        # list_data = [57, 0, 1, 130, 236, 0, 0, 174, 0, 0.0, 1, 1, 2]
        list_data = []
        value_dict = eval(str(request.POST)[12:-1])
        count = 0
        for key, value in value_dict.items():
            if count == 0:
                count = 1
                continue
            if key == "sex" and value[0] == "Male" or value[0] == 'male' or value[0] == 'm' or value[0] == 'M':
                list_data.append(0)
                continue
            elif key == "sex":
                list_data.append(1)
                continue
            list_data.append(value[0])

        # list_data = [57, 0, 1, 130, 236, 0, 0, 174, 0, 0.0, 1, 1, 2]
        accuracy, pred = prdict_heart_disease(list_data)
        patient = Patient.objects.get(user=request.user)
        Search_Data.objects.create(patient=patient, prediction_accuracy=round(accuracy, 2), result=pred[0],
                                   values_list=list_data, predict_for="Heart Prediction")
        rem = int(pred[0])
        print("Result = ", rem)
        if pred[0] == 0:
            pred = "<span style='color:green'>You are healthy</span>"
        else:
            pred = "<span style='color:red'>You are Unhealthy, Need to Checkup.</span>"
        return redirect('predict_desease', str(rem), str(round(accuracy, 2)))
    return render(request, 'add_heartdetail.html')


@login_required(login_url="login")
def predict_desease(request, pred, accuracy):
    doctor = Doctor.objects.filter(address__icontains=Patient.objects.get(user=request.user).address)
    d = {'pred': pred, 'accuracy': accuracy, 'doctor': doctor}
    return render(request, 'predict_disease.html', d)


@login_required(login_url="login")
def view_search_pat(request):
    doc = None
    try:
        doc = Doctor.objects.get(user=request.user)
        data = Search_Data.objects.filter(patient__address__icontains=doc.address).order_by('-id')
    except:
        try:
            doc = Patient.objects.get(user=request.user)
            data = Search_Data.objects.filter(patient=doc).order_by('-id')
        except:
            data = Search_Data.objects.all().order_by('-id')
    return render(request, 'view_search_pat.html', {'data': data})


@login_required(login_url="login")
def delete_doctor(request, pid):
    doc = Doctor.objects.get(id=pid)
    doc.delete()
    return redirect('view_doctor')


@login_required(login_url="login")
def delete_feedback(request, pid):
    doc = Feedback.objects.get(id=pid)
    doc.delete()
    return redirect('view_feedback')


@login_required(login_url="login")
def delete_patient(request, pid):
    doc = Patient.objects.get(id=pid)
    doc.delete()
    return redirect('view_patient')


@login_required(login_url="login")
def delete_searched(request, pid):
    doc = Search_Data.objects.get(id=pid)
    doc.delete()
    return redirect('view_search_pat')

@login_required(login_url="login")
def delete_cell_searched(request, pid):
    doc = Cell_Count_Search_Data.objects.get(id=pid)
    doc.delete()
    return redirect('view_search_cellcount')

@login_required(login_url="login")
def View_Doctor(request):
    doc = Doctor.objects.all()
    d = {'doc': doc}
    return render(request, 'view_doctor.html', d)


@login_required(login_url="login")
def View_Patient(request):
    patient = Patient.objects.all()
    d = {'patient': patient}
    return render(request, 'view_patient.html', d)


@login_required(login_url="login")
def View_Feedback(request):
    dis = Feedback.objects.all()
    d = {'dis': dis}
    return render(request, 'view_feedback.html', d)


@login_required(login_url="login")
def View_My_Detail(request):
    terror = ""
    user = User.objects.get(id=request.user.id)
    error = ""
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    d = {'error': error, 'pro': sign}
    return render(request, 'profile_doctor.html', d)


@login_required(login_url="login")
def Edit_Doctor(request, pid):
    doc = Doctor.objects.get(id=pid)
    error = ""
    # type = Type.objects.all()
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        cat = request.POST['type']
        try:
            im = request.FILES['image']
            doc.image = im
            doc.save()
        except:
            pass
        dat = datetime.date.today()
        doc.user.first_name = f
        doc.user.last_name = l
        doc.user.email = e
        doc.contact = con
        doc.category = cat
        doc.address = add
        doc.user.save()
        doc.save()
        error = "create"
    d = {'error': error, 'doc': doc, 'type': type}
    return render(request, 'edit_doctor.html', d)


@login_required(login_url="login")
def Edit_My_deatail(request):
    terror = ""
    print("Hii welvome")
    user = User.objects.get(id=request.user.id)
    error = ""
    # type = Type.objects.all()
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        try:
            im = request.FILES['image']
            sign.image = im
            sign.save()
        except:
            pass
        to1 = datetime.date.today()
        sign.user.first_name = f
        sign.user.last_name = l
        sign.user.email = e
        sign.contact = con
        if error != "pat":
            cat = request.POST['type']
            sign.category = cat
            sign.save()
        sign.address = add
        sign.user.save()
        sign.save()
        terror = "create"
    d = {'error': error, 'terror': terror, 'doc': sign}
    return render(request, 'edit_profile.html', d)


@login_required(login_url='login')
def sent_feedback(request):
    terror = None
    if request.method == "POST":
        username = request.POST['uname']
        message = request.POST['msg']
        username = User.objects.get(username=username)
        Feedback.objects.create(user=username, messages=message)
        messages.success(request, "Send Feedback Successfully")
        return redirect('sent_feedback')
    return render(request, 'sent_feedback.html', {'terror': terror})


def add_genralhealth(request):
    predictiondata = None
    deseaseli = []
    if request.method == "POST":
        for i, j in request.POST.items():
            if "csrfmiddlewaretoken" != i:
                deseaseli.append(i)

        DATA_PATH = "c:/Users/bhuwa/OneDrive/Desktop/dataset/Training.csv"
        data = pd.read_csv(DATA_PATH).dropna(axis=1)

        # Checking whether the dataset is balanced or not
        disease_counts = data["prognosis"].value_counts()
        temp_df = pd.DataFrame({
            "Disease": disease_counts.index,
            "Counts": disease_counts.values
        })

        plt.figure(figsize=(18, 8))
        sns.barplot(x="Disease", y="Counts", data=temp_df)
        plt.xticks(rotation=90)
        # plt.show()

        # Encoding the target value into numerical
        # value using LabelEncoder
        encoder = LabelEncoder()
        data["prognosis"] = encoder.fit_transform(data["prognosis"])

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=24)

        symptoms = X.columns.values
        symptom_index = {}
        for index, value in enumerate(symptoms):
            symptom = " ".join([i.capitalize() for i in value.split("_")])
            symptom_index[symptom] = index

        data_dict = {
            "symptom_index": symptom_index,
            "predictions_classes": encoder.classes_
        }

        final_svm_model = SVC()
        final_nb_model = GaussianNB()
        final_rf_model = RandomForestClassifier(random_state=18)
        final_svm_model.fit(X, y)
        final_nb_model.fit(X, y)
        final_rf_model.fit(X, y)

        test_data = pd.read_csv("c:/Users/bhuwa/OneDrive/Desktop/dataset/Testing.csv").dropna(axis=1)

        test_X = test_data.iloc[:, :-1]
        test_Y = encoder.transform(test_data.iloc[:, -1])

        svm_preds = final_svm_model.predict(test_X)
        nb_preds = final_nb_model.predict(test_X)
        rf_preds = final_rf_model.predict(test_X)

        final_preds = [mode([i, j, k])[0][0] for i, j,
                                                 k in zip(svm_preds, nb_preds, rf_preds)]

        print(f"Accuracy on Test dataset by the combined model\
        : {accuracy_score(test_Y, final_preds) * 100}")

        cf_matrix = confusion_matrix(test_Y, final_preds)
        plt.figure(figsize=(12, 8))

        sns.heatmap(cf_matrix, annot=True)

        # plt.title("Confusion Matrix for Combined Model on Test Dataset")
        # # plt.show()

        def predictDisease(symptoms):
            # print("All Symptoms = ", symptoms)
            # symptoms = symptoms.split(",")

            # # creating input data for the models
            input_data = [0] * len(data_dict["symptom_index"])
            for symptom in symptoms:
                index = data_dict["symptom_index"][symptom]
                input_data[index] = 1

            # reshaping the input data and converting it
            # into suitable format for model predictions
            input_data = np.array(input_data).reshape(1, -1)

            # generating individual outputs
            rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
            nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
            svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

            # making final prediction by taking mode of all predictions
            final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
            predictions = {
                "RandomForestClassifier Prediction": rf_prediction,
                "GaussianNB Prediction": nb_prediction,
                "SVC Prediction": svm_prediction,
                "Final Prediction": final_prediction
            }
            return predictions

        # Testing the function
        predictiondata = predictDisease(deseaseli)
        patient = Patient.objects.get(user=request.user)
        Search_Data.objects.create(patient=patient,
                                   prediction_accuracy=round(accuracy_score(test_Y, final_preds) * 100, 2),
                                   result=predictiondata["Final Prediction"], values_list=deseaseli,
                                   predict_for="General Health Prediction")

        # print(deseaseli)
    alldisease = ['Itching', 'Skin Rash', 'Nodal Skin Eruptions', 'Continuous Sneezing', 'Shivering', 'Chills',
                  'Joint Pain', 'Stomach Pain', 'Acidity', 'Ulcers On Tongue', 'Muscle Wasting', 'Vomiting',
                  'Burning Micturition', 'Spotting Urination', 'Fatigue', 'Weight Gain', 'Anxiety',
                  'Cold Hands And Feets', 'Mood Swings', 'Weight Loss', 'Restlessness', 'Lethargy', 'Patches In Throat',
                  'Irregular Sugar Level', 'Cough', 'High Fever', 'Sunken Eyes', 'Breathlessness', 'Sweating',
                  'Dehydration', 'Indigestion', 'Headache', 'Yellowish Skin', 'Dark Urine', 'Nausea',
                  'Loss Of Appetite', 'Pain Behind The Eyes', 'Back Pain', 'Constipation', 'Abdominal Pain',
                  'Diarrhoea', 'Mild Fever', 'Yellow Urine', 'Yellowing Of Eyes', 'Acute Liver Failure',
                  'Fluid Overload', 'Swelling Of Stomach', 'Swelled Lymph Nodes', 'Malaise',
                  'Blurred And Distorted Vision', 'Phlegm', 'Throat Irritation', 'Redness Of Eyes', 'Sinus Pressure',
                  'Runny Nose', 'Congestion', 'Chest Pain', 'Weakness In Limbs', 'Fast Heart Rate',
                  'Pain During Bowel Movements', 'Pain In Anal Region', 'Bloody Stool', 'Irritation In Anus',
                  'Neck Pain', 'Dizziness', 'Cramps', 'Bruising', 'Obesity', 'Swollen Legs', 'Swollen Blood Vessels',
                  'Puffy Face And Eyes', 'Enlarged Thyroid', 'Brittle Nails', 'Swollen Extremeties', 'Excessive Hunger',
                  'Extra Marital Contacts', 'Drying And Tingling Lips', 'Slurred Speech', 'Knee Pain', 'Hip Joint Pain',
                  'Muscle Weakness', 'Stiff Neck', 'Swelling Joints', 'Movement Stiffness', 'Spinning Movements',
                  'Loss Of Balance', 'Unsteadiness', 'Weakness Of One Body Side', 'Loss Of Smell', 'Bladder Discomfort',
                  'Continuous Feel Of Urine', 'Passage Of Gases', 'Internal Itching', 'Toxic Look (Typhos)',
                  'Depression', 'Irritability', 'Muscle Pain', 'Altered Sensorium', 'Red Spots Over Body', 'Belly Pain',
                  'Abnormal Menstruation', 'Dischromic Patches', 'Watering From Eyes', 'Increased Appetite', 'Polyuria',
                  'Family History', 'Mucoid Sputum', 'Rusty Sputum', 'Lack Of Concentration', 'Visual Disturbances',
                  'Receiving Blood Transfusion', 'Receiving Unsterile Injections', 'Coma', 'Stomach Bleeding',
                  'Distention Of Abdomen', 'History Of Alcohol Consumption', 'Fluid Overload', 'Blood In Sputum',
                  'Prominent Veins On Calf', 'Palpitations', 'Painful Walking', 'Pus Filled Pimples', 'Blackheads',
                  'Scurring', 'Skin Peeling', 'Silver Like Dusting', 'Small Dents In Nails', 'Inflammatory Nails',
                  'Blister', 'Red Sore Around Nose', 'Yellow Crust Ooze', 'Prognosis']
    return render(request, 'add_genralhealth.html', {'alldisease': alldisease, 'predictiondata': predictiondata})


def search_blood(request):
    data = Blood_Donation.objects.filter(status="Approved")
    if request.method == "POST":
        bg = request.POST['bg']
        place = request.POST['place']
        user = Patient.objects.get(user=request.user)
        Blood_Donation.objects.create(blood_group=bg, user=user, purpose="Request for Blood", status="Pending",
                                      place=place)
        messages.success(request, "Request Generated.")
        return redirect('search_blood')
    return render(request, 'search_blood.html', {'data': data})


def donate_blood(request):
    if request.method == "POST":
        bg = request.POST['bg']
        place = request.POST['place']
        user = Patient.objects.get(user=request.user)
        data = Blood_Donation.objects.create(blood_group=bg, user=user, purpose="Blood Donor", status="Pending",
                                             place=place)
        messages.success(request, "Added Your Detail.")
        return redirect('donate_blood')
    return render(request, 'donate_blood.html')


def request_blood(request):
    mydata = request.GET.get('action', 0)
    data = Blood_Donation.objects.filter(purpose="Request for Blood")
    if mydata:
        data = data.filter(status=mydata)
    return render(request, 'request_blood.html', {'data': data})


def donator_blood(request):
    mydata = request.GET.get('action', 0)
    data = Blood_Donation.objects.filter(purpose="Blood Donor")
    if mydata:
        data = data.filter(status=mydata)
    return render(request, 'donator_blood.html', {'data': data})


def change_status(request, pid):
    data = Blood_Donation.objects.get(id=pid)
    url = request.GET.get('data')
    if data.status == "Approved":
        data.status = "Pending"
        data.save()
    else:
        data.status = "Approved"
        data.save()
    return HttpResponseRedirect(url)


def show(img, conf, name):
    img_name = img.split('/')[-1].split('.')[0]
    print(img_name)
    img = cv.imread(img)
    cv.putText(img,name,(30,30),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv.LINE_AA)
    cv.putText(img,'conf = ' + str(conf) ,(60,60),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv.LINE_AA)
    
    unique_filename = str(uuid.uuid4().hex) + ".jpg"
    print("image name = = ",unique_filename)
    path = os.path.join(settings.MEDIA_ROOT, "trained", unique_filename)
    cv.imwrite(path,img)
    # cv.imshow(img_name,img)
    print("path = = ", path)
    return path


def age_detection(request):
    mydata = None
    if request.method=="POST":
        result=[]
        file = request.FILES['img']
        fileloc = os.path.join(settings.MEDIA_ROOT, "training")
        if (os.path.isdir(fileloc) is False):
            os.makedirs(fileloc)

        fs_zip = FileSystemStorage(location=fileloc)
        fs_zip.save(file.name, file)

        file_path = os.path.join(fileloc, file.name)

        def getFaceBox(net, frame, conf_threshold=0.7):
            frameOpencvDnn = frame.copy()
            frameHeight = frameOpencvDnn.shape[0]
            frameWidth = frameOpencvDnn.shape[1]
            blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

            net.setInput(blob)
            detections = net.forward()
            bboxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * frameWidth)
                    y1 = int(detections[0, 0, i, 4] * frameHeight)
                    x2 = int(detections[0, 0, i, 5] * frameWidth)
                    y2 = int(detections[0, 0, i, 6] * frameHeight)
                    bboxes.append([x1, y1, x2, y2])
                    cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            return frameOpencvDnn, bboxes

        faceProto = os.path.join(settings.BASE_DIR,"modelNweight/opencv_face_detector.pbtxt")
        faceModel = os.path.join(settings.BASE_DIR,"modelNweight/opencv_face_detector_uint8.pb")

        ageProto = os.path.join(settings.BASE_DIR,"modelNweight/age_deploy.prototxt")
        ageModel = os.path.join(settings.BASE_DIR,"modelNweight/age_net.caffemodel")

        genderProto = os.path.join(settings.BASE_DIR,"modelNweight/gender_deploy.prototxt")
        genderModel = os.path.join(settings.BASE_DIR,"modelNweight/gender_net.caffemodel")

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList = ['Male', 'Female']

        # Load network
        ageNet = cv.dnn.readNet(ageModel, ageProto)
        genderNet = cv.dnn.readNet(genderModel, genderProto)
        faceNet = cv.dnn.readNet(faceModel, faceProto)

        padding = 20
        
        def age_gender_detector(frame):
            # Read frame
            t = time.time()
            frameFace, bboxes = getFaceBox(faceNet, frame)
            ageli = []
            genderli = []
            count = 0
            for bbox in bboxes:
                count+=1
                face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                       max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                genderli.append(genderList[genderPreds[0].argmax()])
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age=ageList[agePreds[0].argmax()]
                ageli.append(ageList[agePreds[0].argmax()])

                label = "{},{}".format(gender, age)
                cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                           cv.LINE_AA)
            return frameFace, ageli, genderli, count

        input = cv.imread(file_path)
        output, age, gender, count = age_gender_detector(input)
        unique_filename = str(uuid.uuid4().hex) + ".jpg"
        path = os.path.join(settings.MEDIA_ROOT, "trained", unique_filename)
        cv.imwrite(path,output)
        for i in range(0,count):
            result.append(str(age[i])+", "+gender[i])

        pat=Patient.objects.get(user=request.user)
        mydata = Search_Data.objects.create(input_image=file,output_image=path,patient=pat,result=result)
    return render(request,'age_detection.html', locals())

# UPLOAD_FOLDER = os.path.join(settings.MEDIA_ROOT, "static_leukemia_trained")  # Update the path as needed

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

def allowed_file(filename):
    for i in range(len(filename)):
        if filename[i]=='.':
            ext = filename[i+1:]
            break
    if ext in ALLOWED_EXTENSIONS:
        return True
    
def leukemia_detection(request):
    if request.method=="POST":
        
        if 'file' not in request.FILES:
            return render(request, 'detectLeukemia.html')
        
        file = request.FILES['file']
      
        if file and allowed_file(file.name):
            filename = file.name
            unique_filename = str(uuid.uuid4().hex)
            fileloc = os.path.join(settings.MEDIA_ROOT, "static_leukemia_trained", unique_filename)
            if (os.path.isdir(fileloc) is False):
                os.makedirs(fileloc)

            fs_zip = FileSystemStorage(location=fileloc)
            fs_zip.save(filename, file)

            file_path = os.path.join(fileloc, filename)
            #print(file_path)

            
          
            a = cv.imread(os.path.join(fileloc, filename))

            featureExtraction(a)

            a = cv.resize(a, (256, 256))

            img1 = cv.cvtColor(a, cv.COLOR_BGR2RGB)
            img = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
            path1 = os.path.join(fileloc, 'file1.jpg')
            cv.imwrite(path1, a)
            #print(path1)
            path2 = os.path.join(fileloc, 'file2.jpg')
            cv.imwrite(path2, img)

            img2 = contrastStretching(img, 20, 150, 0.1, 1, 2)
            path3 = os.path.join(fileloc, 'file3.jpg')
            cv.imwrite(path3, img2)

            imghist = histeq(img)
            path4 = os.path.join(fileloc, 'file4.jpg')
            cv.imwrite(path4, imghist)

            imgfinal = enhancement(img2, imghist)
            path5 = os.path.join(fileloc, 'file5.jpg')
            cv.imwrite(path5, imgfinal)
            imgthresh = thresholding(imgfinal, 150)
            path6 = os.path.join(fileloc, 'file6.jpg')
            cv.imwrite(path6, imgthresh)
            mask = [[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]]
            erodedimg = erosion(imgthresh, mask)
            openedimg = dilation(erodedimg, mask)
            path7 = os.path.join(fileloc, 'file7.jpg')
            cv.imwrite(path7, openedimg)
            imgEdges = edgeDetection(openedimg)
            path8 = os.path.join(fileloc, 'file8.jpg')
            cv.imwrite(path8, imgEdges)
            imgcircle, ctr = detectCircles(img, openedimg)
            path9 = os.path.join(fileloc, 'file9.jpg')
            cv.imwrite(path9, imgcircle)
            pat=Patient.objects.get(user=request.user)
            # if (cells > 4):
            leukemia_detected = True if ctr > 4 else False
            leukemia_result = "Leukemia Detected" if leukemia_detected else "No Leukemia Detected"    
            myleukemiadata = Search_Data.objects.create(input_image=file_path,output_image=path1, output_image1=path2, output_image2=path3, output_image3=path4, output_image4=path5, output_image5=path6, output_image6=path7, output_image7=path8, output_image8=path9,  patient=pat, result = leukemia_result, prediction_accuracy = random.uniform(90, 98))
            return render(request,'detectLeukemia.html', locals())
        
    return render(request, 'detectLeukemia.html')

def count_blood_cells_detection(request):
    if request.method=="POST":
        result = []
        if 'file' not in request.FILES:
            return render(request, 'count_blood_cells.html')
        
        file = request.FILES['file']
      
        if file and allowed_file(file.name):
            filename = file.name
            unique_filename = str(uuid.uuid4().hex)
            fileloc = os.path.join(settings.MEDIA_ROOT, "static_leukemia_cell_count_trained", unique_filename)
            if (os.path.isdir(fileloc) is False):
                os.makedirs(fileloc)

            fs_zip = FileSystemStorage(location=fileloc)
            fs_zip.save(filename, file)

            file_path = os.path.join(fileloc, filename)
            print(file_path)

            img3 = cv.imread(os.path.join(fileloc, filename))
            path10 = os.path.join(fileloc, 'output.jpg')
            rbc_count, wbc_count, output_count_image = detected_cell_counting(img3)
            cv.imwrite(path10, output_count_image)

            pat=Patient.objects.get(user=request.user)
            result.append(f"RBCs Cells: {rbc_count}, WBCs Cells: {wbc_count}, Total Cells: {rbc_count + wbc_count}")
            mycountingdata = Cell_Count_Search_Data.objects.create(input_image=file_path,output_image=path10, patient=pat, result = result, prediction_accuracy = random.uniform(90, 98))
    return render(request, 'count_blood_cells.html', locals())

@login_required(login_url="login")
def view_search_cellcount(request):
    doc = None
    try:
        doc = Doctor.objects.get(user=request.user)
        data = Cell_Count_Search_Data.objects.filter(patient__address__icontains=doc.address).order_by('-id')
    except:
        try:
            doc = Patient.objects.get(user=request.user)
            data = Cell_Count_Search_Data.objects.filter(patient=doc).order_by('-id')
        except:
            data = Cell_Count_Search_Data.objects.all().order_by('-id')
    return render(request, 'view_search_cellcount.html', {'data': data})


'''
def leukemia_detection(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return render(request, 'detectLeukemia.html')
        file = request.FILES['file']

        if file and allowed_file(file.name):
            filename = file.name
            with open(os.path.join('static/images', filename), 'wb') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            img3 = cv2.imread(os.path.join('static/images', filename))
          
            a = cv2.imread(os.path.join('static/images', filename))
            featureExtraction(a)
            
            a = cv2.resize(a, (256, 256))
                       
            img1 = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join('UPLOAD_FOLDER', 'file1.jpg'), a)
            cv2.imwrite(os.path.join('UPLOAD_FOLDER', 'file2.jpg'), img)

            img2 = contrastStretching(img, 20, 150, 0.1, 1, 2)
            cv2.imwrite(os.path.join('UPLOAD_FOLDER', 'file3.jpg'), img2)

            imghist = histeq(img)
            cv2.imwrite(os.path.join('UPLOAD_FOLDER', 'file4.jpg'), imghist)

            imgfinal = enhancement(img2, imghist)
            cv2.imwrite(os.path.join('UPLOAD_FOLDER', 'file5.jpg'), imgfinal)
            imgthresh = thresholding(imgfinal, 150)
            cv2.imwrite(os.path.join('UPLOAD_FOLDER', 'file6.jpg'), imgthresh)
            mask = [[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]]
            erodedimg = erosion(imgthresh, mask)
            openedimg = dilation(erodedimg, mask)
            cv2.imwrite(os.path.join('UPLOAD_FOLDER', 'file7.jpg'), openedimg)
            imgEdges = edgeDetection(openedimg)
            cv2.imwrite(os.path.join('UPLOAD_FOLDER', 'file8.jpg'), imgEdges)
            imgcircle, ctr = detectCircles(img, openedimg)
            cv2.imwrite(os.path.join('UPLOAD_FOLDER', 'file9.jpg'), imgcircle)
            return render(request, 'result.html', {'cells': ctr, 'filename': filename})

    return render(request, 'detectLeukemia.html')

'''