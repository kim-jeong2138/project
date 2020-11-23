from flask import Flask, render_template, request, Response
import datetime
import tensorflow as tf
import numpy as np
import cv2
import sys
from PIL import Image
from flask_uploads import UploadSet, configure_uploads, IMAGES

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model("1109.h5")
class_str=['(0 2)f', '(4 6)f', '(8 12)f', '(15 20)f', '(25 32)f', '(38 43)f', '(48 53)f', '(60 100)f',
           '(0 2)m', '(4 6)m', '(8 12)m', '(15 20)m', '(25 32)m', '(38 43)m', '(48 53)m', '(60 100)m']

def prepare(img_path):
    img= Image.open(img_path)
    img=img.resize((256,256),Image.ANTIALIAS)
    arr=np.array(img)
    return arr.reshape(-1,256,256,3)

@app.route('/')
def index():
    now = datetime.datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    return render_template('home.html', time=timeString)

def gen_frames():

    camera = cv2.VideoCapture(0)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH) 
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    now = datetime.datetime.now()
    timeString = now.strftime("%d_%H-%M-%S")
    

    while True:
        ret, image = camera.read()
        gray = cv2.cvtColor(image, cv2.IMREAD_COLOR)
       
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x,y),(x+w,y+h), (0, 255, 0), 2)

        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()
    cv2.destroyAllWindows()
       
 
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['GET', 'POST'])

def upload():
    now = datetime.datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        img_path= 'static/img/'+ filename
        # return img_path
        predict = model.predict([prepare(img_path)])
        predict_index = np.argmax(predict)
        agd=class_str[predict_index]
        if predict_index==0:
            meg= " 0-2, female" 
        if predict_index==1:
            meg= " 4-6, female"
        if predict_index==2:
            meg= " 8-12, female"
        if predict_index==3:
            meg= " 15-20, female"
        if predict_index==4:
            meg= " 25-32, female"
        if predict_index==5:
            meg= " 38-43, female" 
        if predict_index==6:
            meg= " 48-53, female" 
        if predict_index==7:
            meg= " 60-100, female" 
        if predict_index==8:
            meg= " 0-2, male" 
        if predict_index==9:
            meg= " 4-6, male"
        if predict_index==10:
            meg= " 8-12, male"
        if predict_index==11:
            meg= " 15-20, male"
        if predict_index==12:
            meg= " 25-32, male"
        if predict_index==13:
            meg= " 38-43, male" 
        if predict_index==14:
            meg= " 48-53, male" 
        if predict_index==15:
            meg= " 60-100, male" 

        adv_img = 'static/adv/'+ agd +'.jpg' 
            
    return render_template('home.html', img_path=img_path, adv_img=adv_img, gad=agd, meg=meg, time= timeString)



if __name__ == '__main__':
       app.run(debug=True)
