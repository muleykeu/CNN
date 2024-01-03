#Sys ve Os modülleri ile Numpy kütüphanesi import edildi.
import os
import sys
import numpy as np
#Yapay sinir ağı için Tensorflow ve Keras kütüphaneleri, modüller ve widgetlar import edildi.
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.preprocessing import image
#Grafikleri elde edebilmek için Matplotlib kütüphanesi import edildi.
import matplotlib.pyplot as plt
#Confusion Matrix sklearn kütüphanesinden import edildi.
from sklearn.metrics import classification_report, confusion_matrix
#Arayüz oluşturmak için PyQt5 kütüphanesi import edildi.
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QApplication
#Kameraya erişmek için opencv kütüphanesi import edildi.
import cv2

#Datasetimizin bulunduğu dosya yoluna girildi.
os.chdir('C:\\Users\\Hp\\Desktop\\Dataset')

#Train ve test dosyaları değişkenlere atandı.
train_dir = 'train'
test_dir = 'test'

#Dataset ImageDataGenerator class'ı ile modele eklendi.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.3,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_directory(train_dir,
                                                batch_size=64,
                                                target_size=(48,48),
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(test_dir,
                                                batch_size=64,
                                                target_size=(48,48),
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical')

#Train verisetinin içindeki klasörlerin indisleri istendi.
training_set.class_indices

#Katmanları eklemek için sıralı model oluşturuldu.
model = tf.keras.models.Sequential([
    #32 Feature map ve 3x3'lük filtre ile Convolutional katmanı oluşturuldu. Activation hyper parametresi "relu" olarak belirlendi. 48x48x1 boyutunda siyah beyaz görsel.
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    #BatchNormalization katmanının eklenmesi
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    #Dropout hyperparametresinin değeri 0.25 seçildi.
    tf.keras.layers.Dropout(0.25),

    #kernel_regularizer ile 128, 3*3 Conv2D
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    #Görüntü matrisinin yapay sinir ağında düzleşmesi için Flatten katmanı eklendi.
    tf.keras.layers.Flatten(),
    #Fully Connected görevi gören Dense(1024) katmanı eklendi.
    tf.keras.layers.Dense(1024, activation="relu"),
    #Dropout hyperparametresinin değeri 0.5 seçildi.
    tf.keras.layers.Dropout(0.5),
    #7 düğümlü( verisetindeki class sayısı) softmax aktivasyonlu Dense katmanı eklendi.
    tf.keras.layers.Dense(7, activation="softmax")])

#Modeli derlemek için model.compile() kullanıldı.
model.compile(
    #Learning rate ve decay değerleri belirlenmiş Adam optimizer eklendi.
    optimizer = Adam(lr=0.0001, decay=1e-6),
    #Crossentropy kaybı hesaplandı.
    loss="categorical_crossentropy",
    #Doğruluk metriği olarak Accuracy seçildi.
    metrics=["accuracy"],
)

#Model analiz edilerek özetlendi.
model.summary()

#Checkpoint için ağırlıkların kaydedileceği dosya yolu belirlendi.
chk_path = 'duyguTespit.h5'

#ModelCheckpoint callback'i oluşturuldu.
#Checkpointler, dosya adındaki epoch numarası ve doğrulama kaybıyla birlikte kaydedilir. Kaydedileceği dosya chk_path değişkeni ile seçildi.
checkpoint = ModelCheckpoint(filepath=chk_path,
                             #Sadece model "en iyi" olarak kabul edildiğinde kaydeder.
                             save_best_only=True,
                             #verbose=1 seçilerek ayrıntılar yazdırıldı.
                             verbose=1,
                             #Save_best_only = True olduğu için mevcut kaydetme dosyasının üzerine yazma kararı, minimuma indirilmesine bağlı olarak verilir. Performans ölçüsü olarak 'val_loss' değeri kullanıldığı için 'min' seçildi.
                             mode='min',
                             #Performans ölçüsü olarak 'val_loss' değeri kullanıldı.
                             monitor='val_loss')
#ReduceLROnPlateau callback'i oluşturuldu.
#Performans ölçüsü olarak 'val_loss' değeri kullanıldı.
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              #Öğrenme oranının (learning rate) azaltılacağı faktördür. 0.2 deçilmiştir.
                              factor=0.2,
                              #İyileştirmenin olmadığı eopch sayısı. Değişimin olmadığı 6. epoch'tan sonra eğitim durdurulur.
                              patience=6,
                              #verbose=1 seçilerek ayrıntılar yazdırıldı.
                              verbose=1,
                              #Yalnızca önemli değişikliklere odaklanmak için min_delta değeri
                              min_delta=0.0001)
#Oluşturulan Callbackler callback listesine atandı.
callbacks = [checkpoint, reduce_lr]

#Epoch değeri 50 seçilerek model eğitildi.
steps_per_epoch = training_set.n // training_set.batch_size
validation_steps = test_set.n // test_set.batch_size

history = model.fit(x=training_set,
                 validation_data=test_set,
                 epochs=50,
                 callbacks=callbacks,
                 steps_per_epoch=steps_per_epoch,
                 validation_steps=validation_steps)

#accuracy ve loss grafiklerini elde etmek için matplotlib kütüphanesi kullanıldı. 24x8 boyutlu grafikler.
plt.figure(figsize=(24,8))

#Accuracy Grafiği:
plt.subplot(1,2,1)
plt.plot(history.history["val_accuracy"], label="validation_accuracy", linewidth=4)
plt.plot(history.history["accuracy"], label="training_accuracy", linewidth=4)
plt.legend()
plt.title("Accuracy Graphic", fontsize=18)
plt.ylabel('Accuracy', fontsize=15)
plt.xlabel('Epoch', fontsize=15)

#Loss grafiği:
plt.subplot(1,2,2)
plt.plot(history.history["val_loss"], label="validation_loss", linewidth=4)
plt.plot(history.history["loss"], label="training_loss", linewidth=4)
plt.legend()
plt.title("Loss Graphic" ,fontsize=18)
plt.ylabel('Loss', fontsize=15)
plt.xlabel('Epoch', fontsize=15)

#Grafiklerin görüntülerini almak için:
plt.show()

#Loss ve Accuracy değerleri yazdırıldı.
scores = model.evaluate(test_set)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))

#Test için accuracy değeri yüzdelik değerde yazdırıldı.
test_loss, test_accu = model.evaluate(test_set)
print("final validation accuracy = {:.2f}".format(test_accu*100))

#Precision, f1-score ve accuracy için Confusion Matrix ve Classification Report yazdırıldı.
y_pred = model.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)
class_labels = test_set.class_indices
class_labels = {v:k for k,v in class_labels.items()}

cm_test = confusion_matrix(test_set.classes, y_pred)
print('Confusion Matrix')
print(cm_test)
print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(test_set.classes, y_pred, target_names=target_names))

plt.figure(figsize=(8,8))
plt.imshow(cm_test, interpolation='nearest')
plt.colorbar()
tick_mark = np.arange(len(target_names))
_ = plt.xticks(tick_mark, target_names, rotation=90)
_ = plt.yticks(tick_mark, target_names)

#Model test edildi. Test görselinin path'i değişkene atandı.
test_img_path ='test/disgust/PrivateTest_46114477.jpg'

#Görsel load_img ile açıldı.
img_orj = image.load_img(test_img_path)
img = image.load_img(test_img_path, color_mode = "grayscale", target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x /= 255

#Model tahmini için model.predict() kullanıldı.
custom = model.predict(x)
#Görselin hangi class'a ait oldğunu gösteren oranını belirtmek için matplotlib ile grafik oluşturuldu.
objects = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
y_pos = np.arange(len(objects))
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects)
plt.ylabel('Rate')
plt.title('Mood')
plt.show()

x = np.array(x, 'float32')
x = x.reshape([48, 48]);
plt.axis('off')
plt.gray()
plt.imshow(img_orj);

#Model kaydedildi.
model.save('faceExpression.h5')


#İndislere atanan etiketler, emotion_dict adlı dictionary'ye atandı.
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

#Model load_model ile yüklendi.
model = load_model('faceExpression.h5')

#Üzerinde dosya oluşturabilmek için chdir() ile dizin masaüstü olarak seçildi.
os.chdir('C:\\Users\\Hp\\Desktop')
#Program her çalıştığında zaten varolan bir klasör hatasını almamak için exist_ok=true kullanılarak makedirs() ile masaüstünde Captures adında bir klasör oluşturuldu.
os.makedirs('Captures', exist_ok=True)

#Masatüstü bir uygulama haline getirebilmek için QDialog parametreli bir sınıf oluşturuldu.
class masaustuProgram(QDialog):

    def __init__(self):
        QDialog.__init__(self)

        layout = QVBoxLayout()
        #Uygulama hakkında bilgi vermek için iki adet label oluşturuldu.
        label2 = QLabel("Yüz ifadenizi tespit etmek için fotoğraf çekiniz.")
        label = QLabel("Kamera açıldığında fotoğraf çekmek için Space tuşuna, kameradan çıkmak için ESC tuşuna basınız.")
        #Kameraya bağlayan buton oluşturuldu.
        button2 = QPushButton("Kamerayı aç")
        #Uygulamadan çıkmak için bir çıkış butonu oluşturuldu.
        button = QPushButton("Çıkış")

        layout.addWidget(label2)
        layout.addWidget(label)
        layout.addWidget(button2)
        layout.addWidget(button)

        self.setLayout(layout)
        #Açılan ilk pencerenin ismi "Yüz Ifadesi Tespit Etme Programı" yapıldı.
        self.setWindowTitle("Yüz Ifadesi Tespit Etme Programı")
        #id=button self.close ile uygulamayı kapatan bir buton haline getirildi.
        button.clicked.connect(self.close)
        #id=button2, fotoCekme fonksiyonuna bağlandı. Bu sayede bu butona tıklandığında fotoCekme fonksiyonu çağırılır.
        button2.clicked.connect(self.fotoCekme)

    #Kameraya erişip fotoğraf çekebilmek için fotoCekme fonksiyonu oluşturuldu.
    def fotoCekme(self):
        #Bilgisayarın kendi kamerasına erişebilmek için 0 port numarası alınarak cv2.VideoCapture fonksiyonu kullanıldı ve kamera değişkenine atandı.
        kamera = cv2.VideoCapture(0)
        #Açılan ikinci pencerenin ismi "Kamera aciliyor..." yapıldı.
        cv2.namedWindow("Kamera aciliyor...")
        #Terminal'de uygulama ile ilgili bilgiler ekrana yazdırıldı.
        print("Fotograf cekmek icin lutfen Space tusuna basiniz.")
        print("Programi kapatmak icin ESC tusuna basiniz.")

        #Çekilen fotoğrafların sayısını belirtmek için 0'dan başlayan bir sayaç değişkeni tanımlandı.
        foto_sayac = 0

        while True:
            #Kameranın çalışıp çalışmadığını kontrol etmek için ret değişkeni tanımlandı ve kamera, read() ile okundu.
            ret, frame = kamera.read()

            #Frame cv2.cvtcolor kullanılarak gri renge dönüştürüldü ve gray değişkenine atandı.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Görüntülerdeki nesnelerin tanımlanmasını sağlayan Haarcascade sınıfına ait xml dosyası face_cascade değişkenine atandı.
            face_cascade = cv2.CascadeClassifier('C:\\Users\\Hp\\anaconda3\\Lib\\site-packages\\opencv\\data\\haarcascades\\haarcascade_frontalface_default.xml')

            #Daha önce gri renge dönüştürülen ve frame'i yani gray değişkenini içeren ve görüntüdeki yüzleri bulmamızı sağlayan detectMultiScale fonksiyonu faces adlı değişkene atandı.
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            #detectMultiScale fonksiyonunun tespit ettiği yüzlere ait (x,y,w,h) konum bilgisi tutuldu.
            for (x, y, w, h) in faces:
                #cv2.rectangle fonksiyonu ile tutulan koordinat bilgileri dikdörtgen kutu içerisine alındı. Bunun sebebi yüz tespitinin başarısını göstermektir.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                roi_gray = gray[y:y + h, x:x + w]
                #Frame 48,48 şeklinde yeniden boyutlandırılarak genişletilmiş hali cropped_img değişkenine atandı.
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                #cropped_img normalleştirildi.
                cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
                #Çekilen fotoğrafın tahmini için frame'in son hali (cropped_img), model.predict() kullanılarak tahmin edildi.
                prediction = model.predict(cropped_img)
                #cv2.putText() metodu ile görüntüye labellar ve tahmin sonuçları metin olarak eklendi.
                cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

            #Eğer kamera çalışmadıysa "Kamera acilamadi." çıktısı alındı.
            if not ret:
                print("Kamera acilamadi.")
                break
            #cv2.imshow() ile "Kamera" isimli pencere (kamera) çağrıldı.
            cv2.imshow("Kamera", frame)
            k = cv2.waitKey(1)

            #Kamera 32 ASCII kodlu space tuşu ile açıldığında;
            if k % 256 == 32:
                #Fotoğrafları içine kaydedebilmek için dizin Captures olarak seçildi.
                os.chdir('C:\\Users\\Lenovo\\Desktop\\Captures')
                #Fotoğraflar foto_sayac değişkeni arttıkça proje_tasarım_{}'a yazdırılan foto_isim değişkenine atanarak isimlendirildi. Fotoğraflar png formatında alındı.
                foto_isim = "proje_tasarim_{}.png".format(foto_sayac)
                #cv2.imwrite ile fotoğraflar kaydedildi.
                cv2.imwrite(foto_isim, frame)
                #Fotoğraf çekildiğinde Terminal'de "Fotograf cekildi." bilgisi ekrana yazdırıldı.
                print("Fotograf cekildi.")
                #foto_sayac 1 arttırılarak, ikinci fotoğrafın ismini oluşturmak üzere döngüye tekrar girdi.
                foto_sayac += 1

            #Kameradan 27 ASCII kodlu ESC tuşu ile çıkış yapıldığında Terminal'de "Program kapatildi." çıktısı alındı.
            elif k % 256 == 27:
                print("Program kapatildi.")
                break

        #Döngüden çıkıldığında kamera kapatıldı.
        kamera.release()
        cv2.destroyAllWindows()

app = QApplication(sys.argv)
dialog = masaustuProgram()
dialog.show()
app.exec_()