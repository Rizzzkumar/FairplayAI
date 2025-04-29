#!/usr/bin/env python
# coding: utf-8

# In[190]:


get_ipython().system('pip install scikit-learn')


# In[133]:


import sys
print(sys.version)


# In[7]:


get_ipython().system('pip install  opencv-python matplotlib')


# In[144]:


import tensorflow as tf
import os


# In[145]:


gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[146]:


import cv2
import imghdr


# In[147]:


data_dir='C:\\Users\\CHETAN RAJKUMAR\\Desktop\\python_ws\\fairplay_ai\\data'


# In[148]:


image_exts=['jpeg','jpg','bmp','png']


# In[255]:


for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path= os.path.join(data_dir,image_class,image)
        try:
            img=cv2.imread(image_path)
            tip=imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list{}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image{}'.format(image_path))


# In[256]:


import numpy as np
from matplotlib import pyplot as plt


# In[261]:


data=tf.keras.utils.image_dataset_from_directory('C:\\Users\\CHETAN RAJKUMAR\\Desktop\\python_ws\\fairplay_ai\\data')
print(data.class_names)


# In[258]:


data_iterator=data.as_numpy_iterator()


# In[259]:


batch=data_iterator.next()


# In[260]:


fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# In[248]:


data = data.map(lambda x,y: (x/255,y))


# In[156]:


scaled_iterator= data.as_numpy_iterator()


# In[157]:


batch=scaled_iterator.next()


# In[159]:


fig,ax=plt.subplots(ncols=4,figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])


# In[160]:


len(data)


# In[161]:


train_size=int(len(data)*.7)
val_size=int(len(data)*.2)+2
test_size=int(len(data)*.1)


# In[162]:


train= data.take(train_size)
val=data.skip(train_size).take(val_size)
test=data.skip(train_size+val_size).take(test_size)


# In[163]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Dense, Flatten


# In[164]:


model=Sequential()


# In[195]:


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np

# Redefine the model architecture
model = tf.keras.Sequential()

# Adding layers with BatchNormalization and Dropout for regularization
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model before fitting it
model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Early stopping and learning rate scheduler
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))

lr_scheduler = LearningRateScheduler(scheduler)

# Fit the model
hist = model.fit(
    train,
    epochs=20,
    validation_data=val,
    callbacks=[ lr_scheduler, tensorboard_callback]
)

# Visualize training and validation loss
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()


# In[168]:


logdir='logs'


# In[169]:


tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[170]:


hist=model.fit(train,epochs=15,validation_data=val,callbacks=[tensorboard_callback])


# In[171]:


fig=plt.figure()
plt.plot(hist.history['loss'],color='red',label='loss')
plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
fig.suptitle('Loss',fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[1]:


from tensorflow.keras.metrics import Precision, BinaryAccuracy

pre= Precision()
acc=BinaryAccuracy()
# In[211]:


from tensorflow.keras.metrics import Precision, BinaryAccuracy

# Initialize the metrics
pre = Precision()
acc = BinaryAccuracy()

# Iterate through the test dataset and update metrics
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = (model.predict(X) > 0.6).astype(int)   # Predictions, shape (batch_size, 1)

    # Ensure y and yhat shapes are compatible
    if y.ndim == 1:
        y = y.reshape(-1, 1)  # Reshape y to (batch_size, 1) if needed

    # Update metrics with true labels and predictions
    pre.update_state(y, yhat)
  
    acc.update_state(y, yhat)

# Display the final metrics
print("Precision:", pre.result().numpy())
print("Accuracy:", acc.result().numpy())


# In[262]:


img = cv2.imread('C:\\Users\\CHETAN RAJKUMAR\\Desktop\\python_ws\\fairplay_ai\\fbl-eng-pr-fulham-man-city-132630_4150580_20240513091602.jpg')
plt.imshow(img)
plt.show()


# In[263]:


resize=tf.image.resize(img,(256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[222]:


np.expand_dims(resize,0).shape


# In[264]:


yhat=model.predict(np.expand_dims(resize/255,0))


# In[269]:


yhat


# In[270]:


from tensorflow.keras.models import load_model
model.save(os.path.join('models','FairplayAI.h5'))
new_model=load_model(os.path.join('models','FairplayAI.h5'))
yhatnew=new_model.predict(np.expand_dims(resize/255,0))


# In[ ]:




