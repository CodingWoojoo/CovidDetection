from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.densenet import DenseNet201
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def build_densenet():
    densetnet201 = DenseNet201(weights='imagenet', include_top=False)
    input = Input(shape=(224, 224, 3))
    x = Conv2D(3, (3, 3), padding='same')(input)
    x = densetnet201(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    output = Dense(2, activation='softmax', name='root')(x)
    model = Model(input, output)
    optimizer = Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])    
    return model

# Assuming you have `X_train_data`, `Y_train`, `X_val`, and `Y_val` ready
model = build_densenet()
annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.70, patience=10, verbose=1, min_lr=1e-6)
checkpoint = ModelCheckpoint('model.keras', verbose=1, save_best_only=True)
datagen = ImageDataGenerator(rotation_range=20, 
                             width_shift_range=0.2, 
                             height_shift_range=0.2, 
                             horizontal_flip=True) 
datagen.fit(X_train_data)
hist = model.fit(datagen.flow(X_train_data, Y_train, batch_size=32),
                 steps_per_epoch=X_train_data.shape[0] // 32,
                 epochs=50,  # Define your EPOCHS here
                 verbose=1,
                 callbacks=[annealer, checkpoint],
                 validation_data=(X_val, Y_val))

# Save model
model.save('model.h5')

# Evaluate model
final_loss, final_accuracy = model.evaluate(X_val, Y_val)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))
