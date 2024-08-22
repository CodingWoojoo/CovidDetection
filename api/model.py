import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet201
from sklearn.utils.class_weight import compute_class_weight

def build_resnet50():
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
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)
    
    optimizer = Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_model(X_train_data, Y_train, X_val, Y_val):
    model = build_resnet50()
    
    # Adjust class weights if you have an imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train.argmax(axis=1))
    class_weights = dict(enumerate(class_weights))
    annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=10, verbose=1, min_lr=1e-6)
    checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
    )
    datagen.fit(X_train_data)

    model.fit(
        datagen.flow(X_train_data, Y_train, batch_size=32),
        steps_per_epoch=X_train_data.shape[0] // 32,
        epochs=50,
        verbose=1,
        callbacks=[annealer, checkpoint, early_stopping],
        validation_data=(X_val, Y_val),
        class_weight=class_weights  # Use computed class weights
    )

    return model
