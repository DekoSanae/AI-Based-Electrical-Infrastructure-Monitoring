#Create generators 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir, val_dir, test_dir):

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        color_mode='rgb'
    )

    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        color_mode='rgb'
    )

    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )

    return train_gen, val_gen, test_gen