import tensorflow as tf

def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
    model.save('cifar10_model.h5')
