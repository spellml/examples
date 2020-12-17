from keras.applications.resnet50 import ResNet50

model = ResNet50(weights="imagenet")
model.save("model.h5")
