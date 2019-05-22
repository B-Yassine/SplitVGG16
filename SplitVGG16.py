from keras.applications import VGG16
import numpy as np
from keras.models import Model
import keras
from keras import layers
from keras.layers import Input

def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx


vgg = VGG16(weights='imagenet')

vggLayers = [layer.name for layer in vgg.layers]

#choose where you wanna split
print(vggLayers)

model1 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_pool').output)
model1.summary()

idx = getLayerIndexByName(vgg, 'block2_pool')

input_shape = vgg.layers[idx].get_input_shape_at(0) # which is here in my case (None, 55, 55, 256)

layer_input = Input(shape=input_shape[1:]) # as keras will add the batch shape

# create the new nodes for each layer in the path
x = layer_input
for layer in vgg.layers[idx:]:
    x = layer(x)

# create the model
model2 = Model(layer_input, x)
model2.summary()
