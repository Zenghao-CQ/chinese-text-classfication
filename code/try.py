import numpy as np
import keras
x = np.random.rand(3,3,5)
print(x)
print(x.dtype)
x = keras.preprocessing.sequence.pad_sequences(x,maxlen=6,dtype= x.dtype,
                                            padding='pre',value=0.)
print(x)