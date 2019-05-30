
# coding: utf-8

# In[12]:


import math
import numpy as np
import random
from sklearn import manifold
from keras.layers import Input, Dense,Dropout
from keras.models import Model
import matplotlib.pyplot as plt 
import datetime


# In[13]:


input_matrix = 'simulate.txt'
input_label = 'simulate_label.txt'


# In[15]:


dat = np.loadtxt(input_matrix)
X =dat.T
M = np.shape(X)[0]
N = np.shape(X)[1]
print("Cell numbers: " + str(M))
print("Gene numbers: " + str(N))
 
f = open(input_label)
temp = f.read().splitlines()
f.close()
labels=[]
for t in temp:
    labels.append(eval(t))

m = len(set(labels))
print("Cell types: " + str(m))

label_set = []
for n in labels:
    if n not in label_set:
        label_set.append(n)
print(label_set)
        
Y = np.log2(X+1)


# In[20]:


if __name__ == '__main__':
    def DAE_t_SNE(matrix = input_matrix, label = input_label, n1 = N, n2 = N//10, 
                  n3 = max(N//100, 30),test_set_percent = 0.1, encoding_dim = 30,
                  drop = 0.5, activation = 'relu',optimizer = 'adam', loss = 'mse', 
                  epoch = 50, batch_size=10):
        start_time = datetime.datetime.now()   

        X_test = pd.DataFrame(Y).sample(frac=test_set_percent)
        X_train = pd.DataFrame(Y).drop(X_test.index)
        
        input = Input(shape=(n1,))
        input_corrupted = Dropout(drop)(input)

        encoded_1 = Dense(n2, activation = activation)(input_corrupted)
        encoded_2 = Dense(n3, activation = activation)(encoded_1)
        encoder_output = Dense(encoding_dim,activation = activation )(encoded_2)

        decoded_1 = Dense(n3, activation = activation)(encoder_output)
        decoded_2 = Dense(n2, activation = activation)(decoded_1)
        decoder_output = Dense(n1, activation = activation)(decoded_2)
       
        autoencoder = Model(input, decoder_output)
        encoder = Model(input, encoder_output)
        autoencoder.compile(optimizer = optimizer, loss = loss)
        hist = autoencoder.fit(X_train, X_train, epochs = epoch, batch_size = batch_size,
                shuffle=True,  validation_data=(X_test, X_test))
        lowdim = encoder.predict(Y)
        
        def apply_tSNE(X):
            tsne = manifold.TSNE(n_components=2);
            X_tsne = tsne.fit_transform(X);
            return X_tsne
        DAE_tSNE = apply_tSNE(lowdim)
        np.save("DAE-t-SNE_simulate.npy",DAE_tSNE)
        np.savetxt("DAE-t-SNE_simulate.txt",DAE_tSNE) 
        
        finish_time = datetime.datetime.now()
        print("DAE-t-SNE total time taken = "+ str(finish_time - start_time))
    


# In[21]:


DAE_t_SNE()


# In[23]:


DAE_tSNE = np.load("DAE-t-SNE_simulate.npy")
c_basic = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
c_advanced = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 
                     'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 
                     'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 
                     'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta',
                     'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 
                     'darkslateblue', 'darkslategray', 'darkslategrey', 
                     'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 
                     'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
                     'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 
                     'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 
                     'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 
                     'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 
                     'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 
                     'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen',
                     'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 
                     'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 
                     'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 
                     'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 
                     'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 
                     'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
                     'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
                     'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 
                     'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan',
                     'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 
                     'whitesmoke', 'yellow', 'yellowgreen']
c_shuffled= random.sample(c_advanced,len(c_advanced))
c = c_basic + c_shuffled
index = []
for i in range(m):
    index.append([x == label_set[i] for x in labels])
    
for i in range(m):
    plt.scatter(DAE_tSNE[index[i],0], DAE_tSNE[index[i],1],c=c[i],label = label_set[i])
plt.legend(bbox_to_anchor=(1.25,0.35))
plt.title("DAE-t-SNE")
plt.show()
        

