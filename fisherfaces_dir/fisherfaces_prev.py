
# coding: utf-8

# In[1]:


import numpy as np
import csv
import matplotlib.pyplot as plt
import os, sys
import cv2

from skimage import img_as_float
train_info = []


# In[2]:


ids = []
with open("train_images.csv") as f_train_images:
    i=0
    for line in csv.reader(f_train_images):
        train_info.append(line)
        ids.append(int(line[2]))
        i = i+1


# In[3]:


print train_info[0]


# In[4]:


#n_patches = len(train_info)
n_train_images = 2000
n = 32*32
train_data = np.zeros([n_train_images, n])


# In[5]:


for i in range(n_train_images):
    img = cv2.imread('train_images/%s'%train_info[i][0])
    train_data[i,0:1024] = np.matrix(img_as_float(img[0:32,0:32,1]).flatten())
    


# In[6]:


print train_data.shape


# In[7]:


print train_data.mean(0)


# In[8]:


print train_data.mean(0).shape


# In[9]:


train_data_ms =np.matrix(train_data- np.transpose(train_data.mean(0)))
print train_data_ms.shape
train_data_mean = np.matrix(np.mean(train_data[:,0:1024],axis=0))
cov_matrix = (1.0/(n_train_images-1))*np.transpose(train_data_ms)*train_data_ms


# In[10]:


from numpy import linalg as la
E,P = la.eig(cov_matrix)


# In[11]:


eig_pairs_pca = [(np.abs(E[i]), P[:,i]) for i in range(len(E))]
eig_pairs_pca = sorted(eig_pairs_pca, key=lambda k: k[0], reverse=True)


# In[12]:


print('Variance:\n')
eigv_sum = sum(E)
temp=0
for i,j in enumerate(eig_pairs_pca):
    temp = temp + (j[0]/eigv_sum).real
    if temp > 0.95:
        k=i
        break


# In[13]:


print np.matrix(P[0:30])[0,:]
#train_data_pca = train_data


# In[14]:


train_data_pca = train_data*np.transpose(np.matrix(P[0:k]))
print train_data_pca.shape


# In[15]:


print E
print P
from matplotlib import pyplot as plt
plt.plot(E[0:k])
plt.show()
plt.savefig('Eigen vals.jpg')


# In[16]:


indices = [i for i,x in enumerate(ids[0:2000]) if x==1]
print len(indices)


# In[17]:


np.set_printoptions(precision=4)
num_of_classes = 6

mean_vectors = np.zeros([num_of_classes,k])

for cl in range(num_of_classes):
    
    indices = [i for i,x in enumerate(ids[0:2000]) if x ==cl]
    class_samples = train_data_pca[[indices],:]
    mean_vectors[cl,:] = np.matrix(np.mean(class_samples, axis=1))
   # print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))


# In[18]:


#yt = train_data[[indices],:][0].shape


# In[19]:


S_W = np.zeros((k,k))
for cl in range(num_of_classes):
    indices = [i for i,x in enumerate(ids[0:2000]) if x ==cl]
    print indices
    class_samples = train_data_pca[[indices],:][0]
    print class_samples
    class_sc_mat = np.transpose(np.matrix(class_samples-mean_vectors[cl,:]))*(np.matrix(class_samples-mean_vectors[cl,:]))
    S_W += class_sc_mat            
    print class_sc_mat.shape
    
print('within-class Scatter Matrix:\n', S_W)


#print class_samples_11.shape


# In[20]:


S_B = np.zeros((k,k))
for cl in range(num_of_classes):
    N_cl = len([i for i,x in enumerate(ids[0:2000]) if x ==cl])
    S_B += N_cl*np.transpose(np.matrix(mean_vectors[cl,:]-np.mean(mean_vectors,axis=0)))*np.matrix(mean_vectors[cl,:]-np.mean(mean_vectors,axis=0))
print ('within-class Scatter Matrix:\n', S_B)


# In[21]:


print S_B.shape


# In[22]:


eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))


# In[23]:


eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)


# In[24]:


print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
    print(i[0])


# In[25]:


print('Variance explained:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))


# In[26]:


W = np.hstack((eig_pairs[0][1].reshape(k,1), eig_pairs[1][1].reshape(k,1),
               eig_pairs[2][1].reshape(k,1),eig_pairs[3][1].reshape(k,1),
               eig_pairs[4][1].reshape(k,1)))
print('Matrix W:\n', W.real)


# In[27]:


yt=train_data_pca*W
print yt


# In[ ]:


#Eigen db
encoding_mat = np.transpose(np.matrix(P[0:k]))*W.real
transformed_data = train_data_pca*W.real


# In[ ]:


ids_test = []
test_info = []
f_t=open('val_rsults.csv','a') 
f_t.write('image_name,label,predicted_label')
with open("validate_images.csv") as val_images:
    for line in csv.reader(val_images):
        img = cv2.imread('val_images/%s'%line[0])
        test_image = np.matrix(img_as_float(img[0:32,0:32,1]).flatten())
        test_trans_image = (test_image-train_data_mean)*encoding_mat
        d=[]
        for t in range (n_train_images):
            d.append(la.norm(test_trans_image-transformed_data[t,:]))
        line.append(train_info[d.index(min(d))][2])
        f_t.write("%s,%s,%d,%d\n"%(line[0],line[1],int(line[2]),int(line[3])))
            
        test_info.append(line)
    

