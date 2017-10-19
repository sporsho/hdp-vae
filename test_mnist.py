import numpy as np
import tensorflow as tf
import os.path
import matplotlib.pyplot as plt
from mnist_2 import MNIST
from hdp_vae import HDP_VAE_Model

np.random.seed(0)
tf.set_random_seed(0)
#input output data
data= MNIST()

n_epochs=100 
n_batch=100 
n_latent=20
n_sample= data.x_train.shape[0]
network_arch= dict(n_hidden_encoder_image_1=100,
                   n_hidden_encoder_image_2=100,
                   n_hidden_encoder_sub_1=100,
                   n_hidden_encoder_sub_2=100,
                   n_hidden_encoder_sup_1=100,
                   n_hidden_encoder_sup_2=100,
                   n_hidden_generative_image_1=100,
                   n_hidden_generative_image_2=100,
                   n_hidden_generative_sub_1=100,
                   n_hidden_generative_sub_2=100,
                   n_hidden_generative_sup_1=100,
                   n_hidden_generative_sup_2=100,
                   n_input=784, #32*32*3 images
                   n_input_sub=10, #cifar 100 image has 100 category
                   n_input_sup=2, #cifar 100 images have 20 super category
                   n_z=25, #size of latent space of hierarchy 1
                   n_z_sub=25,
                   n_z_sup=25)

def train(network_arch, learning_rate=0.001, batch_size=100, training_epochs=50):
    hdp_vae= HDP_VAE_Model(network_arch)
    err=[]
    for epoch in range(training_epochs):
        avg_cost=0.
        total_batch= int(n_sample/batch_size)
        for i in range(total_batch):
            batch_data= data.x_train[i*batch_size: (i+1)*batch_size]
            #batch_data=batch_data/256.
            #print(batch_data[0])
            batch_super=data.train_super[i*batch_size: (i+1)*batch_size]
            batch_sub= data.y_train[i*batch_size: (i+1)*batch_size]
            cost= hdp_vae.fit_data(batch_data, batch_super, batch_sub)
            avg_cost+=cost/n_sample*batch_size
            
            #print("Batch: "+str(i)+" Cost : "+str(cost))
        err.append(avg_cost)
        if (epoch+1)%5==0:
         print("Average Cost after Epoch: "+ str(epoch+1)+ "is "+str(avg_cost))
    hdp_vae.save_learned_model()
    return hdp_vae, err

if os.path.isfile("hdpvae.ckpt.index"):
    tf.reset_default_graph()
    trainedNetwork=HDP_VAE_Model(network_arch)
    trainedNetwork.restore()
else:
    trainedNetwork, err= train(network_arch)
    plt.plot(err)
    plt.ylabel('costs')
    plt.show()
x_sample= data.x_test
sup_sample=data.test_super
sub_sample=data.y_test



x_reconstruct, sub_reconstruction, super_reconstruction= trainedNetwork.reconstruct_data(x_sample, sup_sample, sub_sample)
n_test=len(x_sample)
print(n_test)
#visualize latent space
sup= np.zeros((10, 2), dtype=float)
sub= np.zeros((10, 10), dtype=float)
#calculate test errror for inference
MSE_image=0.
MSE_class=0.
MSE_super=0.
print(sub_sample.shape)
for i in range(n_test):
    MSE_image+= sum((x_reconstruct[i]-x_sample[i])*(x_reconstruct[i]-x_sample[i]))/network_arch["n_input"]
    MSE_class+=sum((sub_reconstruction[i]-sub_sample[i])*(sub_reconstruction[i]-sub_sample[i]))/network_arch["n_input_sub"]
    MSE_super+= sum((super_reconstruction[i]-sup_sample[i])*(super_reconstruction[i]-sup_sample[i]))/network_arch["n_input_sup"]
MSE_image=MSE_image/n_test
MSE_class=MSE_class/n_test
MSE_super=MSE_super/n_test

print("Image Reconstruction error: "+str(MSE_image)+ " Class classificaiton Error: "+ str(MSE_class)+ " Super Class classificaiton error: "+str(MSE_super))

lat= trainedNetwork.map_to_latent_space(x_sample, sup, sub)

#print(np.argmax(lat[0]))
for i in range(10):
    plt.plot(lat[i])
plt.ylabel('latent variable')
plt.title("For MNIST dataset")
plt.show()
print(lat.shape)

sup=["odd", "even"]

plt.figure(figsize=(8,12))
plt.subplots_adjust(wspace=0.5, hspace=.5)
for i in range(5):
    print("figure")
    ax1=plt.subplot(5,2,2*i+1)
    
    
    x_data_reformed= x_sample[i]
    x_data_reformed= x_data_reformed.reshape(28,28)
    plt.imshow(x_data_reformed, vmin=0, vmax=1, cmap='gray')
    title= "test super: "+str(sup[np.argmax(sup_sample[i])])+" sub: "+str(np.argmax(sub_sample[i]))
    #plt.title(title)
    plt.text(0.5, 1.08, title,
         horizontalalignment='center',
         fontsize=10,
         transform = ax1.transAxes)
    plt.colorbar()
    ax2=plt.subplot(5,2,2*i+2)
    
    x_data_rr= x_reconstruct[i]
    x_data_rr= x_data_rr.reshape(28,28)
    plt.imshow(x_data_rr, vmin=0, vmax=1, cmap='gray')
    title= "Inference super: "+str(sup[np.argmax(super_reconstruction[i])])+" sub: "+str(np.argmax(sub_reconstruction[i]))
    #plt.title(title)
    plt.text(0.5, 1.08, title,
         horizontalalignment='center',
         fontsize=10,
         transform = ax2.transAxes)
    plt.colorbar()
plt.show()