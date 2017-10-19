###Combination of Hierarchical Dirichlet Process prior and Stick Breaking VAE ###
### No Rights reserved. No guarrantee is provided ###
### Reference: https://arxiv.org/pdf/1605.06197.pdf
### Part of Masters Practikum ###
### Deep Learning for Real World ###
### Faculty of Informatics ###
### Technical University of Munich ###

import tensorflow as tf
import numpy as np
from xavier import xavier_init
import os.path

class HDP_VAE_Model(object):
    def __init__(self, network_arch, activation_func=tf.nn.softplus, learning_rate=0.001, batch_size=10):
        self.network_arch= network_arch
        self.activation_func= activation_func
        self.learning_rate= learning_rate
        self.batch_size=batch_size
        
        #input data placeholder
        self.x= tf.placeholder(tf.float64, [None, self.network_arch["n_input"]])
        self.y_fine=tf.placeholder(tf.float64, [None, self.network_arch["n_input_sub"]])
        self.y_coarse=tf.placeholder(tf.float64, [None, self.network_arch["n_input_sup"]])
        #create vae network
        self._create_network()
        #initialize tf
        init=tf.global_variables_initializer()
        #launch the session
        self.sess= tf.InteractiveSession()
        self.sess.run(init)
    def dirichlet_process(self, alpha, latent_size, beta_k):
        """dirichlet process with stick breaking process"""
        def cond(k, beta_k):
            if k==latent_size:
                return True
            else:
                return False
        def body(k, beta_k):
            beta_k*=tf.cast(tf.distributions.Beta(1.0, alpha).sample((self.batch_size,latent_size)), tf.float64)
            return k+1, beta_k
        k= tf.constant(0)
        #beta_k=tf.distributions.Beta(1.0, alpha).sample((self.batch_size, latent_size))
        stick_num, stick_beta= tf.while_loop(cond, body, loop_vars=[0, beta_k])
        return stick_beta
    
    def _create_network(self):
        network_weights= self._initialize_weights(**self.network_arch)
        network_biases= self._initialize_biases(**self.network_arch)
        
        self.z_mean, self.z_log_sigma_sq= self._encoder_network(network_weights["encoder"], network_biases["encoder"])
        #we will implement HDP in the latent space
        self.z_h1= self.dirichlet_process(1.0, self.network_arch["n_z"], self.z_mean)
        probability_super_class=0.05 # number of super class 20
        probability_sub_class=0.01 # number of subclass 100
        self.z_h2= self.dirichlet_process(probability_sub_class, self.network_arch["n_z_sub"], self.z_h1 )
        self.z_h3= self.dirichlet_process(probability_super_class, self.network_arch["n_z_sup"], self.z_h2 )
        
        
        self.x_reconstruction_mean, self.super_reconstruction, self.sub_reconstruction= self._generative_network(network_weights["generative"], network_biases["generative"])
        
        
        #loss and optimizer
        loss_image= - tf.reduce_sum(self.x*tf.log(1e-10+self.x_reconstruction_mean)+(1-self.x)*tf.log(1e-10+1-self.x_reconstruction_mean) )
        loss_sub= - tf.reduce_sum(self.y_fine*tf.log(1e-10+self.sub_reconstruction)+(1-self.y_fine)*tf.log(1e-10+1-self.sub_reconstruction) )
        loss_super= - tf.reduce_sum(self.y_coarse*tf.log(1e-10+self.super_reconstruction)+(1-self.y_coarse)*tf.log(1e-10+1-self.super_reconstruction) )
        #loss_image= tf.reduce_mean(tf.square(self.x_reconstruction_mean- self.x))
        #loss_sub= tf.reduce_mean(tf.square(self.sub_reconstruction-self.y_fine))
        #loss_super= tf.reduce_mean(tf.square(self.super_reconstruction-self.y_coarse))
        KL= -0.5*tf.reduce_sum(1+self.z_log_sigma_sq-tf.square(self.z_mean)-tf.exp(self.z_log_sigma_sq),1)
        #self.totalLoss= tf.reduce_mean(tf.add(loss_image, tf.add(loss_sub, loss_super)))
        self.totalLoss=loss_image+loss_sub+loss_super+tf.reduce_mean(KL)
        
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.totalLoss)
        
    def _encoder_network(self, weights, biases):
        layer_1_h1= self.activation_func(tf.add(tf.matmul(self.x, weights['w1h1']),biases['b1h1']))
        layer_2_h1= self.activation_func(tf.add(tf.matmul(layer_1_h1, weights['w2h1']), biases['b2h1']))
        z_mean_h1=tf.add(tf.matmul(layer_2_h1, weights['out_mean_h1']), biases['out_mean_h1'])
        z_log_sigma_sq_h1=tf.add(tf.matmul(layer_2_h1, weights['out_log_sigma_h1']), biases['out_log_sigma_h1'])
        
        #layer_1_h2= self.activation_func(tf.add(tf.matmul(self.y_fine, weights['w1h2']),biases['b1h2']))
        #layer_2_h2= self.activation_func(tf.add(tf.matmul(layer_1_h1, weights['w2h2']), biases['b2h2']))
        #z_mean_h2=tf.add(tf.matmul(layer_2_h2, weights['out_mean_h2']), biases['out_mean_h2'])
        #z_log_sigma_sq_h2=tf.add(tf.matmul(layer_2_h2, weights['out_log_sigma_h2']), biases['out_log_sigma_h2'])
        
        #layer_1_h3= self.activation_func(tf.add(tf.matmul(self.y_coarse, weights['w1h3']),biases['b1h3']))
        #layer_2_h3= self.activation_func(tf.add(tf.matmul(layer_1_h3, weights['w2h3']), biases['b2h3']))
        #z_mean_h3=tf.add(tf.matmul(layer_2_h3, weights['out_mean_h3']), biases['out_mean_h3'])
        #z_log_sigma_sq_h3=tf.add(tf.matmul(layer_2_h3, weights['out_log_sigma_h3']), biases['out_log_sigma_h3'])
 
        #z_mean= tf.add(z_mean_h1, tf.add(z_mean_h2, z_mean_h3))
        z_mean= z_mean_h1
        #z_log_sigma_sq= tf.add(z_log_sigma_sq_h1, tf.add(z_log_sigma_sq_h2, z_log_sigma_sq_h3))
        z_log_sigma_sq= z_log_sigma_sq_h1
        return (z_mean, z_log_sigma_sq)
    
    def _generative_network(self, weights, biases):
        layer_1_h1= self.activation_func(tf.add(tf.matmul(self.z_h1, weights['w1h1']), biases['b1h1']))
        layer_2_h1= self.activation_func(tf.add(tf.matmul(layer_1_h1, weights['w2h1']),biases['b2h1']))
        
        layer_1_h2= self.activation_func(tf.add(tf.matmul(self.z_h2, weights['w1h2']),biases['b1h2']))
        layer_2_h2= self.activation_func(tf.add(tf.matmul(layer_1_h2, weights['w2h2']),biases['b2h2']))
        
        layer_1_h3=self.activation_func(tf.add(tf.matmul(self.z_h3, weights['w1h3']),biases['b1h3']))
        layer_2_h3= self.activation_func(tf.add(tf.matmul(layer_1_h3, weights['w2h3']),biases['b2h3'] ))
        
        x_reconstruction_mean= tf.nn.sigmoid(tf.add(tf.matmul(layer_2_h1, weights['out_mean_h1']),biases['out_mean_h1']))
        
        sub_reconstruction=tf.nn.sigmoid(tf.add(tf.matmul(layer_2_h2, weights['out_mean_h2']),biases['out_mean_h2']))
        
        super_reconstruction=tf.nn.sigmoid(tf.add(tf.matmul(layer_2_h3, weights['out_mean_h3']),biases['out_mean_h3']))
        
        return (x_reconstruction_mean, super_reconstruction, sub_reconstruction)
    
    def _initialize_weights(self, n_hidden_encoder_image_1=100,
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
                   n_input=3072, #32*32*3 images
                   n_input_sub=100, #cifar 100 image has 100 category
                   n_input_sup=20, #cifar 100 images have 20 super category
                   n_z=25, #size of latent space of hierarchy 1
                   n_z_sub=25,
                   n_z_sup=25):
        all_weights= dict()
        all_weights["encoder"]={
            'w1h1': tf.Variable(xavier_init(n_input, n_hidden_encoder_image_1)),
            'w2h1': tf.Variable(xavier_init(n_hidden_encoder_image_1, n_hidden_encoder_image_2)),
            'out_mean_h1': tf.Variable(xavier_init(n_hidden_encoder_image_2, n_z)),
            'out_log_sigma_h1': tf.Variable(xavier_init(n_hidden_encoder_image_2, n_z)),
            'w1h2': tf.Variable(xavier_init(n_input_sub, n_hidden_encoder_sub_1)),
            'w2h2': tf.Variable(xavier_init(n_hidden_encoder_sub_1, n_hidden_encoder_sub_2)),
            'out_mean_h2': tf.Variable(xavier_init(n_hidden_encoder_sub_2, n_z_sub)),
            'out_log_sigma_h2': tf.Variable(xavier_init(n_hidden_encoder_sub_2, n_z_sub)),
            'w1h3': tf.Variable(xavier_init(n_input_sup, n_hidden_encoder_sup_1)),
            'w2h3': tf.Variable(xavier_init(n_hidden_encoder_sup_1, n_hidden_encoder_sup_2)),
            'out_mean_h3': tf.Variable(xavier_init(n_hidden_encoder_sup_2, n_z_sup)),
            'out_log_sigma_h3': tf.Variable(xavier_init(n_hidden_encoder_sup_2, n_z_sup))}
        all_weights["generative"]={
            'w1h1': tf.Variable(xavier_init(n_z, n_hidden_generative_image_1)),
            'w2h1': tf.Variable(xavier_init(n_hidden_generative_image_1, n_hidden_generative_image_2)),
            'out_mean_h1': tf.Variable(xavier_init(n_hidden_generative_image_2, n_input)),
            'out_log_sigma_h1': tf.Variable(xavier_init(n_hidden_generative_image_2, n_input)),
            'w1h2': tf.Variable(xavier_init(n_z_sub, n_hidden_generative_sub_1)),
            'w2h2': tf.Variable(xavier_init(n_hidden_generative_sub_1, n_hidden_generative_sub_2)),
            'out_mean_h2': tf.Variable(xavier_init(n_hidden_generative_sub_2, n_input_sub)),
            'out_log_sigma_h2': tf.Variable(xavier_init(n_hidden_generative_sub_2, n_input_sub)),
            'w1h3': tf.Variable(xavier_init(n_z_sup, n_hidden_generative_sup_1)),
            'w2h3': tf.Variable(xavier_init(n_hidden_generative_sup_1, n_hidden_generative_sup_2)),
            'out_mean_h3': tf.Variable(xavier_init(n_hidden_generative_sup_2, n_input_sup)),
            'out_log_sigma_h3': tf.Variable(xavier_init(n_hidden_generative_sup_2, n_input_sup))}
        
        return all_weights
    
    def _initialize_biases(self,  n_hidden_encoder_image_1=100,
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
                   n_input=3072, #32*32*3 images
                   n_input_sub=100, #cifar 100 image has 100 category
                   n_input_sup=20, #cifar 100 images have 20 super category
                   n_z=25, #size of latent space of hierarchy 1
                   n_z_sub=25,
                   n_z_sup=25):
        all_biases= dict()
        all_biases["encoder"]={
            'b1h1': tf.Variable(tf.zeros([n_hidden_encoder_image_1], dtype=tf.float64)),
            'b2h1': tf.Variable(tf.zeros([n_hidden_encoder_image_2], dtype=tf.float64)),
            'out_mean_h1': tf.Variable(tf.zeros([n_z], dtype=tf.float64)),
            'out_log_sigma_h1': tf.Variable(tf.zeros([n_z], dtype=tf.float64)),
            'b1h2': tf.Variable(tf.zeros([n_hidden_encoder_sub_1], dtype=tf.float64)),
            'b2h2': tf.Variable(tf.zeros([n_hidden_encoder_sub_2], dtype=tf.float64)),
            'out_mean_h2': tf.Variable(tf.zeros([n_z_sub], dtype=tf.float64)),
            'out_log_sigma_h2': tf.Variable(tf.zeros([n_z_sub], dtype=tf.float64)),
            'b1h3': tf.Variable(tf.zeros([n_hidden_encoder_sup_1], dtype=tf.float64)),
            'b2h3': tf.Variable(tf.zeros([n_hidden_encoder_sup_2], dtype=tf.float64)),
            'out_mean_h3': tf.Variable(tf.zeros([n_z_sup], dtype=tf.float64)),
            'out_log_sigma_h3': tf.Variable(tf.zeros([n_z_sup], dtype=tf.float64))}
        all_biases["generative"]={
            'b1h1': tf.Variable(tf.zeros([n_hidden_generative_image_1], dtype=tf.float64)),
            'b2h1': tf.Variable(tf.zeros([n_hidden_generative_image_2], dtype=tf.float64)),
            'out_mean_h1': tf.Variable(tf.zeros([n_input], dtype=tf.float64)),
            'out_log_sigma_h1': tf.Variable(tf.zeros([n_input], dtype= tf.float64)),
            'b1h2': tf.Variable(tf.zeros([n_hidden_generative_sub_1], dtype=tf.float64)),
            'b2h2': tf.Variable(tf.zeros([n_hidden_generative_sub_2], dtype=tf.float64)),
            'out_mean_h2': tf.Variable(tf.zeros([n_input_sub], dtype=tf.float64)),
            'out_log_sigma_h2': tf.Variable(tf.zeros([n_input_sub], dtype= tf.float64)),
            'b1h3': tf.Variable(tf.zeros([n_hidden_generative_sup_1], dtype=tf.float64)),
            'b2h3': tf.Variable(tf.zeros([n_hidden_generative_sup_2], dtype=tf.float64)),
            'out_mean_h3': tf.Variable(tf.zeros([n_input_sup], dtype=tf.float64)),
            'out_log_sigma_h3': tf.Variable(tf.zeros([n_input_sup], dtype= tf.float64))}
        return all_biases
    
    def save_learned_model(self):
        saver= tf.train.Saver()
        save_path=saver.save(self.sess, "hdpvae.ckpt")
        print("Model saved in file: %s"% save_path)
    
    def restore(self):
        #tf.reset_default_graph()
        saver=tf.train.Saver()
        saver.restore(self.sess, "hdpvae.ckpt")
        print("Model Restored")
        
    def fit_data(self, x, sup, sub):
        opt, cost= self.sess.run((self.optimizer, self.totalLoss), feed_dict={self.x:x, self.y_fine:sub, self.y_coarse:sup})
        return cost
    def map_to_latent_space(self, x, sup, sub):
        return self.sess.run(self.z_mean, feed_dict={self.x:x, self.y_fine:sub, self.y_coarse:sup})
    def generate_data(self, z_mu=None):
        if z_mu is None:
            z_mu= np.random_normal(size=self.network_arch["n_z"])
        return self.sess.run(self.x_reconstruction_mean, feed_dict={self.z_mean:z_mu})
    def reconstruct_data(self, x, sup, sub):
        return self.sess.run((self.x_reconstruction_mean, self.sub_reconstruction, self.super_reconstruction), feed_dict={self.x:x, self.y_fine:sub, self.y_coarse:sup})
        
        