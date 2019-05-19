# Abstract Art Using Randomly Initialized Networks

A `PyTorch` implementation of work by [David Ha](http://blog.otoro.net/2015/06/19/neural-network-generative-art/). For writing the code, [Tuan Le's implementation](https://github.com/tuanle618/neural-net-random-art) was immensely helpful.

For an image of required width and height, the artwork is generated pixel by pixel. We initialize a feed-forward network randomly (choosing an appropriate initialization scheme is important as it affects the output drastically), and then pass an dimensional vector to it. 

The components of the vector are the two coordinates of the pixel, distance from the origin and latent random variables, z, as described in [this post](http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/) by Davia Ha. The output of the network is a 3-dimensional vector with each component in the interval (0, 1). The three dimensions denote the values for the Red, Green and Blue channels respectively.  

We experimented with various initialization schemes and following were the results:

**Normal, 3 latent variables**

![Seed 1735192669](/random_init/images/normal-init_6-dim-inp_100.0-scale_1.0-shift_activation-relu_9-hidden_seed-1735192669.jpg){:height="50%" width="50%"}

![Seed 2743082523](/random_init/images/normal-init_6-dim-inp_100.0-scale_1.0-shift_activation-relu_9-hidden_seed-2743082523.jpg){:height="50%" width="50%"}



**Uniform, 3 latent variables**

![Seed 3563121762](/random_init/images/uniform-init_6-dim-inp_100.0-scale_1.0-shift_activation-relu_9-hidden_seed-3563121762.jpg){:height="50%" width="50%"}

![Seed 533904542](/random_init/images/uniform-init_6-dim-inp_100.0-scale_1.0-shift_activation-relu_9-hidden_seed-533904542.jpg){:height="50%" width="50%"}



**Xavier Normal, 3 latent variables**

![Seed 1068043976](/random_init/images/xavier-normal-init_6-dim-inp_100.0-scale_1.0-shift_activation-relu_9-hidden_seed-1068043976.jpg){:height="50%" width="50%"}

![Seed 1165003786](/random_init/images/xavier-normal-init_6-dim-inp_100.0-scale_1.0-shift_activation-relu_9-hidden_seed-1165003786.jpg){:height="50%" width="50%"}



**Xavier Uniform, 3 latent variables**

![Seed 176051726](/random_init/images/xavier-uniform-init_6-dim-inp_100.0-scale_1.0-shift_activation-relu_9-hidden_seed-176051726.jpg){:height="50%" width="50%"}

![Seed 1165003786](/random_init/images/xavier-uniform-init_6-dim-inp_100.0-scale_1.0-shift_activation-relu_9-hidden_seed-1051366071.jpg){:height="50%" width="50%"}



**Kaiming Normal, 3 latent variables**

![Seed 1540315861](/random_init/images/kaiming-normal-init_6-dim-inp_100.0-scale_1.0-shift_activation-relu_9-hidden_seed-1540315861.jpg){:height="50%" width="50%"}

![Seed 567902760](/random_init/images/kaiming-normal-init_6-dim-inp_100.0-scale_1.0-shift_activation-relu_9-hidden_seed-567902760.jpg){:height="50%" width="50%"}



**Kaiming Uniform, 3 latent variables**

![Seed 3780842880](/random_init/images/kaiming-uniform-init_6-dim-inp_100.0-scale_1.0-shift_activation-relu_9-hidden_seed-3780842880.jpg){:height="50%" width="50%"}

![Seed 2933311876](/random_init/images/kaiming-uniform-init_6-dim-inp_100.0-scale_1.0-shift_activation-relu_9-hidden_seed-2933311876.jpg){:height="50%" width="50%"}

