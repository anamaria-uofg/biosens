Biosens

An algorithm developed for the detection of a "blob" in a 16x16 image array with various degrees of Gaussian noise added to it.
The algorithm is based on Sequential Monte Carlo.

The image arrays were synthetically generated. These are the same as particles. The particles are represented by object _Circle_.
The object's main attributes are the x and y coordinates of the circle (blob) centre (x,y), the radius length (r) and intensity (i)
of the circle.

SMC generative modelling summary

N random particles are generated and each is compared with the image of interest.
Based on their Euclidean distance to the image, each particle is attributed a weight.

N particles are resampled from the original sample, according to their weights.
These are then modified by adding Gaussian noise to it, creating thus a new sample of N particles.
These are again compared with the image of interest by calculating the weights.

Resampling is performed X times.
