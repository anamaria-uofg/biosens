import numpy as np

import math

from copy import copy, deepcopy



class MCircle(object):

    """ MCircle class creates an image array with multiple circles of x,y coordinates. """



    def __init__(self, n):

        """ Create the attributes (x,y coordinates) of the object. """
        self.number_circles = n

        self.centre_x = []

        self.centre_y = []

        self.radius = []

        self.intensity = []


        self.image_array = []

        self.image_array_noise = []



    def set_characteristics(self, x = [], y = [] , r = [] , i = []):

        """ Set the x and y coordinates. """

        for j in range(self.number_circles):

            self.centre_x.append(x[j])

            self.centre_y.append(y[j])

            self.radius.append(r[j])

            self.intensity.append(i[j])


    def _copy(self):

        memo = MCircle(self.number_circles)

        memo = deepcopy(self)

        return memo


    def add_noise_to_characteristics(self, mean = 0, var = 0.05):

        """ Add noise to x,y coordinates and radius and intensity. """
        self_copy = self._copy()

        for i in range(self.number_circles):

            self_copy.centre_x[i] += np.random.normal(mean, var)

            self_copy.centre_y[i] += np.random.normal(mean, var)

            self_copy.radius[i] += np.random.normal(mean, var)

            self_copy.intensity[i] += np.random.normal(mean, var)

        return self_copy


    def get_x(self):

        return self.centre_x

    def get_y(self):

        return self.centre_y

    def get_r(self):

        return self.radius

    def get_i(self):

        return self.intensity



    def generate_image_array(self, baseline_value = 0, steps_in_pixel=9, im_size=16):

        """ Generate the image with the x,y coordinates as the circle's centre. """

        final_image_vector = [0 for x in range(256)]


        sumpropIn = 0.0

        for k in range(self.number_circles):

            image_vector = []
            for i in range(im_size):
                for j in range(im_size):

                    dx = i - self.centre_x[k]
                    dy = j - self.centre_y[k]

                    propIn = 0
                    for m in range(steps_in_pixel):
                        for n in range(steps_in_pixel):
                            pdx = dx + float(m)/steps_in_pixel
                            pdy = dy + float(n)/steps_in_pixel

                            dsq = np.sqrt(pdx**2 + pdy**2)

                            if dsq <= self.radius[k]:

                                propIn += self.intensity[k]

                    propIn = float(propIn)/(steps_in_pixel*steps_in_pixel) + baseline_value/self.number_circles

                    image_vector.append(propIn)

            final_image_vector = [final_image_vector[i]+image_vector[i] for i in range(len(image_vector))]


        self.image_array = np.array(final_image_vector).reshape(im_size,im_size)




    def add_noise_image(self, mean = 0, var = 0.05, im_size=16):

        """ Add noise to the whole image matrix. """

        #make a copy of the matrix first

        self.image_array_noise = deepcopy(self.image_array)

        sigma = var ** 0.5

        self.image_array_noise += np.random.normal(mean,sigma,(im_size, im_size))



    def _get_distance_between_arrays(self,x,y):

        """Calculate the Euclidean distance between 2 matrices. """

        return np.sqrt(np.sum((x-y)**2))



    def _gauss(self, dist, sigma):

        """Calculate the distribution used for calculating the weight of the particles."""

        gauss_prob = (-0.5)*(dist**2)/sigma**2 #/(sigma*np.sqrt(2*math.pi))

        return gauss_prob



    def get_weight(self, chip_image_array, variance):

        """Returns a list with the weights of a corresponding Circle object list."""

        dist_array = self._get_distance_between_arrays(self.image_array, chip_image_array)

        weight = self._gauss(dist_array, variance)

        return weight



    def add_noise_coef_image(self, coef, im_size=16):

        """ Add noise to the whole image matrix. """

        #make a copy of the matrix first

        self.image_array_noise = deepcopy(self.image_array)

        self.image_array_noise += coef





    def add_noise_coef_image2(self, coef, im_size=16):

        #make a copy of the matrix first

        self.image_array_noise = deepcopy(self.image_array)

        self.image_array_noise *= coef



   # def get_circle(centre,r):

   #     """ Another way to draw a circle """

   #     th = np.arange(0,2*math.pi,math.pi/30)

   #     coord = [(round((float(centre[0]) + r *math.cos(item)),1),

   #           round((float(centre[1]) + r *math.sin(item)),1))

   #          for item in th]



   #     return coord



    def add_circles(array, circle):

        """ Draw a circle on the image matrix. """

        fig, ax = plt.subplots(1, 1)



        ax.imshow(array)

        x, y, r = circle

        c = plt.Circle((x, y), r, color='r', linewidth=2, fill=True, alpha=0.2)

        ax.add_patch(c)



        return ax



    def __str__(self):


        return "({},{},{},{})".format(self.centre_x,self.centre_y,self.radius,self.intensity)
