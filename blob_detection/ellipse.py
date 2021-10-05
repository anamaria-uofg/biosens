import numpy as np

import math

from copy import copy, deepcopy

from matplotlib import pyplot as plt



class Ellipse(object):

    """ Circle class creates an image array with a circle of x,y coordinates. """



    def __init__(self):

        """ Create the attributes (x,y coordinates) of the object. """

        self.centre_x = 0.0

        self.centre_y = 0.0

        self.radius_x = 0.0

        self.radius_y = 0.0

        self.intensity = 0.0



        self.image_array = []

        self.image_array_noise = []

        self.__dict__.update()



    def set_characteristics(self, x, y, rx, ry, i):

        """ Set the x and y coordinates. """

        self.centre_x = x

        self.centre_y = y

        self.radius_x = rx

        self.radius_y = ry

        self.intensity = i


    def _copy(self):

        memo = Ellipse()

        memo = deepcopy(self)

        return memo


    def add_noise_to_characteristics(self, mean = 0, var = 0.05):

        """ Add noise to x,y coordinates and radius and intensity. """
        self_copy = self._copy()

        self_copy.centre_x += np.random.normal(mean, var)

        self_copy.centre_y += np.random.normal(mean, var)

        self_copy.radius_x += np.random.normal(mean, var)

        self_copy.radius_y += np.random.normal(mean, var)

        self_copy.intensity += np.random.normal(mean, var)

        return self_copy


    def get_x(self):

        return self.centre_x

    def get_y(self):

        return self.centre_y

    def get_rx(self):

        return self.radius_x

    def get_ry(self):

        return self.radius_y

    def get_i(self):

        return self.intensity



    def generate_image_array(self, baseline_value = 10, steps_in_pixel=9, im_size=16):

        """ Generate the image with the x,y coordinates as the circle's centre. """


        image_vector = []

        sumpropIn = 0.0

        for i in range(im_size):
            for j in range(im_size):
                propIn = 0

                for m in range(steps_in_pixel):
                    for n in range(steps_in_pixel):

                        dx = i + float(m)/steps_in_pixel
                        dy = j + float(n)/steps_in_pixel

                        pdx = dx - self.centre_x
                        pdy = dy - self.centre_y

                        dsq = (self.radius_y**2)*(pdx**2) + (self.radius_x**2)*(pdy**2)

                        if dsq <= (self.radius_x**2)*(self.radius_y**2):
                            propIn += 1

                propIn = float(propIn)/(steps_in_pixel*steps_in_pixel)*1

                image_vector.append(baseline_value+propIn*self.intensity)

                sumpropIn += propIn





        self.image_array = np.array(image_vector).reshape(im_size,im_size)





    def _sum_chunk(self, x, chunk_size, axis=-1):

        shape = x.shape
        if axis < 0:
            axis += x.ndim
        shape = shape[:axis] + (-1, chunk_size) + shape[axis+1:]
        x = x.reshape(shape)
        return x.sum(axis=axis+1)


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



    def add_circles(self, circle):

        """ Draw a circle on the image matrix. """

        fig, ax = plt.subplots(1, 1)



        ax.imshow(self.image_array)

        x, y, r = circle

        c = plt.Circle((x, y), r, color='r', linewidth=2, fill=True, alpha=0.2)

        ax.add_patch(c)



        return ax



    def __str__(self):


        return "({},{},{},{})".format(self.centre_x,self.centre_y,self.radius,self.intensity)



    def generate_image_array_withlessforloops(self, baseline_value = 0, steps_in_pixel=9, im_size=16):

        image_vector = []

        sumpropIn = 0.0

        propIn  = baseline_value + self.intensity
        propOut = baseline_value

        for i in range(im_size*steps_in_pixel):

            for j in range(im_size*steps_in_pixel):

                pdx = i - self.centre_x * steps_in_pixel

                pdy = j - self.centre_y * steps_in_pixel

                dsq = np.sqrt(pdx**2 + pdy**2)

                if dsq <= self.radius*steps_in_pixel:

                    image_vector.append(propIn)
                else:
                    image_vector.append(propOut)



        image_array = np.array(image_vector).reshape(im_size*steps_in_pixel,im_size*steps_in_pixel)
        image_array = self._sum_chunk(image_array, steps_in_pixel, axis = 1)
        self.image_array = self._sum_chunk(image_array, steps_in_pixel, axis = 0)
