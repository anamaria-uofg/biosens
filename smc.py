import random
import math
import numpy as np

import sys
basedir = '/Users/anamaria/git/biosens/blob_detection'
sys.path.append(basedir)
from particle import Circle




def generate_particles(number_particles, variance = 0.05, coord = [[]], baseline_value=0, random=False, add_noise = False):

        """Returns a list of Circle objects generated with random x,y values or with given x,y values. """

        particle_list = []

        if random == True:

            for i in range(number_particles):

                particle = Circle()
                particle.set_characteristics(np.random.uniform(0,16),
                                         np.random.uniform(0,16),
                                         np.random.uniform(1,8),
                                         np.random.uniform(1,80))
                particle.generate_image_array(baseline_value)
                if add_noise == True:
                    particle.add_noise_image(var = variance) #var set at 0.05: noise=var*0.5
                particle_list.append(particle)


        else:

            for i in range(number_particles):

                particle = Circle()
                particle.set_characteristics(coord[i][0], coord[i][1],
                                         coord[i][2], coord[i][3])
                particle.generate_image_array(baseline_value)
                if add_noise == True:
                    particle.add_noise_image(var = variance)
                particle_list.append(particle)

        return particle_list


def get_weight_list(particle_list, chip_image_array, variance):

    """Returns a list with the weights of a corresponding Circle object list."""

    weight_list = []

    for i in range(len(particle_list)):

        weight_list.append(particle_list[i].get_weight(chip_image_array,variance))

    #normalize weights vector to sum 1
    weight_list = np.array(weight_list)
    weight_list = np.exp(weight_list-weight_list.max())
    weight_list /= weight_list.sum()

    return weight_list

def get_resampled_particles(resampling_steps, particle_numbers,
                            particle_list, image_array, gauss_variance):

    """Returns the resampled Circle objects and
    the mean and variance of x,y of the Circle object from every resampling step."""

    resampled_items = []
    resampled_means = []
    resampled_vars = []
    resampled_weights = []

    for i in range(resampling_steps):
        resampled_particle_list = []

        weight_list = get_weight_list(particle_list, image_array, gauss_variance)

        resampled_weights.append(weight_list)


        for j in range(particle_numbers):
            random_resampled_object = np.random.choice(particle_list,
                                                       p=weight_list, replace=False)


            new_random_resampled_object = random_resampled_object.add_noise_to_characteristics()

            new_random_resampled_object.generate_image_array()

            resampled_particle_list.append(new_random_resampled_object)

        particle_list = np.array(resampled_particle_list) # fix here



        resampled_means.append(get_mean(particle_list))
        resampled_vars.append(get_var(particle_list))
        resampled_items.append(particle_list)

        # Some useful output to see if convergence
        #print("-------",i,"--------")
        #print(np.array([p.centre_x for p in resampled_items[-1]]).mean())
        #print(np.array([p.centre_y for p in resampled_items[-1]]).mean())
        #print(np.array([p.radius for p in resampled_items[-1]]).mean())
        #print(np.array([p.intensity for p in resampled_items[-1]]).mean())
        #print()


    return resampled_means, resampled_vars, resampled_items, resampled_weights



def get_mean(list_circle_objects):
    """Returns the mean of the x,y coordinates from a list of Circle objects."""
    meanx = []
    meany = []
    meani = []
    meanr = []

    for i in list_circle_objects:
        meanx.append(i.get_x())
        meany.append(i.get_y())
        meanr.append(i.get_r())
        meani.append(i.get_i())

    return np.array(meanx).mean(), np.array(meany).mean(), np.array(meanr).mean(), np.array(meani).mean()

def get_var(list_circle_objects):
    """Returns the variance of the x,y coordinates from a list of Circle objects."""
    varx = []
    vary = []
    vari = []
    varr = []

    for i in list_circle_objects:
        varx.append(i.get_x())
        vary.append(i.get_y())
        varr.append(i.get_r())
        vari.append(i.get_i())

    return np.array(varx).var(), np.array(vary).var(), np.array(varr).var(), np.array(vari).var()
