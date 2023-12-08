def delete_particle(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted out of bounds at lon = ' + str(particle.lon) + ', lat =' + str(
        particle.lat) + ', depth =' + str(particle.depth))
    particle.delete()

def delete_particle_interp(particle, fieldset, time):
    """Kernel for deleting particles if they are out of bounds."""
    print('particle is deleted due to an interpolation error at lon = ' + str(particle.lon) + ', lat =' + str(
        particle.lat) + ', depth =' + str(particle.depth))
    particle.delete()