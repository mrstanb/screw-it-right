import math

def compute_max_influential_force(alpha: float, k_mod: float = 0.7, gamma_m: float = 1.3,
                                  f_k: float = 80, len_inside_wall: float = 70,
                                  len_outside_wall: float = 10, width_of_screw: float = 1.5) -> float:
    '''
    Computes the maximum allowed influential force of a screw, screwed in a wooden wall
    Note: This function currently assumes a wooden birch wall with specific elasticity, as well as other
        parameters which are taken from the German handbook for civil engineers
            Parameters:
                    alpha (float): The angle at which the screw is screwed into the wall (in degrees)
                    k_mod (float): Parameter for wooden material depending on the deployment conditions
                    gamma_m (float): Threshold for wooden material ensuring the wooden material is not overstrained
                    f_k (float): Flexural strength of the type of the wooden material
                    len_inside_wall (float): The length of the screw inside the wall (in mm)
                    len_outside_wall (float): The length of the screw outside the wall (in mm)
                    width_of_screw (float): The width of the screw (not the screw head width) (in mm)
            Returns:
                    The maximum allowed influential force of the screw upon the wall (in kg.)
    '''

    # Make sure alpha is a positive angle
    if (alpha < 0):
        alpha = convert_negative_degrees_to_positive(alpha)

    # Calculate f_d which is needed for wooden material and requires the wooden material coefficients,
    # such as k_mod, gamma_m and f_k
    f_d = (k_mod / gamma_m) * f_k

    # q is needed for computing the net force
    q = f_d * width_of_screw

    # The net force is needed for influential force (and for the resistant torque)
    net_force = (1 / 2) * len_inside_wall * q * math.cos(alpha)

    '''
    The resistant torque (R_T) is computed as follows:
        net_force * (2 / 3) * len_inside_wall * math.cos(alpha)

    The influential torque (I_T) (which the screw is exercising) is computed as follows:
        influential_force * len_outside_wall * math.cos(alpha)

    For R_T and I_T, balance must hold, i.e.: R_T = I_T
    Else, if I_T > R_T, then the influential force is too strong and the wall would be damaged
    Hence, the maximum allowed influential force is the one which allows for R_T = I_T to hold
    Any values above this maximum allowed value would impose potential damage to the wall

    After some algebraic simplifications, one could compute the maximum allowed influential force as follows:
        max_influential_force = (net_force * (2 / 3) * len_inside_wall) / len_outside_wall
    '''

    # Compute the maximum allowed influential force
    max_influential_force = (net_force * (2 / 3) * len_inside_wall) / len_outside_wall

    # Make sure to convert the force to kg (for easier intuition)
    return convert_newton_to_kg(max_influential_force)

def convert_negative_degrees_to_positive(negative_degrees: float) -> float:
    return 360 + negative_degrees

def convert_newton_to_kg(newton_value: float) -> float:
    return newton_value / 9.81