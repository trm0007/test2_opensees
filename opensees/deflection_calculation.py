def beam_slope_y(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI, P_delta=False):
    """Returns the slope of the elastic curve at any point `x` along the segment (Y-direction).

    :param x: Location (relative to start of segment) where slope is to be calculated.
    :param V1: Internal shear force at start of segment
    :param M1: Internal moment at start of segment  
    :param P1: Internal axial force at start of segment
    :param w1: Distributed load magnitude at start of segment
    :param w2: Distributed load magnitude at end of segment
    :param theta1: Slope at start of segment
    :param delta1: Displacement at start of segment
    :param L: Length of segment
    :param EI: Flexural stiffness of segment
    :param P_delta: Whether P-little-delta effects should be included. Defaults to False.
    :return: The slope of the elastic curve (radians) at location `x`.
    """
    
    if P_delta:
        delta_x = beam_deflection_y(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI, P_delta)
        return theta1 + (-V1*x**2/2 - w1*x**3/6 + x*(-M1 - P1*delta1 + P1*delta_x) + x**4*(w1 - w2)/(24*L))/EI
    else:
        return theta1 + (-V1*x**2/2 - w1*x**3/6 + x*(-M1) + x**4*(w1 - w2)/(24*L))/EI


def beam_slope_z(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI, P_delta=False):
    """Returns the slope of the elastic curve at any point `x` along the segment (Z-direction).

    :param x: Location (relative to start of segment) where slope is to be calculated.
    :param V1: Internal shear force at start of segment
    :param M1: Internal moment at start of segment  
    :param P1: Internal axial force at start of segment
    :param w1: Distributed load magnitude at start of segment
    :param w2: Distributed load magnitude at end of segment
    :param theta1: Slope at start of segment
    :param delta1: Displacement at start of segment
    :param L: Length of segment
    :param EI: Flexural stiffness of segment
    :param P_delta: Whether P-little-delta effects should be included. Defaults to False.
    :return: The slope of the elastic curve (radians) at location `x`.
    """

    if P_delta:
        delta_x = beam_deflection_z(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI, P_delta)
        theta_x = theta1 - (-V1*x**2/2 - w1*x**3/6 + x*(M1 - P1*delta1 + P1*delta_x) + x**4*(w1 - w2)/(24*L))/EI
    else:
        theta_x = theta1 - (-V1*x**2/2 - w1*x**3/6 + x*M1 + x**4*(w1 - w2)/(24*L))/EI

    return theta_x


def beam_deflection_y(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI, P_delta=False):
    """Returns the deflection at a location on the segment (Y-direction).
    
    :param x: Location (relative to start of segment) where deflection is calculated.
    :param V1: Internal shear force at start of segment
    :param M1: Internal moment at start of segment  
    :param P1: Internal axial force at start of segment
    :param w1: Distributed load magnitude at start of segment
    :param w2: Distributed load magnitude at end of segment
    :param theta1: Slope at start of segment
    :param delta1: Displacement at start of segment
    :param L: Length of segment
    :param EI: Flexural stiffness of segment
    :param P_delta: Whether P-little-delta effects should be included. Defaults to False.
    :return: The deflection at location `x`.
    """
    
    # Iteration is required to calculate P-little-delta effects
    if P_delta:
        # Initialize the deflection at `x` to match the deflection at the start of the segment
        delta_x = delta1
        d_delta = 1

        # Iterate until we reach a deflection convergence of 1%
        while d_delta > 0.01: 
            # Save the deflection value from the last iteration
            delta_last = delta_x

            # Compute the deflection
            delta_x = delta1 - theta1*x + V1*x**3/(6*EI) + w1*x**4/(24*EI) - x**2*(-M1 - P1*delta1 + P1*delta_x)/(2*EI) - x**5*(w1 - w2)/(120*EI*L)

            # Check the change in deflection between iterations
            if delta_last != 0:
                d_delta = abs(delta_x/delta_last - 1)
            else:
                # Members with no relative deflection after at least one iteration need no further iterations
                if delta1 - delta_x == 0:
                    break
        
        return delta_x
    
    # Non-P-delta solutions are not iterative
    else:
        return delta1 - theta1*x + V1*x**3/(6*EI) + w1*x**4/(24*EI) - x**2*(-M1)/(2*EI) - x**5*(w1 - w2)/(120*EI*L)


def beam_deflection_z(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI, P_delta=False):
    """Returns the deflection at a location on the segment (Z-direction).
    
    :param x: Location (relative to start of segment) where deflection is calculated.
    :param V1: Internal shear force at start of segment
    :param M1: Internal moment at start of segment  
    :param P1: Internal axial force at start of segment
    :param w1: Distributed load magnitude at start of segment
    :param w2: Distributed load magnitude at end of segment
    :param theta1: Slope at start of segment
    :param delta1: Displacement at start of segment
    :param L: Length of segment
    :param EI: Flexural stiffness of segment
    :param P_delta: Whether P-little-delta effects should be included. Defaults to False.
    :return: The deflection at location `x`.
    """

    # Iteration is required to calculate P-little-delta effects
    if P_delta:
        # Initialize the deflection at `x` to match the deflection at the start of the segment
        delta_x = delta1
        d_delta = 1

        # Iterate until we reach a deflection convergence of 1%
        while d_delta > 0.01:
            # Save the deflection value from the last iteration
            delta_last = delta_x

            # Compute the deflection
            delta_x = delta1 + theta1*x + V1*x**3/(6*EI) + w1*x**4/(24*EI) + x**2*(-M1 + P1*delta1 - P1*delta_x)/(2*EI) + x**5*(-w1 + w2)/(120*EI*L)

            # Check the change in deflection between iterations
            if delta_last != 0:
                d_delta = abs(delta_x/delta_last - 1)
            else:
                # Members with no relative deflection after at least one iteration need no further iterations
                if delta1 - delta_x == 0:
                    break

        return delta_x

    # Non-P-delta solutions are not iterative
    else:
        return delta1 + theta1*x + V1*x**3/(6*EI) + w1*x**4/(24*EI) + x**2*(-M1)/(2*EI) + x**5*(-w1 + w2)/(120*EI*L)


def beam_axial_deflection(x, delta_x1, P1, p1, p2, L, EA):
    """Returns the axial deflection at a location on the segment.
    
    :param x: Location (relative to start of segment) where axial deflection is calculated.
    :param delta_x1: Axial displacement at start of segment
    :param P1: Internal axial force at start of segment
    :param p1: Distributed axial load magnitude at start of segment
    :param p2: Distributed axial load magnitude at end of segment
    :param L: Length of segment
    :param EA: Axial stiffness of segment
    :return: The axial deflection at location `x`.
    """
    
    return delta_x1 - 1/EA*(P1*x + p1*x**2/2 + (p2 - p1)*x**3/(6*L))


# Example usage:
if __name__ == "__main__":
    # Example parameters for a beam segment
    x = 5.0        # Location along segment
    V1 = 1000      # Shear force at start
    M1 = 2000      # Moment at start  
    P1 = 500       # Axial force at start
    w1 = 100       # Distributed load at start
    w2 = 150       # Distributed load at end
    theta1 = 0.01  # Initial slope
    delta1 = 0.05  # Initial deflection
    L = 10.0       # Segment length
    EI = 1e6       # Flexural stiffness
    EA = 1e8       # Axial stiffness
    
    # Calculate deflections and slopes
    deflection_y = beam_deflection_y(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI)
    deflection_z = beam_deflection_z(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI)
    slope_y = beam_slope_y(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI)
    slope_z = beam_slope_z(x, V1, M1, P1, w1, w2, theta1, delta1, L, EI)
    axial_def = beam_axial_deflection(x, delta1, P1, 50, 75, L, EA)
    
    print(f"Y-deflection: {deflection_y:.6f}")
    print(f"Z-deflection: {deflection_z:.6f}")
    print(f"Y-slope: {slope_y:.6f}")
    print(f"Z-slope: {slope_z:.6f}")
    print(f"Axial deflection: {axial_def:.6f}")