import numpy as np

def generate_data(n_points, noise, percent):
    inner_count = int(n_points*percent)
    outter_count = int(n_points*(1.0-percent))

    inner_angles = np.linspace(0, 2 * np.pi, num=inner_count)
    inner_radii = np.random.uniform(0, 1, size=inner_count)

    outter_angles = np.linspace(0, 2 * np.pi, num=outter_count)
    outter_radii = np.random.uniform(0, 1, size=outter_count)
    
    inner_circle_x = inner_radii * np.cos(inner_angles)
    inner_circle_y = inner_radii * np.sin(inner_angles)
    
    outer_circle_x = (outter_radii + 1.2) * np.cos(outter_angles)
    outer_circle_y = (outter_radii + 1.2) * np.sin(outter_angles)
    
    X = np.vstack((np.column_stack((inner_circle_x, inner_circle_y)), np.column_stack((outer_circle_x, outer_circle_y))))
    y = np.hstack((np.zeros(inner_count), np.ones(outter_count)))
    
    X += np.random.uniform(-noise, noise, X.shape)
    
    return X, y