import data_generation.cross as cross
import data_generation.linear as linear
import data_generation.spiral as spriral
import data_generation.xor as xor
import data_generation.circle as circle
import data_generation.linear_circle as linear_circle

def get_dataset(dataset_choice, n_points=200, noise=0.1):
    if dataset_choice == "linear":
        return linear.generate_data(n_points, noise)
    elif dataset_choice == "spiral":
        return spriral.generate_data(n_points, noise)
    elif dataset_choice == "xor":
        return xor.generate_data(n_points, noise)
    elif dataset_choice == "cross":
        return cross.generate_data(n_points, noise)
    elif dataset_choice == "circle":
        return circle.generate_data(n_points, noise, 0.2)
    elif dataset_choice == "linear_circle":
        return linear_circle.generate_data(n_points, noise);
    else:
        raise ValueError("Invalid dataset choice. Please choose from 'linear', 'spiral', 'xor', 'cross'.")