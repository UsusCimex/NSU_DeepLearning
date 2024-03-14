import data_generation.cross as cross
import data_generation.linear as linear
import data_generation.spiral as sprial
import data_generation.xor as xor

def get_dataset(dataset_choice, n_points=200, noise=0.1):
    if dataset_choice == "linear":
        return linear.generate_data(n_points, noise)
    elif dataset_choice == "spiral":
        return sprial.generate_data(n_points, noise)
    elif dataset_choice == "xor":
        return xor.generate_data(n_points, noise)
    elif dataset_choice == "cross":
        return cross.generate_data(n_points, noise)
    else:
        raise ValueError("Invalid dataset choice. Please choose from 'linear', 'spiral', 'xor', 'cross'.")