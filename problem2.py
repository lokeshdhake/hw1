"""
Problem 2: kNN Predictions on CIFAR-10

Implement a simple k-nearest neighbors classifier.
The autograder will import and call these functions.
"""

from typing import Tuple, Dict
import numpy as np
import sys
import torchvision
import os

def load_cifar10(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 data using torchvision and return as numpy arrays.
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Download and load the training and test datasets
    #  "return the training and testing data"
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

    # Extract data (images) and targets (labels)
    X_train_raw = train_set.data
    y_train_raw = np.array(train_set.targets, dtype=np.int64)

    X_test_raw = test_set.data
    # --- FIX BELOW: assigned to y_test_raw, not y_train_raw ---
    y_test_raw = np.array(test_set.targets, dtype=np.int64) 

    # Flatten the images: (N, 32, 32, 3) -> (N, 3072)
    # Normalize to [0, 1] by dividing by 255.0
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1).astype(np.float64) / 255.0
    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1).astype(np.float64) / 255.0

    return X_train, y_train_raw, X_test, y_test_raw


def compute_distances(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances between training and test samples.
     "compute pairwise distances between samples"
    """
    # Vectorized L2 Distance: (x-y)^2 = x^2 + y^2 - 2xy
    
    # x^2 (sum over features for test set)
    test_sum_sq = np.sum(np.square(X_test), axis=1, keepdims=True)
    
    # y^2 (sum over features for train set)
    train_sum_sq = np.sum(np.square(X_train), axis=1, keepdims=True).T
    
    # -2xy (dot product)
    # Resulting shape: (n_test, n_train)
    dists_sq = test_sum_sq + train_sum_sq - 2 * np.dot(X_test, X_train.T)
    
    # Clip negative values due to floating point errors and take sqrt
    dists_sq = np.maximum(dists_sq, 0.0)
    
    return np.sqrt(dists_sq)


def predict_knn(
    dists: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Predict labels for test data based on precomputed distances. 
     "return predicted labels for test data"
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test, dtype=np.int64)

    # Find indices of the k smallest distances for each test sample
    # argsort sorts ascending; take first k columns
    closest_indices = np.argsort(dists, axis=1)[:, :k]

    # Retrieve labels for these neighbors
    closest_y = y_train[closest_indices]

    # Majority vote
    for i in range(num_test):
        counts = np.bincount(closest_y[i])
        y_pred[i] = np.argmax(counts)

    return y_pred


def evaluate_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_values: Tuple[int, ...] = (1, 3, 5),
) -> Dict[int, float]:
    """
    Evaluate kNN accuracy for several k values.
    [cite: 50] "compute accuracy for several k values"
    """
    accuracies = {}
    
    # Compute distances once (expensive operation)
    # Using a smaller subset for print demo to avoid huge computation time if needed,
    # but the autograder usually expects full functionality.
    print(f"Computing distances for {X_test.shape[0]} test images...")
    dists = compute_distances(X_train, X_test)
    
    for k in k_values:
        y_pred = predict_knn(dists, y_train, k=k)
        
        # Accuracy = (Correct Predictions) / (Total Predictions)
        num_correct = np.sum(y_pred == y_test)
        accuracy = float(num_correct) / len(y_test)
        
        accuracies[k] = accuracy
        print(f"k={k}: accuracy={accuracy:.4f}")
        
    return accuracies


def main() -> int:
    """
    Minimal demo runner for Problem 2.
    """
    # load_cifar10
    try:
        X_train, y_train, X_test, y_test = load_cifar10("data")
        print(
            f"load_cifar10: X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_test={X_test.shape}, y_test={y_test.shape}"
        )
    except NotImplementedError:
        print("load_cifar10: NotImplemented")
        return 0
    except Exception as e:
        print(f"load_cifar10: error: {e}")
        return 1

    # To speed up the local demo, we can slice the data.
    # The autograder will run on whatever it chooses, but for testing
    # 50k x 10k matrix calculation might take a moment on a laptop.
    # Uncomment lines below to test on a subset:
    # X_train, y_train = X_train[:5000], y_train[:5000]
    # X_test, y_test = X_test[:500], y_test[:500]

    # compute_distances
    try:
        dists = compute_distances(X_train, X_test)
        print(f"compute_distances: dists.shape={dists.shape}")
    except NotImplementedError:
        print("compute_distances: NotImplemented")
        return 0
    except Exception as e:
        print(f"compute_distances: error: {e}")
        return 1

    # predict_knn for a few k values
    for k in (1, 3, 5):
        try:
            y_pred = predict_knn(dists, y_train, k)
            acc = float(np.mean(y_pred == y_test))
            print(f"predict_knn: k={k}, accuracy={acc:.4f}")
        except NotImplementedError:
            print(f"predict_knn: k={k} NotImplemented")
        except Exception as e:
            print(f"predict_knn: k={k} error: {e}")

    # evaluate_accuracy
    try:
        accuracies = evaluate_accuracy(X_train, y_train, X_test, y_test, (1, 3, 5))
        print(f"evaluate_accuracy: {accuracies}")
    except NotImplementedError:
        print("evaluate_accuracy: NotImplemented")
    except Exception as e:
        print(f"evaluate_accuracy: error: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())