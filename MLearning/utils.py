"""
Utility Functions Module

This module provides utility functions for numerical stability,
visualization, logging, and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt


def log_sum_exp(log_probs: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Compute log(sum(exp(log_probs))) in a numerically stable way.

    Args:
        log_probs: Array of log probabilities
        axis: Axis along which to compute (None for all)

    Returns:
        Log of sum of exponentials

    TODO:
    - Find maximum value along axis: max_log_prob
    - Subtract max_log_prob from all values
    - Compute exp, sum, and log
    - Add max_log_prob back
    - This prevents numerical overflow/underflow
    - Return result
    """
    # Find maximum value along the specified axis
    max_log_prob = np.max(log_probs, axis=axis, keepdims=True)

    # Subtract max, compute exp and sum, then log
    result = max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob), axis=axis, keepdims=True))

    # Remove extra dimensions if keepdims was used
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    else:
        result = np.squeeze(result)

    return result


def normalize_log_probs(log_probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Normalize log probabilities to sum to 1.

    Args:
        log_probs: Array of log probabilities
        axis: Axis along which to normalize

    Returns:
        Normalized log probabilities

    TODO:
    - Compute log normalizer using log_sum_exp
    - Subtract normalizer from log_probs
    - Return normalized log probabilities
    - exp(result) should sum to 1 along axis
    """
    # Compute log normalizer using log_sum_exp
    log_normalizer = log_sum_exp(log_probs, axis=axis)

    # Expand dimensions if necessary to enable broadcasting
    if axis is not None:
        # Expand the dimensions for proper broadcasting
        shape = [1] * log_probs.ndim
        shape[axis] = log_probs.shape[axis]
        log_normalizer = np.reshape(log_normalizer, [s if i != axis else 1 for i, s in enumerate(log_probs.shape)])

    # Subtract normalizer from log_probs
    normalized = log_probs - log_normalizer

    return normalized


def ensure_positive_definite(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Ensure a matrix is positive definite by adding small value to diagonal.

    Args:
        matrix: Square matrix (e.g., covariance matrix)
        epsilon: Small value to add to diagonal

    Returns:
        Positive definite matrix

    TODO:
    - Check if matrix is already positive definite
    - If not, add epsilon * I to diagonal
    - Verify all eigenvalues are positive
    - Return adjusted matrix
    """
    # Make a copy to avoid modifying the original
    result = matrix.copy()

    # Check if matrix is already positive definite
    try:
        # Attempt Cholesky decomposition - only works for positive definite matrices
        np.linalg.cholesky(result)
        return result
    except np.linalg.LinAlgError:
        # Matrix is not positive definite, adjust it
        pass

    # Add epsilon to diagonal to make it positive definite
    n = result.shape[0]
    result = result + epsilon * np.eye(n)

    # Verify all eigenvalues are positive
    eigenvalues = np.linalg.eigvalsh(result)
    while np.min(eigenvalues) <= 0:
        # If still not positive definite, increase epsilon
        epsilon *= 10
        result = matrix + epsilon * np.eye(n)
        eigenvalues = np.linalg.eigvalsh(result)

    return result


def compute_accuracy_metrics(true_states: np.ndarray, predicted_states: np.ndarray) -> Dict[str, float]:
    """
    Compute classification accuracy metrics.

    Args:
        true_states: Ground truth state sequence
        predicted_states: Predicted state sequence

    Returns:
        Dictionary with accuracy, precision, recall, F1 score

    TODO:
    - Compute overall accuracy: correct / total
    - Compute per-class precision and recall
    - Compute F1 score: 2 * (precision * recall) / (precision + recall)
    - Handle multi-class case appropriately
    - Return dict with all metrics
    """
    # Compute overall accuracy
    accuracy = np.mean(true_states == predicted_states)

    # Get unique states
    states = np.unique(np.concatenate([true_states, predicted_states]))

    # Compute per-class precision and recall
    precisions = []
    recalls = []
    f1_scores = []

    for state in states:
        # True positives: correctly predicted as this state
        tp = np.sum((true_states == state) & (predicted_states == state))
        # False positives: incorrectly predicted as this state
        fp = np.sum((true_states != state) & (predicted_states == state))
        # False negatives: should be this state but predicted as another
        fn = np.sum((true_states == state) & (predicted_states != state))

        # Precision: of all predicted as this state, how many were correct
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall: of all true instances of this state, how many were found
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Compute macro-averaged metrics
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    return {
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1
    }


def plot_state_probabilities(time_index: np.ndarray, state_probs: np.ndarray,
                            state_names: List[str], title: str = "State Probabilities") -> None:
    """
    Plot state probabilities over time.

    Args:
        time_index: Time indices or dates
        state_probs: Matrix of state probabilities [time, n_states]
        state_names: Names of states (e.g., ["Bull", "Bear"])
        title: Plot title

    TODO:
    - Create figure and axis
    - Plot each state probability as a line or area
    - Use different colors for each state
    - Add legend with state names
    - Label axes appropriately
    - Add grid for readability
    - Display or save plot
    """
    pass


def plot_regime_overlay(time_index: np.ndarray, prices: np.ndarray,
                       most_likely_states: np.ndarray, state_names: List[str]) -> None:
    """
    Plot price data with regime overlay.

    Args:
        time_index: Time indices or dates
        prices: Price series
        most_likely_states: Most likely state at each time
        state_names: Names of states

    TODO:
    - Create figure with price plot
    - Color background regions based on regime
    - Use different colors for bull/bear states
    - Add legend indicating which color = which regime
    - Label axes
    - Display or save plot
    """
    pass


def plot_returns_by_regime(returns: np.ndarray, states: np.ndarray,
                          state_names: List[str]) -> None:
    """
    Plot distribution of returns by regime.

    Args:
        returns: Return series
        states: State assignment for each return
        state_names: Names of states

    TODO:
    - Create histograms of returns for each regime
    - Use subplots or overlaid distributions
    - Show mean and std for each regime
    - Add kernel density estimates
    - Label clearly which distribution is which regime
    - Display or save plot
    """
    pass


def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio of returns.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio

    TODO:
    - Compute mean excess return: mean(returns) - risk_free_rate
    - Compute std of returns
    - Sharpe = mean_excess / std
    - Optionally annualize if needed
    - Return Sharpe ratio
    """
    # Compute mean excess return
    mean_return = np.mean(returns)
    excess_return = mean_return - risk_free_rate

    # Compute standard deviation of returns
    std_return = np.std(returns, ddof=1)

    # Compute Sharpe ratio
    if std_return == 0:
        return 0.0

    sharpe_ratio = excess_return / std_return

    return sharpe_ratio


def compute_max_drawdown(cumulative_returns: np.ndarray) -> Tuple[float, int, int]:
    """
    Compute maximum drawdown.

    Args:
        cumulative_returns: Cumulative return series

    Returns:
        Tuple of (max_drawdown, start_idx, end_idx)

    TODO:
    - Compute running maximum
    - Compute drawdown at each point: (cumulative - running_max) / running_max
    - Find maximum drawdown and its location
    - Find start (peak) and end (trough) indices
    - Return (max_dd, start, end)
    """
    # Compute running maximum
    running_max = np.maximum.accumulate(cumulative_returns)

    # Compute drawdown at each point
    drawdown = (cumulative_returns - running_max) / running_max

    # Find maximum drawdown and its end (trough) index
    end_idx = np.argmin(drawdown)
    max_drawdown = drawdown[end_idx]

    # Find start (peak) index - the last time we were at running_max before the trough
    start_idx = np.argmax(cumulative_returns[:end_idx + 1])

    return (max_drawdown, start_idx, end_idx)


def create_confusion_matrix(true_states: np.ndarray, predicted_states: np.ndarray,
                           n_states: int) -> np.ndarray:
    """
    Create confusion matrix for state predictions.

    Args:
        true_states: Ground truth states
        predicted_states: Predicted states
        n_states: Number of states

    Returns:
        Confusion matrix [n_states, n_states]

    TODO:
    - Initialize n_states x n_states matrix
    - For each prediction, increment matrix[true, predicted]
    - Optionally normalize rows to show percentages
    - Return confusion matrix
    """
    # Initialize confusion matrix
    confusion = np.zeros((n_states, n_states), dtype=int)

    # Populate confusion matrix
    for true, pred in zip(true_states, predicted_states):
        confusion[int(true), int(pred)] += 1

    return confusion


def plot_confusion_matrix(confusion_matrix: np.ndarray, state_names: List[str]) -> None:
    """
    Plot confusion matrix as heatmap.

    Args:
        confusion_matrix: Confusion matrix
        state_names: Names of states

    TODO:
    - Create heatmap using matplotlib or seaborn
    - Annotate cells with counts or percentages
    - Label rows as "True State" and columns as "Predicted State"
    - Use appropriate color scheme
    - Display or save plot
    """
    pass


def save_model_checkpoint(model, file_path: str) -> None:
    """
    Save model parameters to file.

    Args:
        model: BayesianRegimeSwitchingModel instance
        file_path: Path to save file

    TODO:
    - Extract model parameters (transition matrix, emission params, etc.)
    - Serialize to JSON or pickle
    - Save to specified file path
    - Include metadata (timestamp, version, etc.)
    """
    pass


def load_model_checkpoint(file_path: str):
    """
    Load model parameters from file.

    Args:
        file_path: Path to saved model file

    Returns:
        BayesianRegimeSwitchingModel instance

    TODO:
    - Load serialized data from file
    - Validate data integrity
    - Create new model instance
    - Set loaded parameters
    - Return initialized model
    """
    pass


def compute_information_criterion(log_likelihood: float, n_params: int,
                                  n_observations: int, criterion: str = "aic") -> float:
    """
    Compute AIC or BIC for model selection.

    Args:
        log_likelihood: Log likelihood of model
        n_params: Number of model parameters
        n_observations: Number of data points
        criterion: "aic" or "bic"

    Returns:
        Information criterion value

    TODO:
    - AIC = 2 * n_params - 2 * log_likelihood
    - BIC = n_params * log(n_observations) - 2 * log_likelihood
    - Lower values indicate better model
    - Return criterion value
    """
    if criterion.lower() == "aic":
        # AIC = 2 * n_params - 2 * log_likelihood
        return 2 * n_params - 2 * log_likelihood
    elif criterion.lower() == "bic":
        # BIC = n_params * log(n_observations) - 2 * log_likelihood
        return n_params * np.log(n_observations) - 2 * log_likelihood
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Use 'aic' or 'bic'.")


def generate_synthetic_regime_data(n_samples: int, transition_matrix: np.ndarray,
                                   emission_params: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data from regime-switching model.

    Args:
        n_samples: Number of samples to generate
        transition_matrix: State transition matrix
        emission_params: List of dicts with 'mean' and 'std' for each state

    Returns:
        Tuple of (observations, true_states)

    TODO:
    - Initialize first state randomly
    - For each time step:
        - Sample observation from current state's emission distribution
        - Sample next state from transition probabilities
    - Return (observations, states) arrays
    - Useful for testing and validation
    """
    n_states = len(emission_params)

    # Initialize arrays
    states = np.zeros(n_samples, dtype=int)
    observations = np.zeros(n_samples)

    # Initialize first state randomly (uniform distribution)
    states[0] = np.random.randint(0, n_states)

    # Generate observations and state transitions
    for t in range(n_samples):
        current_state = states[t]

        # Sample observation from current state's emission distribution
        mean = emission_params[current_state]['mean']
        std = emission_params[current_state]['std']
        observations[t] = np.random.normal(mean, std)

        # Sample next state from transition probabilities (if not last time step)
        if t < n_samples - 1:
            transition_probs = transition_matrix[current_state, :]
            states[t + 1] = np.random.choice(n_states, p=transition_probs)

    return observations, states


def bootstrap_confidence_intervals(data: np.ndarray, statistic_fn: callable,
                                   n_bootstrap: int = 1000,
                                   confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence intervals for a statistic.

    Args:
        data: Original data
        statistic_fn: Function to compute statistic on data
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)

    TODO:
    - For n_bootstrap iterations:
        - Resample data with replacement
        - Compute statistic on resampled data
    - Sort bootstrap statistics
    - Compute percentiles for confidence interval
    - Return (lower, upper) bounds
    """
    # Store bootstrap statistics
    bootstrap_stats = np.zeros(n_bootstrap)

    # Perform bootstrap sampling
    n_samples = len(data)
    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)

        # Compute statistic on resampled data
        bootstrap_stats[i] = statistic_fn(bootstrap_sample)

    # Compute percentiles for confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)

    return (lower_bound, upper_bound)


def validate_transition_matrix(transition_matrix: np.ndarray) -> bool:
    """
    Validate that a matrix is a valid transition matrix.

    Args:
        transition_matrix: Matrix to validate

    Returns:
        True if valid, False otherwise

    TODO:
    - Check if square matrix
    - Check if all elements are >= 0
    - Check if all elements are <= 1
    - Check if each row sums to 1 (within tolerance)
    - Return validation result
    """
    # Check if it's a 2D array
    if transition_matrix.ndim != 2:
        return False

    # Check if square matrix
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        return False

    # Check if all elements are >= 0
    if np.any(transition_matrix < 0):
        return False

    # Check if all elements are <= 1
    if np.any(transition_matrix > 1):
        return False

    # Check if each row sums to 1 (within tolerance)
    row_sums = np.sum(transition_matrix, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-8):
        return False

    return True


def format_regime_report(model, observations: np.ndarray) -> str:
    """
    Generate human-readable report of current regime analysis.

    Args:
        model: Fitted BayesianRegimeSwitchingModel
        observations: Recent observations

    Returns:
        Formatted string report

    TODO:
    - Get current regime probabilities
    - Get regime statistics
    - Format into readable report with:
        - Current regime probabilities
        - Most likely regime
        - Transition probabilities
        - Regime characteristics (mean return, volatility)
    - Return formatted string
    """
    pass
