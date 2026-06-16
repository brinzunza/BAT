"""
Bayesian Regime-Switching Model

This module implements a Bayesian approach to detecting market regime changes
(bull vs bear states) using hidden Markov models and Bayesian inference.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import log_sum_exp, normalize_log_probs


class BayesianRegimeSwitchingModel:
    """
    A Bayesian Regime-Switching model for detecting bull/bear market states.

    This model uses a Hidden Markov Model framework with Bayesian inference
    to estimate the probability of being in different market regimes.

    Attributes:
        n_states: Number of hidden states (e.g., 2 for bull/bear)
        transition_matrix: Matrix of transition probabilities between states
        emission_params: Parameters for emission distributions (mean, variance per state)
        state_probs: Current probability distribution over states
    """

    def __init__(self, n_states: int = 2):
        """
        Initialize the Bayesian Regime-Switching model.

        Args:
            n_states: Number of hidden states (default 2 for bull/bear)

        TODO:
        - Initialize transition matrix with uniform or informed priors
        - Initialize emission parameters (mean returns, volatility) for each state
        - Initialize initial state probability distribution
        - Set hyperparameters for Bayesian priors (e.g., Dirichlet for transitions)
        """
        self.n_states = n_states

        # Initialize transition matrix with high self-transition probability
        # Regimes tend to persist
        self.transition_matrix = np.zeros((n_states, n_states))
        for i in range(n_states):
            # 90% probability of staying in same state, 10% of switching
            self.transition_matrix[i, i] = 0.9
            self.transition_matrix[i, :] = (1 - 0.9) / (n_states - 1) if n_states > 1 else 1.0
            self.transition_matrix[i, i] = 0.9

        # Initialize emission parameters (mean and std for each state)
        self.emission_params = [
            {'mean': 0.0, 'std': 1.0} for _ in range(n_states)
        ]

        # Initialize initial state probabilities (uniform)
        self.initial_state_probs = np.ones(n_states) / n_states

        # Flag to track if model has been fitted
        self.is_fitted = False

    def initialize_parameters(self, data: np.ndarray) -> None:
        """
        Initialize model parameters using data-driven heuristics.

        Args:
            data: Historical price or return data

        TODO:
        - Use k-means or quantile-based clustering to initialize states
        - Estimate initial mean and variance for each regime from clustered data
        - Set reasonable initial transition probabilities (e.g., high self-transition)
        - Initialize prior distributions for Bayesian framework
        """
        from sklearn.cluster import KMeans

        # Reshape data for k-means (needs 2D array)
        data_reshaped = data.reshape(-1, 1)

        # Use k-means clustering to identify initial regimes
        kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_reshaped)

        # Estimate mean and std for each cluster
        for i in range(self.n_states):
            cluster_data = data[labels == i]

            if len(cluster_data) > 0:
                self.emission_params[i]['mean'] = np.mean(cluster_data)
                self.emission_params[i]['std'] = np.std(cluster_data)

                # Ensure minimum std to avoid numerical issues
                if self.emission_params[i]['std'] < 1e-6:
                    self.emission_params[i]['std'] = 1e-6

        # Sort states by mean (ascending order: bear to bull)
        sorted_indices = np.argsort([p['mean'] for p in self.emission_params])
        self.emission_params = [self.emission_params[i] for i in sorted_indices]

    def compute_emission_probability(self, observation: float, state: int) -> float:
        """
        Compute the probability of observing a value given a hidden state.

        Args:
            observation: The observed value (e.g., return)
            state: The hidden state index

        Returns:
            Probability of observation given state

        TODO:
        - Implement Gaussian emission probability: P(observation | state)
        - Use state-specific mean and variance parameters
        - Consider alternative distributions (Student-t for heavy tails)
        - Handle edge cases (zero variance, extreme values)
        """
        from scipy.stats import norm

        # Get emission parameters for this state
        mean = self.emission_params[state]['mean']
        std = self.emission_params[state]['std']

        # Compute Gaussian probability density
        prob = norm.pdf(observation, loc=mean, scale=std)

        # Handle edge case of very small probability
        if prob < 1e-300:
            prob = 1e-300

        return prob

    def forward_algorithm(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run forward algorithm to compute filtering probabilities.

        Args:
            observations: Sequence of observations

        Returns:
            Tuple of (alpha matrix, log-likelihood)
            - alpha[t, i] = P(observations[0:t+1], state[t]=i)

        TODO:
        - Initialize forward probabilities at t=0
        - Recursively compute forward probabilities for t=1..T
        - Use log-space computation to prevent numerical underflow
        - Return normalized alpha matrix and total log-likelihood
        """
        T = len(observations)
        log_alpha = np.zeros((T, self.n_states))

        # Initialize at t=0: alpha[0, i] = P(state[0]=i) * P(obs[0] | state[0]=i)
        for i in range(self.n_states):
            emission_prob = self.compute_emission_probability(observations[0], i)
            log_alpha[0, i] = np.log(self.initial_state_probs[i]) + np.log(emission_prob)

        # Forward recursion for t=1..T-1
        for t in range(1, T):
            for j in range(self.n_states):
                # Compute sum over all previous states
                log_probs = np.zeros(self.n_states)
                for i in range(self.n_states):
                    log_probs[i] = log_alpha[t-1, i] + np.log(self.transition_matrix[i, j])

                # Emission probability
                emission_prob = self.compute_emission_probability(observations[t], j)

                # Alpha[t, j] = P(obs[t] | state[t]=j) * sum_i(alpha[t-1, i] * transition[i, j])
                log_alpha[t, j] = log_sum_exp(log_probs) + np.log(emission_prob)

        # Compute total log-likelihood
        log_likelihood = log_sum_exp(log_alpha[-1, :])

        return log_alpha, log_likelihood

    def backward_algorithm(self, observations: np.ndarray) -> np.ndarray:
        """
        Run backward algorithm for smoothing.

        Args:
            observations: Sequence of observations

        Returns:
            beta matrix where beta[t, i] = P(observations[t+1:T] | state[t]=i)

        TODO:
        - Initialize backward probabilities at t=T
        - Recursively compute backward probabilities for t=T-1..0
        - Use log-space computation to prevent numerical underflow
        - Return normalized beta matrix
        """
        T = len(observations)
        log_beta = np.zeros((T, self.n_states))

        # Initialize at t=T-1: beta[T-1, i] = 1 (log = 0)
        log_beta[T-1, :] = 0.0

        # Backward recursion for t=T-2..0
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                # Compute sum over all next states
                log_probs = np.zeros(self.n_states)
                for j in range(self.n_states):
                    emission_prob = self.compute_emission_probability(observations[t+1], j)
                    log_probs[j] = (np.log(self.transition_matrix[i, j]) +
                                   np.log(emission_prob) +
                                   log_beta[t+1, j])

                # Beta[t, i] = sum_j(transition[i, j] * P(obs[t+1] | state[t+1]=j) * beta[t+1, j])
                log_beta[t, i] = log_sum_exp(log_probs)

        return log_beta

    def compute_state_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute filtered and smoothed state probabilities.

        Args:
            observations: Sequence of observations

        Returns:
            gamma matrix where gamma[t, i] = P(state[t]=i | observations)

        TODO:
        - Run forward and backward algorithms
        - Combine alpha and beta to get gamma (smoothed probabilities)
        - Normalize probabilities to sum to 1 at each time step
        - Return smoothed state probability matrix
        """
        # Run forward and backward algorithms
        log_alpha, _ = self.forward_algorithm(observations)
        log_beta = self.backward_algorithm(observations)

        # Combine: gamma[t, i] = alpha[t, i] * beta[t, i] / P(observations)
        # In log space: log_gamma[t, i] = log_alpha[t, i] + log_beta[t, i] - log_normalizer
        log_gamma = log_alpha + log_beta

        # Normalize at each time step
        T = len(observations)
        gamma = np.zeros((T, self.n_states))

        for t in range(T):
            log_normalizer = log_sum_exp(log_gamma[t, :])
            gamma[t, :] = np.exp(log_gamma[t, :] - log_normalizer)

        return gamma

    def compute_transition_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute pairwise state transition probabilities.

        Args:
            observations: Sequence of observations

        Returns:
            xi matrix where xi[t, i, j] = P(state[t]=i, state[t+1]=j | observations)

        TODO:
        - Use forward-backward results to compute xi
        - xi[t,i,j] = alpha[t,i] * transition[i,j] * emission[j,obs[t+1]] * beta[t+1,j]
        - Normalize at each time step
        - Return pairwise transition probability tensor
        """
        T = len(observations)

        # Run forward and backward algorithms
        log_alpha, _ = self.forward_algorithm(observations)
        log_beta = self.backward_algorithm(observations)

        # Initialize xi tensor
        xi = np.zeros((T-1, self.n_states, self.n_states))

        # Compute xi for each time step
        for t in range(T-1):
            log_xi = np.zeros((self.n_states, self.n_states))

            for i in range(self.n_states):
                for j in range(self.n_states):
                    emission_prob = self.compute_emission_probability(observations[t+1], j)
                    log_xi[i, j] = (log_alpha[t, i] +
                                   np.log(self.transition_matrix[i, j]) +
                                   np.log(emission_prob) +
                                   log_beta[t+1, j])

            # Normalize
            log_normalizer = log_sum_exp(log_xi.flatten())
            xi[t, :, :] = np.exp(log_xi - log_normalizer)

        return xi

    def baum_welch_step(self, observations: np.ndarray) -> float:
        """
        Perform one iteration of Baum-Welch (EM) algorithm.

        Args:
            observations: Sequence of observations

        Returns:
            Log-likelihood of observations given current parameters

        TODO:
        - E-step: Compute gamma and xi using forward-backward
        - M-step: Update transition matrix using expected transitions
        - M-step: Update emission parameters using expected state occupancies
        - Incorporate Bayesian priors in M-step (MAP estimation)
        - Return current log-likelihood for convergence check
        """
        T = len(observations)

        # E-step: Compute gamma and xi
        gamma = self.compute_state_probabilities(observations)
        xi = self.compute_transition_probabilities(observations)

        # Compute log-likelihood
        _, log_likelihood = self.forward_algorithm(observations)

        # M-step: Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = np.sum(xi[:, i, j])
                denominator = np.sum(gamma[:-1, i])

                if denominator > 0:
                    self.transition_matrix[i, j] = numerator / denominator
                else:
                    self.transition_matrix[i, j] = 1.0 / self.n_states

        # Normalize transition matrix rows
        for i in range(self.n_states):
            row_sum = np.sum(self.transition_matrix[i, :])
            if row_sum > 0:
                self.transition_matrix[i, :] /= row_sum

        # M-step: Update emission parameters
        for i in range(self.n_states):
            # Weighted mean
            weights = gamma[:, i]
            weight_sum = np.sum(weights)

            if weight_sum > 0:
                mean = np.sum(weights * observations) / weight_sum
                # Weighted variance
                variance = np.sum(weights * (observations - mean)**2) / weight_sum
                std = np.sqrt(variance)

                # Ensure minimum std
                if std < 1e-6:
                    std = 1e-6

                self.emission_params[i]['mean'] = mean
                self.emission_params[i]['std'] = std

        # Update initial state probabilities
        self.initial_state_probs = gamma[0, :]

        return log_likelihood

    def fit(self, observations: np.ndarray, max_iterations: int = 100,
            tolerance: float = 1e-4) -> None:
        """
        Fit the model to observed data using Baum-Welch algorithm.

        Args:
            observations: Historical data (returns or prices)
            max_iterations: Maximum number of EM iterations
            tolerance: Convergence threshold for log-likelihood change

        TODO:
        - Initialize parameters if not already done
        - Iterate Baum-Welch steps until convergence or max iterations
        - Track log-likelihood at each iteration
        - Check for convergence using tolerance threshold
        - Store final parameters in model attributes
        """
        # Initialize parameters if not already done
        if not self.is_fitted:
            self.initialize_parameters(observations)

        # Track log-likelihoods
        log_likelihoods = []
        prev_log_likelihood = -np.inf

        # Iterate Baum-Welch until convergence
        for iteration in range(max_iterations):
            # Perform one Baum-Welch step
            log_likelihood = self.baum_welch_step(observations)
            log_likelihoods.append(log_likelihood)

            # Check for convergence
            improvement = log_likelihood - prev_log_likelihood

            if iteration > 0 and abs(improvement) < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break

            prev_log_likelihood = log_likelihood

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Log-likelihood = {log_likelihood:.4f}")

        # Mark as fitted
        self.is_fitted = True
        self.log_likelihoods_ = log_likelihoods

    def predict_state_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict current state probabilities given observations.

        Args:
            observations: Recent observation sequence

        Returns:
            Current state probability distribution

        TODO:
        - Run forward algorithm on observation sequence
        - Extract final state probabilities
        - Normalize to get probability distribution
        - Return array of probabilities for each state
        """
        # Run forward algorithm
        log_alpha, _ = self.forward_algorithm(observations)

        # Extract final state probabilities
        log_probs = log_alpha[-1, :]

        # Normalize
        log_normalizer = log_sum_exp(log_probs)
        probs = np.exp(log_probs - log_normalizer)

        return probs

    def predict_regime_change(self, observations: np.ndarray,
                             from_state: int, to_state: int) -> float:
        """
        Predict probability of regime change from one state to another.

        Args:
            observations: Recent observation sequence
            from_state: Source state index (e.g., 0 for bull)
            to_state: Target state index (e.g., 1 for bear)

        Returns:
            Probability of transitioning from from_state to to_state

        TODO:
        - Get current state probabilities
        - Weight transition probabilities by current state beliefs
        - Compute expected transition probability to target state
        - Consider using Viterbi for most likely state sequence
        - Return probability of regime shift
        """
        # Get current state probabilities
        current_probs = self.predict_state_probabilities(observations)

        # Probability of being in from_state and transitioning to to_state
        transition_prob = current_probs[from_state] * self.transition_matrix[from_state, to_state]

        return transition_prob

    def viterbi_algorithm(self, observations: np.ndarray) -> Tuple[List[int], float]:
        """
        Find most likely state sequence using Viterbi algorithm.

        Args:
            observations: Sequence of observations

        Returns:
            Tuple of (most likely state sequence, log probability)

        TODO:
        - Initialize Viterbi probabilities at t=0
        - Recursively compute max probabilities for each state
        - Track backpointers for path reconstruction
        - Backtrack to find most likely state sequence
        - Return state sequence and its log probability
        """
        T = len(observations)

        # Initialize Viterbi probability matrix and backpointers
        log_delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialize at t=0
        for i in range(self.n_states):
            emission_prob = self.compute_emission_probability(observations[0], i)
            log_delta[0, i] = np.log(self.initial_state_probs[i]) + np.log(emission_prob)

        # Recursively compute max probabilities
        for t in range(1, T):
            for j in range(self.n_states):
                # Find max over previous states
                log_probs = np.zeros(self.n_states)
                for i in range(self.n_states):
                    log_probs[i] = log_delta[t-1, i] + np.log(self.transition_matrix[i, j])

                # Store max and argmax
                psi[t, j] = np.argmax(log_probs)
                max_log_prob = np.max(log_probs)

                # Add emission probability
                emission_prob = self.compute_emission_probability(observations[t], j)
                log_delta[t, j] = max_log_prob + np.log(emission_prob)

        # Backtrack to find most likely sequence
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(log_delta[T-1, :])
        max_log_prob = np.max(log_delta[T-1, :])

        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states.tolist(), max_log_prob

    def sample_posterior(self, observations: np.ndarray, n_samples: int = 1000) -> Dict:
        """
        Sample from posterior distribution of parameters using MCMC.

        Args:
            observations: Historical data
            n_samples: Number of MCMC samples

        Returns:
            Dictionary of posterior samples for transition matrix and emission params

        TODO:
        - Implement Gibbs sampling or Metropolis-Hastings
        - Sample transition matrix from Dirichlet posterior
        - Sample emission parameters from Normal-Inverse-Gamma posterior
        - Implement proper burn-in and thinning
        - Return dictionary with parameter samples
        """
        pass

    def compute_regime_statistics(self, observations: np.ndarray) -> Dict:
        """
        Compute statistics about regime behavior.

        Args:
            observations: Historical data

        Returns:
            Dictionary with regime duration, transition counts, etc.

        TODO:
        - Compute most likely state sequence using Viterbi
        - Calculate average duration in each regime
        - Count transitions between regimes
        - Compute regime-specific return statistics (mean, volatility)
        - Return dictionary of statistics
        """
        # Compute most likely state sequence
        states, _ = self.viterbi_algorithm(observations)
        states = np.array(states)

        stats = {}

        # Calculate average duration in each regime
        durations = {i: [] for i in range(self.n_states)}
        current_state = states[0]
        duration = 1

        for t in range(1, len(states)):
            if states[t] == current_state:
                duration += 1
            else:
                durations[current_state].append(duration)
                current_state = states[t]
                duration = 1

        # Add final duration
        durations[current_state].append(duration)

        # Compute average durations
        stats['average_duration'] = {}
        for state in range(self.n_states):
            if len(durations[state]) > 0:
                stats['average_duration'][state] = np.mean(durations[state])
            else:
                stats['average_duration'][state] = 0

        # Count transitions between regimes
        transition_counts = np.zeros((self.n_states, self.n_states))
        for t in range(len(states) - 1):
            transition_counts[states[t], states[t+1]] += 1

        stats['transition_counts'] = transition_counts

        # Compute regime-specific return statistics
        stats['regime_stats'] = {}
        for state in range(self.n_states):
            state_mask = states == state
            state_obs = observations[state_mask]

            if len(state_obs) > 0:
                stats['regime_stats'][state] = {
                    'mean': np.mean(state_obs),
                    'std': np.std(state_obs),
                    'min': np.min(state_obs),
                    'max': np.max(state_obs),
                    'count': len(state_obs)
                }
            else:
                stats['regime_stats'][state] = {
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0
                }

        return stats

    def get_current_regime_probability(self, observations: np.ndarray) -> Dict[str, float]:
        """
        Get current probability of being in each regime with labels.

        Args:
            observations: Recent observation data

        Returns:
            Dictionary mapping regime names to probabilities

        TODO:
        - Predict current state probabilities
        - Map state indices to regime names (bull, bear, etc.)
        - Return human-readable dictionary of regime probabilities
        """
        # Predict current state probabilities
        probs = self.predict_state_probabilities(observations)

        # Map state indices to regime names
        if self.n_states == 2:
            # Binary: bear (low returns) and bull (high returns)
            regime_names = ['Bear', 'Bull']
        else:
            # Multi-state: use generic names
            regime_names = [f'Regime_{i}' for i in range(self.n_states)]

        # Create human-readable dictionary
        regime_probs = {name: prob for name, prob in zip(regime_names, probs)}

        return regime_probs
