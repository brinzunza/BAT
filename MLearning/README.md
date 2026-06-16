# Bayesian Regime-Switching Model

A Bayesian approach to detecting market regime changes (bull vs bear states) using Hidden Markov Models and Bayesian inference.

## Overview

This project implements a Bayesian Regime-Switching model that calculates the probability that the market has shifted from one state to another (e.g., "bull" to "bear") based on hidden variables, rather than just predicting a price point.

The model uses:
- **Hidden Markov Models (HMM)** for regime detection
- **Bayesian inference** for parameter estimation
- **Forward-Backward algorithm** for state probability computation
- **MCMC sampling** for posterior inference

## Project Structure

```
MLearning/
├── model.py           # Core regime-switching model (HMM)
├── inference.py       # Bayesian inference methods (MCMC, priors)
├── data_loader.py     # Data loading and preprocessing
├── utils.py           # Utility functions (plotting, metrics, etc.)
├── tests/
│   ├── test_model.py
│   ├── test_inference.py
│   ├── test_data_loader.py
│   └── test_utils.py
├── requirements.txt
└── README.md
```

## Installation

```bash
cd MLearning
pip install -r requirements.txt
```

## Implementation Guide

Follow this order to implement the functions. Each section builds on previous ones, and tests are provided to verify correctness.

---

## Phase 1: Utility Functions (Start Here!)

**File:** `utils.py`

These are standalone functions needed throughout the project. Start here because they have no dependencies.

### 1.1 Numerical Stability Functions

Implement these first as they're used everywhere:

1. **`log_sum_exp`** - Prevents numerical overflow when working with log probabilities
   - Test: `pytest tests/test_utils.py::TestNumericalStability::test_log_sum_exp_basic -v`

2. **`normalize_log_probs`** - Normalizes log probabilities to sum to 1
   - Test: `pytest tests/test_utils.py::TestNumericalStability::test_normalize_log_probs -v`

3. **`ensure_positive_definite`** - Ensures covariance matrices are valid
   - Test: `pytest tests/test_utils.py::TestMatrixOperations -v`

### 1.2 Validation Functions

4. **`validate_transition_matrix`** - Validates transition matrix properties
   - Test: `pytest tests/test_utils.py::TestValidation -v`

### 1.3 Metric Functions

5. **`compute_accuracy_metrics`** - Computes classification metrics
   - Test: `pytest tests/test_utils.py::TestAccuracyMetrics -v`

6. **`compute_sharpe_ratio`** - Financial performance metric
   - Test: `pytest tests/test_utils.py::TestFinancialMetrics::test_compute_sharpe_ratio_positive -v`

7. **`compute_max_drawdown`** - Maximum drawdown computation
   - Test: `pytest tests/test_utils.py::TestFinancialMetrics::test_compute_max_drawdown -v`

8. **`create_confusion_matrix`** - Creates confusion matrix
   - Test: `pytest tests/test_utils.py::TestConfusionMatrix -v`

9. **`compute_information_criterion`** - AIC/BIC for model selection
   - Test: `pytest tests/test_utils.py::TestInformationCriterion -v`

### 1.4 Data Generation

10. **`generate_synthetic_regime_data`** - Generate synthetic data for testing
    - Test: `pytest tests/test_utils.py::TestSyntheticData -v`
    - **This is crucial for testing your model implementation!**

11. **`bootstrap_confidence_intervals`** - Bootstrap CIs
    - Test: `pytest tests/test_utils.py::TestBootstrap -v`

### 1.5 Plotting (Optional - can implement last)

- `plot_state_probabilities`
- `plot_regime_overlay`
- `plot_returns_by_regime`
- `plot_confusion_matrix`

### 1.6 Model Persistence (Optional - implement when needed)

- `save_model_checkpoint`
- `load_model_checkpoint`
- `format_regime_report`

**Verify Phase 1:** `pytest tests/test_utils.py -v`

---

## Phase 2: Data Loading and Preprocessing

**File:** `data_loader.py`

Now implement data handling. You can test these with actual market data from your system.

### 2.1 Basic Data Operations

1. **`__init__`** - Initialize data loader
   - Test: `pytest tests/test_data_loader.py::TestDataLoaderInitialization -v`

2. **`load_price_data`** - Load OHLCV data from your data source
   - Test: `pytest tests/test_data_loader.py::TestLoadPriceData -v`

3. **`compute_returns`** - Compute log or simple returns
   - Test: `pytest tests/test_data_loader.py::TestComputeReturns -v`

### 2.2 Feature Engineering

4. **`compute_volatility`** - Rolling volatility calculation
   - Test: `pytest tests/test_data_loader.py::TestComputeVolatility -v`

5. **`compute_momentum`** - Momentum indicator
   - Test: `pytest tests/test_data_loader.py::TestComputeMomentum -v`

6. **`prepare_features`** - Combine all features
   - Test: `pytest tests/test_data_loader.py::TestPrepareFeatures -v`

### 2.3 Data Processing

7. **`normalize_data`** - Standardization or min-max scaling
   - Test: `pytest tests/test_data_loader.py::TestNormalization::test_normalize_standardize -v`

8. **`denormalize_data`** - Reverse normalization
   - Test: `pytest tests/test_data_loader.py::TestNormalization::test_denormalize_data -v`

9. **`handle_missing_data`** - Handle NaN values
   - Test: `pytest tests/test_data_loader.py::TestMissingDataHandling -v`

10. **`split_train_test`** - Time-series train/test split
    - Test: `pytest tests/test_data_loader.py::TestSplitTrainTest -v`

### 2.4 Advanced Features (Optional)

- `create_sequences` - For sequence-based models
- `add_time_features` - Time-based features
- `compute_technical_indicators` - RSI, MACD, etc.
- `validate_data_quality` - Data quality checks
- `resample_data` - Frequency conversion

**Verify Phase 2:** `pytest tests/test_data_loader.py -v`

---

## Phase 3: Core Model - Hidden Markov Model

**File:** `model.py`

This is the heart of the project. Implement in this order:

### 3.1 Model Initialization

1. **`__init__`** - Initialize model with n_states
   - Test: `pytest tests/test_model.py::TestModelInitialization -v`

2. **`initialize_parameters`** - Initialize transition matrix and emission params
   - Test: `pytest tests/test_model.py::TestParameterInitialization -v`
   - Hint: Use k-means clustering to identify initial regimes

### 3.2 Emission Probabilities

3. **`compute_emission_probability`** - P(observation | state)
   - Test: `pytest tests/test_model.py::TestEmissionProbability -v`
   - Use Gaussian distribution: `scipy.stats.norm.pdf(observation, mean, std)`

### 3.3 Forward-Backward Algorithm

4. **`forward_algorithm`** - Compute alpha matrix
   - Test: `pytest tests/test_model.py::TestForwardAlgorithm -v`
   - **Critical:** Use log-space to prevent underflow
   - Use `log_sum_exp` from utils.py

5. **`backward_algorithm`** - Compute beta matrix
   - Test: `pytest tests/test_model.py::TestBackwardAlgorithm -v`
   - Also use log-space computation

6. **`compute_state_probabilities`** - Combine alpha and beta for smoothed probabilities
   - Test: `pytest tests/test_model.py::TestStateProbabilities -v`
   - Formula: gamma[t,i] = alpha[t,i] * beta[t,i] / sum_j(alpha[t,j] * beta[t,j])

### 3.4 Viterbi Algorithm

7. **`viterbi_algorithm`** - Most likely state sequence
   - Test: `pytest tests/test_model.py::TestViterbi -v`
   - Use dynamic programming with backpointers

### 3.5 Parameter Estimation

8. **`compute_transition_probabilities`** - Compute xi matrix
   - Test: (part of Baum-Welch tests)

9. **`baum_welch_step`** - One EM iteration
   - Test: `pytest tests/test_model.py::TestBaumWelch::test_baum_welch_step_increases_likelihood -v`
   - E-step: compute gamma and xi
   - M-step: update parameters

10. **`fit`** - Full Baum-Welch training
    - Test: `pytest tests/test_model.py::TestBaumWelch::test_fit_converges -v`
    - Iterate until convergence

### 3.6 Prediction and Analysis

11. **`predict_state_probabilities`** - Current regime probabilities
    - Test: `pytest tests/test_model.py::TestPrediction -v`

12. **`predict_regime_change`** - Probability of regime transition
    - Test: `pytest tests/test_model.py::TestPrediction::test_predict_regime_change -v`
    - This is your main use case!

13. **`compute_regime_statistics`** - Regime duration, statistics
    - Test: `pytest tests/test_model.py::TestRegimeStatistics -v`

14. **`get_current_regime_probability`** - Human-readable output
    - Test: `pytest tests/test_model.py::TestRegimeStatistics::test_get_current_regime_probability -v`

### 3.7 Advanced (Optional)

15. **`sample_posterior`** - MCMC sampling
    - Test: (complex, implement after Phase 4)

**Verify Phase 3:** `pytest tests/test_model.py -v`

---

## Phase 4: Bayesian Inference

**File:** `inference.py`

Add full Bayesian treatment to the model.

### 4.1 Prior Specification

1. **`__init__`** - Initialize inference engine
   - Test: `pytest tests/test_inference.py::TestInferenceInitialization -v`

2. **`set_transition_prior`** - Dirichlet prior for transitions
   - Test: `pytest tests/test_inference.py::TestPriorSpecification -v`

3. **`set_emission_prior`** - Normal-Inverse-Gamma prior
   - Test: `pytest tests/test_inference.py::TestPriorSpecification::test_set_emission_prior_valid -v`

### 4.2 Posterior Sampling

4. **`sample_transition_matrix`** - Sample from Dirichlet posterior
   - Test: `pytest tests/test_inference.py::TestTransitionSampling -v`
   - Use `numpy.random.dirichlet`

5. **`sample_emission_parameters`** - Sample from NIG posterior
   - Test: `pytest tests/test_inference.py::TestEmissionSampling -v`
   - Use conjugate update formulas

### 4.3 MCMC Algorithms

6. **`forward_filtering_backward_sampling`** - Sample state sequences
   - Test: `pytest tests/test_inference.py::TestForwardFilteringBackwardSampling -v`

7. **`gibbs_sampling`** - Full Gibbs sampler
   - Test: `pytest tests/test_inference.py::TestGibbsSampling -v`
   - Alternate between sampling states and parameters

8. **`metropolis_hastings_step`** - MH step (alternative to Gibbs)
   - Test: `pytest tests/test_inference.py::TestMetropolisHastings -v`

### 4.4 Model Evaluation

9. **`compute_log_prior`** - Log prior probability
   - Test: `pytest tests/test_inference.py::TestLogPrior -v`

10. **`compute_posterior_predictive`** - Posterior predictive distribution
    - Test: `pytest tests/test_inference.py::TestPosteriorPredictive -v`

11. **`compute_evidence`** - Marginal likelihood
    - Test: `pytest tests/test_inference.py::TestEvidence -v`

12. **`compute_bayes_factor`** - Model comparison
    - Test: `pytest tests/test_inference.py::TestBayesFactor -v`

### 4.5 Diagnostics

13. **`compute_convergence_diagnostics`** - Check MCMC convergence
    - Test: `pytest tests/test_inference.py::TestConvergenceDiagnostics -v`

14. **`compute_credible_intervals`** - Bayesian confidence intervals
    - Test: `pytest tests/test_inference.py::TestCredibleIntervals -v`

**Verify Phase 4:** `pytest tests/test_inference.py -v`

---

## Testing Strategy

### Run Tests Incrementally

After implementing each function:

```bash
# Test a single function
pytest tests/test_utils.py::TestNumericalStability::test_log_sum_exp_basic -v

# Test a class of functions
pytest tests/test_model.py::TestEmissionProbability -v

# Test entire module
pytest tests/test_model.py -v

# Run all tests
pytest tests/ -v
```

### Test-Driven Development

1. Read the function's TODO comments
2. Look at the corresponding test to understand expected behavior
3. Implement the function
4. Run the test to verify correctness
5. Move to the next function

### Use Synthetic Data

Generate synthetic data with known regimes to verify your model works:

```python
from utils import generate_synthetic_regime_data

transition_matrix = np.array([[0.95, 0.05], [0.1, 0.9]])
emission_params = [
    {'mean': 0.01, 'std': 0.01},   # Bull regime
    {'mean': -0.01, 'std': 0.02}    # Bear regime
]

observations, true_states = generate_synthetic_regime_data(
    n_samples=500,
    transition_matrix=transition_matrix,
    emission_params=emission_params
)

# Now test if your model can recover the true parameters and states
```

---

## Usage Example (After Implementation)

```python
from model import BayesianRegimeSwitchingModel
from data_loader import MarketDataLoader

# Load data
loader = MarketDataLoader()
data = loader.load_price_data("BTC-USD", "2023-01-01", "2024-01-01")
returns = loader.compute_returns(data['close'], method='log')

# Create and fit model
model = BayesianRegimeSwitchingModel(n_states=2)
model.initialize_parameters(returns.values)
model.fit(returns.values, max_iterations=50)

# Predict current regime
recent_returns = returns.tail(30).values
current_probs = model.get_current_regime_probability(recent_returns)
print(f"Current regime probabilities: {current_probs}")

# Predict regime change
change_prob = model.predict_regime_change(
    recent_returns,
    from_state=0,  # Bull
    to_state=1      # Bear
)
print(f"Probability of bull→bear transition: {change_prob:.2%}")
```

---

## Key Concepts

### Hidden Markov Model

- **States:** Hidden market regimes (bull, bear)
- **Observations:** Returns, volatility, etc.
- **Transition Matrix:** P(state_t | state_{t-1})
- **Emission Distribution:** P(observation_t | state_t)

### Forward-Backward Algorithm

- **Forward (alpha):** P(observations_{1:t}, state_t)
- **Backward (beta):** P(observations_{t+1:T} | state_t)
- **Smoothed (gamma):** P(state_t | observations_{1:T})

### Baum-Welch (EM Algorithm)

- **E-step:** Compute expected state occupancies (gamma, xi)
- **M-step:** Update parameters to maximize expected log-likelihood

### Bayesian Inference

- Specify priors on parameters
- Use MCMC to sample from posterior
- Get full uncertainty quantification

---

## Tips and Gotchas

1. **Always use log-space** for probability computations to avoid numerical underflow
2. **Normalize probabilities** after combining to ensure they sum to 1
3. **Test with synthetic data** first before using real market data
4. **Start with 2 states** (bull/bear) before trying more complex models
5. **Check convergence** of Baum-Welch using log-likelihood values
6. **Use informative priors** in Bayesian inference (e.g., higher self-transition probability)
7. **Visualize results** to ensure they make sense (use plotting functions)

---

## Implementation Checklist

### Phase 1: Utils ☐
- [ ] Numerical stability functions
- [ ] Validation functions
- [ ] Metric functions
- [ ] Synthetic data generation
- [ ] All utils tests pass

### Phase 2: Data Loading ☐
- [ ] Load and preprocess data
- [ ] Compute returns and features
- [ ] Normalization and missing data handling
- [ ] All data_loader tests pass

### Phase 3: Core Model ☐
- [ ] Model initialization
- [ ] Emission probabilities
- [ ] Forward-backward algorithms
- [ ] Viterbi algorithm
- [ ] Baum-Welch training
- [ ] Prediction functions
- [ ] All model tests pass

### Phase 4: Bayesian Inference ☐
- [ ] Prior specification
- [ ] Posterior sampling
- [ ] MCMC algorithms
- [ ] Model evaluation
- [ ] Convergence diagnostics
- [ ] All inference tests pass

### Phase 5: Integration ☐
- [ ] End-to-end test with real data
- [ ] Visualizations
- [ ] Documentation
- [ ] Example usage scripts

---

## Resources

### Mathematical Background

- **HMM Tutorial:** Rabiner (1989) "A Tutorial on Hidden Markov Models"
- **Bayesian Methods:** Bishop (2006) "Pattern Recognition and Machine Learning", Chapter 13
- **Regime Switching:** Hamilton (1989) "A New Approach to the Economic Analysis of Nonstationary Time Series"

### Code References

- Forward-Backward: Use log-space version to prevent underflow
- Viterbi: Dynamic programming with backtracking
- Baum-Welch: Iterative EM until convergence
- Gibbs Sampling: Alternate between state and parameter updates

---

## Questions or Issues?

If you get stuck:

1. Check the TODO comments in the function
2. Look at the corresponding test for expected behavior
3. Review the mathematical formulas in the resources
4. Use synthetic data to debug (you know the true answer)
5. Print intermediate values to understand what's happening

Good luck with your implementation!
