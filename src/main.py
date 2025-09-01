import sys
import os
from pathlib import Path

# Add the notebooks directory to path
notebooks_path = Path(__file__).parent / 'data' / 'notebooks'
sys.path.append(str(notebooks_path))

# Import the active learning module
from data.notebooks.AL_BO_Experiment_Slow_Fast import (
    OptimizedActiveLearningPipeline,
    ConfigManager,
    config_manager
)

def run_active_learning_experiment(custom_config=None, speed_mode='slow'):
    """
    Run active learning experiment with custom configuration.
    
    Args:
        custom_config (dict): Override default configuration
        speed_mode (str): 'slow' or 'fast' for production mode
    
    Returns:
        dict: Results from the experiment
    """
    # Set speed mode if in production
    if config_manager.mode == 'production':
        config_manager.set_speed(speed_mode)
    
    # Example custom configuration
    if custom_config is None:
        custom_config = {
            'max_iterations': 3,  # Override default
            'epochs_per_iteration': 5,  # Faster for testing
            'strategies': ['random', 'least_confidence', 'entropy']  # Subset of strategies
        }
    
    # Create and run pipeline
    pipeline = OptimizedActiveLearningPipeline(custom_config)
    results = pipeline.run_comparison()
    
    return results

def main():
    """Main entry point for your project."""
    
    print("="*80)
    print("MAIN PROJECT PIPELINE")
    print("="*80)
    
    # Option 1: Run with default settings
    # print("\n1. Running Active Learning with default settings...")
    # results = run_active_learning_experiment()
    
    # Option 2: Run with custom configuration
    custom_config = {
        'batch_size': 64,
        'initial_labeled': 1000,
        'query_size': 500,
        'max_iterations': 3,
        'epochs_per_iteration': 10,
        'num_runs': 1,
        'strategies': ['random', 'entropy', 'bald', 'diverse_entropy']
    }
    
    print("\n2. Running Active Learning with custom settings...")
    results = run_active_learning_experiment(
        custom_config=custom_config,
        speed_mode='slow'  # or 'fast'
    )

    return results
    
    # Option 3: Programmatically switch modes
    # if config_manager.mode == 'production':
    #     print("\n3. Testing fast mode...")
    #     config_manager.set_speed('fast')
    #     results_fast = run_active_learning_experiment()
    
    # return results

if __name__ == "__main__":
    results = main()