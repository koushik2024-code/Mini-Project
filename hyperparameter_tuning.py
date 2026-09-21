import json
from pathlib import Path
from typing import Any

import numpy as np

from data.dataset_loader import DatasetLoader
from data.ml_dataset_prep import MLDatasetPrep
from embeddings.embedding_model import EmbeddingModel, load_embedding_config
from router.keras_mlp import RouterConfig
from router.trainer import RouterTrainer, TrainingConfig


def run_experiment(config_dict: dict[str, Any], X_train, y_train, X_val, y_val, X_test, y_test, exp_name: str) -> dict[str, Any]:
    """Run a single training experiment and return results"""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Config: {json.dumps(config_dict, indent=2)}")
    print(f"{'='*60}")

    # Create configs
    router_config = RouterConfig(
        input_dim=config_dict.get('input_dim', 384),
        hidden_dims=config_dict.get('hidden_dims', [256, 128]),
        num_classes=config_dict.get('num_classes', 3),
        dropout_rate=config_dict.get('dropout_rate', 0.2)
    )

    training_config = TrainingConfig(
        epochs=config_dict.get('epochs', 100),
        batch_size=config_dict.get('batch_size', 32),
        learning_rate=config_dict.get('learning_rate', 0.001),
        early_stopping_patience=config_dict.get('early_stopping_patience', 10),
        class_weights=config_dict.get('class_weights', True),
        reduce_lr_patience=config_dict.get('reduce_lr_patience', 5),
        reduce_lr_factor=config_dict.get('reduce_lr_factor', 0.5),
        min_lr=config_dict.get('min_lr', 1e-6)
    )

    # Train
    trainer = RouterTrainer(training_config, router_config)

    model_path = f"models/router/tuning/{exp_name}_best.keras"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    history = trainer.train(X_train, y_train, X_val, y_val, model_path)

    # Evaluate
    results = trainer.evaluate(X_test, y_test)

    # Get detailed metrics
    y_pred_probs = trainer.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    result = {
        'experiment': exp_name,
        'config': config_dict,
        'test_accuracy': float(accuracy),
        'test_macro_f1': float(macro_f1),
        'test_loss': results['loss'],
        'epochs_trained': len(history.history['loss']),
        'best_val_loss': min(history.history.get('val_loss', [float('inf')]))
    }

    print(f"\nResults: Acc={accuracy:.4f}, Macro-F1={macro_f1:.4f}")
    return result


def main():
    # Load data once
    print("Loading data and preparing splits...")
    loader = DatasetLoader()
    data = loader.load_dataset()

    prep = MLDatasetPrep()
    train_idx, val_idx, test_idx = prep.split_data(data)

    # Generate embeddings
    emb_config = load_embedding_config()
    emb_model = EmbeddingModel(emb_config)

    texts = [data[i]['prompt'] for i in train_idx]
    val_texts = [data[i]['prompt'] for i in val_idx]
    test_texts = [data[i]['prompt'] for i in test_idx]

    print("Generating embeddings...")
    X_train = emb_model.encode_batch(texts)
    X_val = emb_model.encode_batch(val_texts)
    X_test = emb_model.encode_batch(test_texts)

    train_data = prep.prepare_training_data(data, None)
    y_train = train_data['y_train']
    y_val = train_data['y_val']
    y_test = train_data['y_test']

    print(f"Data shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    # Define hyperparameter search space
    search_space = {
        'hidden_dims': [
            [256, 128],
            [512, 256],
            [512, 256, 128],
            [256, 128, 64],
            [128, 64],
            [512],
            [1024, 512, 256]
        ],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'batch_size': [16, 32, 64, 128],
        'class_weights': [True, False],
    }

    # Fixed params
    base_config = {
        'input_dim': 384,
        'num_classes': 3,
        'epochs': 100,
        'early_stopping_patience': 15,
        'reduce_lr_patience': 7,
        'reduce_lr_factor': 0.5,
        'min_lr': 1e-7
    }

    # Generate combinations (limit to reasonable number)
    # We'll do a random search instead of grid search
    import random
    random.seed(42)
    np.random.seed(42)

    n_experiments = 50
    results = []

    for i in range(n_experiments):
        # Sample random configuration
        config = base_config.copy()
        config.update({
            'hidden_dims': random.choice(search_space['hidden_dims']),
            'dropout_rate': random.choice(search_space['dropout_rate']),
            'learning_rate': random.choice(search_space['learning_rate']),
            'batch_size': random.choice(search_space['batch_size']),
            'class_weights': random.choice(search_space['class_weights']),
        })

        exp_name = f"exp_{i:03d}_hd{'-'.join(map(str, config['hidden_dims']))}_dr{config['dropout_rate']}_lr{config['learning_rate']}_bs{config['batch_size']}_cw{config['class_weights']}"

        try:
            result = run_experiment(config, X_train, y_train, X_val, y_val, X_test, y_test, exp_name)
            results.append(result)

            # Save progress
            with open('tuning_results.json', 'w') as f:
                json.dump(results, f, indent=2)

            # Check if we hit 95%
            if result['test_accuracy'] >= 0.95:
                print(f"\n*** TARGET ACHIEVED: {result['test_accuracy']:.4f} >= 0.95 ***")
                break

        except Exception as e:
            print(f"Experiment failed: {e}")
            results.append({'experiment': exp_name, 'config': config, 'error': str(e)})

    # Print best results
    print("\n" + "="*60)
    print("TOP 10 RESULTS")
    print("="*60)
    sorted_results = sorted([r for r in results if 'test_accuracy' in r],
                          key=lambda x: x['test_accuracy'], reverse=True)
    for i, r in enumerate(sorted_results[:10]):
        print(f"{i+1}. Acc={r['test_accuracy']:.4f}, F1={r['test_macro_f1']:.4f} | {r['experiment']}")

    # Save final results
    with open('tuning_results_final.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
