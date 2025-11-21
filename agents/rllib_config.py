def default_config(env_name: str, num_workers: int = 1, custom_model: str = None, custom_model_config: dict = None):
    cfg = {
        'env': env_name,
        'num_workers': num_workers,
        'framework': 'torch',
        'model': {
            'fcnet_hiddens': [256, 256],
            'fcnet_activation': 'relu',
        },
        'train_batch_size': 2000,
        'sgd_minibatch_size': 256,
        'num_sgd_iter': 10,
        'lr': 1e-4,
    }
    if custom_model is not None:
        cfg['model'] = {
            'custom_model': custom_model,
            'custom_model_config': custom_model_config or {},
        }
    return cfg
