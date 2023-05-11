# Performances

## Grid Search

Model 18: 0.7555 in best epoch
{
    'dense_layer_size': 512,
    'filter_block1': 32,
    'filter_block2': 64,
    'filter_block3': 64,
    'kernel_size_block1': 3,
    'kernel_size_block2': 3,
    'kernel_size_block3': 7,
    'learning_rate': 0.001,
    'optimizer': RMSprop
}

## Random Search

Model 1: 0.7413 in best epoch
{
    'dense_layer_size': 734,
    'filter_block1': 29,
    'filter_block2': 45,
    'filter_block3': 61,
    'kernel_size_block1': 3,
    'kernel_size_block2': 6,
    'kernel_size_block3': 5,
    'learning_rate': 0.0010996100625235421,
    'optimizer': Adam
}

## Final Performance

Model: 0.7200
{
    'dense_layer_size': 512,
    'filter_block1': 32,
    'filter_block2': 64,
    'filter_block3': 64,
    'kernel_size_block1': 3,
    'kernel_size_block2': 3,
    'kernel_size_block3': 7,
    'learning_rate': 0.001,
    'optimizer': RMSprop
}
