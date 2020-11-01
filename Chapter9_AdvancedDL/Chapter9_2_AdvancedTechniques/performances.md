# Performances

## ReLU, GlorotNormal

Test performance: [0.7723388218813094, 0.83991253]

## ReLU, RandomNormal

Test performance: [1.0958411046052114, 0.80116606]

## ReLU, RandomUniform

Test performance: [0.6931655853715251, 0.49775293]

## ReLU, VarianceScaling

Test performance: [0.8773047265806074, 0.84537834]

## ReLU, GlorotUniform

Test performance: [1.024208527672382, 0.8688206]

## ELU, GlorotUniform

Test performance: [1.3764146850043364, 0.77310824]

## LeakyReLU, GlorotUniform

Test performance: [1.162869258232747, 0.8344467]

## ReLU, GlorotUniform, Dropout=0.0, BatchNorm=True

Test performance: [0.5189515141030208, 0.8935989]

## ReLU, GlorotUniform, Dropout=0.0, BatchNorm=True, LRS=1

Test performance: [0.4675678464344353, 0.8945678]

## ReLU, GlorotUniform, Dropout=0.0, BatchNorm=True, LRS=2

Test performance: [0.4626723082556725, 0.9087817]

## ReLU, GlorotUniform, Dropout=0.0, BatchNorm=True, LRPL1

Test performance: [0.5035442909159985, 0.9120612]

## ReLU, GlorotUniform, Dropout=0.0, BatchNorm=True, LRPL2

Test performance: [0.45570962834181566, 0.904652]

## Flatten, Dense

Total params: 2,685,218
Test performance: [0.45570962834181566, 0.904652]

## Flatten, Without Dense Layer

Total params: 589,090
Test performance: [0.4583214995971966, 0.85218024]

## Global Average/Max Pooling, Without Dense Layer

Total params: 585,250
Test performance: [0.35320026831936463, 0.89748573]

## Global Average/Max Pooling, Dense Layer

Total params: 719,138
Test performance: [0.3560482378079392, 0.8921414]

## Final Model

Test performance: [0.5897449593881088, 0.9254221]
