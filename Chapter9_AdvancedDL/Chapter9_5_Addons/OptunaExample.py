from typing import Any

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop

import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.trial import Trial
from optuna.trial import TrialState
from tf_utils.mnistDataAdvanced import MNIST


BATCHSIZE = 128
CLASSES = 10
EPOCHS = 20


def create_model(trial: Trial) -> Sequential:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    model = Sequential()
    model.add(Flatten())
    for i in range(n_layers):
        num_hidden = trial.suggest_int(f"n_units_l{i}", 4, 128, log=True)
        model.add(Dense(num_hidden, activation="relu"))
        dropout = trial.suggest_float(f"dropout_l{i}", 0.2, 0.5)
        model.add(Dropout(rate=dropout))
    model.add(Dense(CLASSES, activation="softmax"))

    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-1, log=True
    )  # 0.00001 0.1
    model.compile(
        loss="categorical_crossentropy",
        optimizer=RMSprop(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    return model


def objective(trial: Trial) -> Any:
    data = MNIST()

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()
    test_dataset = data.get_test_set()

    model = create_model(trial)

    model.fit(
        x=train_dataset,
        batch_size=BATCHSIZE,
        callbacks=[TFKerasPruningCallback(trial, "val_accuracy")],
        epochs=EPOCHS,
        validation_data=val_dataset,
        verbose=1,
    )

    score = model.evaluate(x=test_dataset, verbose=0)
    return score[1]


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(
        deepcopy=False, states=(TrialState.PRUNED,)
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=(TrialState.COMPLETE,)
    )
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
