from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class DQN(Model):
    def __init__(self, state_shape, num_actions, learning_rate):
        super().__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate

        input_state = Input(shape=state_shape)
        x = Dense(20)(input_state)
        x = Activation("relu")(x)
        x = Dense(20)(x)
        x = Activation("relu")(x)
        output_pred = Dense(self.num_actions)(x)

        self.model = Model(inputs=input_state, outputs=output_pred)
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

    def train(self, states, q_values):
        self.model.fit(states, q_values, verbose=0)

    def predict(self, state):
        return self.model.predict(state)

    def load_model(self, path):
        self.model.load_weights(path)

    def save_model(self, path):
        self.model.save_weights(path)
