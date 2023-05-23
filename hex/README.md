## Topology

HexEnvironment needs a "Transformer",
responsible for transforming the board to the correct configuration for the model.

QEngine needs a "QModel", that describes the NN architecture.
The QModel input and the Transformer output need to match.

The model always needs a last layer with a sigmoid activation function and
as many output neurons as there are actions.