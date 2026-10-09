# Long Short-Term Memory (LSTM) Networks

## What an LSTM does

An LSTM is a recurrent neural network architecture designed to learn patterns
in sequences. It maintains a cell state that carries information between time
steps. Gates regulate how information is retained, added, and exposed as output.

## Forget gate

The forget gate controls how much of the previous cell state is retained.
It uses the current input and previous hidden state to produce sigmoid values
between zero and one. A value near zero removes most of the corresponding
information; a value near one preserves most of it. These values multiply the
previous cell state element by element.

## Input gate and cell update

The input gate controls how much new candidate information is added to the
cell state. The candidate values are computed using a tanh activation. The new
cell state combines the retained old cell state with the gated candidate values.

## Output gate

The output gate controls which parts of the updated cell state contribute to
the current hidden state. The updated cell state passes through tanh and is
multiplied by the output gate values to produce the hidden state.

## Why gates matter

The cell state and gated updates help LSTMs handle long-term dependencies and
reduce the vanishing-gradient difficulties of a basic recurrent neural network.
They do not guarantee perfect memory or eliminate every training difficulty.

## Example use cases

LSTMs can be used for time-series forecasting, text sequence modeling, and
sequence classification. For example, a forecasting model can use previous
observations in a sequence to estimate a future value.
