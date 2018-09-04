# Spell Workflow Example: Char-RNN

This is an example of how to use the Spell Python API and a workflow to
train a multi-layer recurrent neural network from character-level language models.

For more information regarding Spell, the Python API, runs, and workflows,
check out the [Spell documentation](https://spell.run/docs).

To run this code you first need to [sign up for a Spell account](https://web.spell.run/register)
and install Spell: `pip install spell`.

# Run it

1. Clone the [Tensorflow character-level language RNN model](https://github.com/sherjilozair/char-rnn-tensorflow)
```ShellSession
$ git clone https://github.com/sherjilozair/char-rnn-tensorflow.git
```
2. Clone this repo:
```ShellSession
$ git clone https://github.com/spellrun/spell-examples.git && cd spell-examples
```
3. Run the workflow
```ShellSession
$ spell workflow --repo char-rnn=../char-rnn-tensorflow/ python char-rnn-workflow/workflow.py
```

# Details

The file `workflow.py` is a python script that:
1. downloads a text file containing War and Peace, by Leo Tolstoy.
2. trains a recurrent neural network (RNN) on the data to predict the next characters
in a sequence
3. uses the trained RNN to generate new text
