# Simple Spell Workflow Example

This is an example of how to use the Spell Python API and a workflow to
start runs, stream logs of runs, wait for a run to reach a specific state,
or to wait for a specific 

# Run it

1. Clone this repo:
```ShellSession
$ git clone https://github.com/spellrun/spell-examples.git
```
2. Run the workflow:
```ShellSession
$ cd spell-examples
$ spell workflow --repo metrics_sender=. python workflows/simple/workflow.py
```

# Details

The file `workflow.py` is a python script that:
1. downloads a text file containing War and Peace, by Leo Tolstoy.
2. trains a recurrent neural network (RNN) on the data to predict the next characters
in a sequence
3. uses the trained RNN to generate new text
