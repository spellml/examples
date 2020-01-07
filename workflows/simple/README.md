# Simple Spell Workflow Example

This is an example of how to use the Spell Python API and a Workflow to
start runs, stream logs of runs, wait for a run to reach a specific state,
or to wait for certain metric values.

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
