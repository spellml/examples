import spell.client

from spell.client.runs import RunsService

from spell.api.models import ValueSpec

# Create a Spell client
client = spell.client.from_environment()

# Launch a basic grid search. The first argument are the parameters
# they are a dictionary of their name mapped to the array of values
# new_grid_search accepts most run arguments as kwargs as well
#
# This launches a 6 run search
# a: 0, b: 1.3
# a: 2, b: 1.3
# a: 4, b: 1.3
# a: 0, b: 1.7
# a: 2, b: 1.7
# a: 4, b: 1.7
search = client.hyper.new_grid_search(
    {'a': ValueSpec([0,2,4]),
     'b': ValueSpec([1.3, 1.7])},
    commit_label="spell-examples",
    command="python hyperparameters/basic.py --start :a: --stepsize :b: --steps 30",
)

# Wait for all runs to complete
for run in search.runs:
    run.wait_status(*RunsService.FINAL)
    run.refresh()
    if run.status != RunsService.COMPLETE:
        print("Warning run {} finished with status {}".format(run.id, run.status))
    elif run.user_exit_code != 0:
        print("Warning run {} finished with nonzero exit code {}".format(run.id, run.user_exit_code))

print("All runs complete")
