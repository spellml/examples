import spell.client

# Create a Spell client
client = spell.client.from_environment()

print("workflow: {}".format(client.active_workflow))

# Launch a basic run
run = client.runs.new(
    command="echo workflow $VAR",
    envvars={
        "VAR": "SUCCESS!!!"
    }
)

# Fetch the logs from the basic run via API
print("created run: {}".format(run))
print("run logs:")
for line in run.logs(follow=True):
    print("\t{}".format(line))

# Launch a run which saves a file, wait for this run to complete
run = client.runs.new(command="sleep 10 && echo 'file contents' > file.txt")
print("\ncreated run {}".format(run.id))
print("waiting for run {} to complete...".format(run.id))
run.wait_status(client.runs.COMPLETE)

# Launch a new run which mounts the output of the prior run
run = client.runs.new(
    command="cat /mnt/file.txt",
    attached_resources={
        "runs/{}/file.txt".format(run.id): "/mnt/file.txt"
    },
)
print("\ncreated run {}".format(run.id))
print("run logs:")
for line in run.logs(follow=True):
    if line.status == client.runs.RUNNING and not line.status_event:
        print("\t{}".format(line))

run.wait_status(client.runs.COMPLETE)

# Launch a new run which sends a metric 'loss' with values 30, 29, 28, ...
run = client.runs.new(
    command="python workflows/simple/send_metrics.py",
    commit_label="metrics_sender",
)
print("\ncreated run {}".format(run.id))

# Wait for the loss metric to go below 10, labels this run 'killed'
print("waiting for metric value...")
run.wait_metric("loss", client.runs.LESS_THAN, 10)
run.kill()
run.add_label("killed")
print("loss reached less than 10!")

# Launch multiple runs in parallel
print("\nLaunching 3 runs in parallel...")
runs = []
for x in range(3):
    runs.append(client.runs.new(command="echo I am run {}".format(x)))

# Wait for all runs to complete
print("Waiting for all 3 runs to complete...")
for run in runs:
    run.wait_status(client.runs.COMPLETE)
print("All 3 runs completed!")
