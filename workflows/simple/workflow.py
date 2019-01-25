import spell.client

# create a client
client = spell.client.from_environment()

print("workflow: {}".format(client.active_workflow))

# basic run
run = client.runs.new(
    command="echo workflow $VAR",
    envvars={
        "VAR": "SUCCESS!!!"
    }
)
print("created run: {}".format(run))
print("run logs:")
for line in run.logs(follow=True):
    print("\t{}".format(line))

# wait for run status
run = client.runs.new(command="sleep 10 && echo 'file contents' > file.txt")
print("\ncreated run {}".format(run.id))
print("waiting for run {} to complete...".format(run.id))
run.wait_status(client.runs.COMPLETE)

# mount a previous run output
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

# wait for run metric value
run = client.runs.new(
    command="python workflows/simple/send_metrics.py",
    commit_label="metrics_sender",
)
print("\ncreated run {}".format(run.id))
print("waiting for metric value...")
run.wait_metric("loss", client.runs.LESS_THAN, 10)
run.kill()
print("loss reached less than 10!")

print("\nLaunching 3 runs in parallel...")
runs = []
for x in range(3):
    runs.append(client.runs.new(command="echo I am run {}".format(x)))

print("Waiting for all 3 runs to complete...")
for run in runs:
    run.wait_status(client.runs.COMPLETE)

print("All 3 runs completed!")
