import spell.client

# Create a client
client = spell.client.from_environment()

# create the first run to download the dataset (War and Peace, by Leo Tolstoy)
# if desired, replace data_url with url to another plain text file to train on
data_url = "https://www.gutenberg.org/files/2600/2600-0.txt"
r = client.runs.new(
    command="wget -O input.txt {}".format(data_url)
)
print("waiting for run {} to complete".format(r.id))
r.wait_status(client.runs.COMPLETE)

# create the second run to train char-RNN on the dataset
data_dir = "/data"
r = client.runs.new(
    machine_type="K80",  # can choose V100 if preferred
    command="python train.py --data_dir={}".format(data_dir),
    attached_resources={
        "runs/{}/input.txt".format(r.id): "{}/input.txt".format(data_dir)
    },
    commit_label="char-rnn",
)
print("waiting for run {} to complete".format(r.id))
r.wait_status(client.runs.COMPLETE)

# create the third run that samples the model to generate some text
r = client.runs.new(
    machine_type="K80",
    command="python sample.py",
    attached_resources={"runs/{}/save".format(r.id): "save"},
    commit_label="char-rnn",
)
print("waiting for run {} to complete".format(r.id))
r.wait_status(client.runs.COMPLETE)

# print the logs from the last run
# generated text should be the last log line
print("Logs from run {}:".format(r.id))
for line in r.logs():
    if line.status == client.runs.RUNNING and not line.status_event:
        print(line)
