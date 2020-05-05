import spell.client
client = spell.client.from_environment()

# create the first run to download the dataset (War and Peace, by Leo Tolstoy)
# if desired, replace data_url with url to another plain text file to train on
data_url = "https://www.gutenberg.org/files/2600/2600-0.txt"
r1 = client.runs.new(
    command="wget -O input.txt {}".format(data_url)
)
print("waiting for run {} to complete".format(r1.id))
r1.wait_status(*client.runs.FINAL)
r1.refresh()
if r1.status != client.runs.COMPLETE:
    raise OSError(f"failed at run {r1.id}")

# create the second run to train char-RNN on the dataset
data_dir = "/data"
r2 = client.runs.new(
    machine_type="V100",
    command="python train.py --data_dir={}".format(data_dir),
    attached_resources={
        "runs/{}/input.txt".format(r1.id): "{}/input.txt".format(data_dir)
    },
    commit_label="char-rnn",
)
print("waiting for run {} to complete".format(r2.id))

r2.wait_status(*client.runs.FINAL)
r2.refresh()
if r2.status != client.runs.COMPLETE:
    raise OSError(f"failed at run {r2.id}")

# create the third run that samples the model to generate some text
r3 = client.runs.new(
    machine_type="V100",
    command="python sample.py",
    attached_resources={"runs/{}/save".format(r3.id): "save"},
    commit_label="char-rnn",
)
print("waiting for run {} to complete".format(r3.id))

r3.wait_status(*client.runs.FINAL)
r3.refresh()
if r3.status != client.runs.COMPLETE:
    raise OSError(f"failed at run {r3.id}")
