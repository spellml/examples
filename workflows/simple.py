import spell.client
client = spell.client.from_environment()

print(client.active_workflow)

r1 = client.runs.new(command="echo Hello World! > foo.txt")
r1.wait_status(*client.runs.FINAL)
r1.refresh()
if r1.status != client.runs.COMPLETE:
    raise OSError(f"failed at run {r1.id}")

r2 = client.runs.new(
    command="cat /mnt/foo.txt",
    attached_resources={f"runs/{r1.id}/foo.txt": "/mnt/foo.txt"}
)
r2.wait_status(*client.runs.FINAL)
r2.refresh()
if r2.status != client.runs.COMPLETE:
    raise OSError(f"failed at run {r2.id}")

print("Finished workflow!")
