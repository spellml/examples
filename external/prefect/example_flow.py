import spell.client
from prefect import Task, Flow
from prefect.utilities.tasks import defaults_from_attrs
from prefect.tasks.secrets import EnvVarSecret


class CreateSpellRun(Task):
    def __init__(self, owner, **kwargs):
        self.owner = owner
        super().__init__(**kwargs)

    @defaults_from_attrs("owner")
    def run(self, command, token, owner=None, **kwargs):
        # Build the client object
        client = spell.client.SpellClient(token=token, owner=owner)

        # Start the run and wait for it to finish
        run = client.runs.new(command=command, **kwargs)
        run.wait_status(*client.runs.FINAL)
        run.refresh()

        # If the run finished with a failed status raise a ValueError so that Prefect knows to
        # mark this task as failed.
        if run.status in [
            client.runs.FAILED,
            client.runs.BUILD_FAILED,
            client.runs.MOUNT_FAILED,
        ]:
            raise ValueError(f"Run #{run.id} failed with status `{run.status}`.")
        if run.user_exit_code != 0:
            raise ValueError(
                f"Run finished with nonzero exit code {run.user_exit_code}."
            )

        return run


def main():
    create_run_in_org = CreateSpellRun(owner="spell-org")

    with Flow("example-flow") as f:
        token = EnvVarSecret("SPELL_TOKEN")
        state = create_run_in_org(
            command="python models/train.py",
            machine_type="t4",
            github_url="https://github.com/spellml/cnn-cifar10.git",
            token=token,
        )

    state = f.run()
    assert state.is_successful()


if __name__ == "__main__":
    main()
