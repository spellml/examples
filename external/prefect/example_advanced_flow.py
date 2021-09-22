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
        spell_run = client.runs.new(command=command, **kwargs)
        spell_run.wait_status(*client.runs.FINAL)
        spell_run.refresh()

        # If the run finished with a failed status raise a ValueError so that Prefect knows to
        # mark this task as failed.
        if spell_run.status in [
            client.runs.FAILED,
            client.runs.BUILD_FAILED,
            client.runs.MOUNT_FAILED,
        ]:
            raise ValueError(
                f"Run #{spell_run.id} failed with status `{spell_run.status}`."
            )
        if spell_run.user_exit_code != 0:
            raise ValueError(
                f"Run finished with nonzero exit code {spell_run.user_exit_code}."
            )

        return spell_run


class RegisterSpellModel(Task):
    def __init__(self, owner, **kwargs):
        self.owner = owner
        super().__init__(**kwargs)

    @defaults_from_attrs("owner")
    def run(self, token, model_name, spell_run, owner=None, **kwargs):
        # Build the client object
        client = spell.client.SpellClient(token=token, owner=owner)

        # Register the model using the run outputs
        model_files_spellfs_path = f"runs/{spell_run.id}"
        model = client.models.new(model_name, resource=model_files_spellfs_path)

        return model


def main():
    create_spell_run = CreateSpellRun(owner="spell-org")
    register_spell_model = RegisterSpellModel(owner="spell-org")

    with Flow("example-flow") as f:
        token = EnvVarSecret("SPELL_TOKEN")
        training_run = create_spell_run(
            command="python models/train.py",
            machine_type="t4",
            github_url="https://github.com/spellml/cnn-cifar10.git",
            token=token,
        )
        model_artifact = register_spell_model(
            model_name="cnn-cifar10-prefect",
            spell_run=training_run,
            run_output_path="checkpoints/model_final.pth",
            token=token,
        )

    state = f.run()
    assert state.is_successful()


if __name__ == "__main__":
    main()
