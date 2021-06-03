import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
# import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

import numpy as np
import torch

TARGET = "llvm -mcpu=skylake-avx512"


def time_it(model_func):
    import timeit

    timing_number = 10
    timing_repeat = 10
    timing = (
        np.array(timeit.Timer(model_func).repeat(repeat=timing_repeat, number=timing_number))
        * 1000
        / timing_number
    )
    results = {
        "mean": np.mean(timing),
        "median": np.median(timing),
        "std": np.std(timing),
    }
    return results


# XXX: TVM currently only supports JIT PyTorch models which have been *traced*:
# https://tvm.apache.org/docs/api/python/relay/frontend.html#tvm.relay.frontend.from_pytorch.
# Tracing is one of three quantization approaches implemented in PyTorch, and far from appropriate
# for every model.
#
# To satisfy this requirement, the following code applies torch.jit.trace to the input module.
# However, actual model quantization was performed using QAT upstream. Tracing is probably a
# no-op, but it's hard to know for sure.
def get_tvm_model(model, X_ex):
    model = torch.jit.trace(model)
    mod, params = relay.frontend.from_pytorch(model, input_infos=[('input0', X_ex.shape)])

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=TARGET, params=params)

    dev = tvm.device(str(TARGET), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

    module.set_input("input0", X_ex)
    module.run()  # just a test run to make sure it works

    # mod is an IR struct. Used downstream. Same with params, used downstream.
    # module is a Relay Python callable
    return mod, params, module


def tune(mod, params, X_ex):
    number = 10
    repeat = 1
    min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
    timeout = 10  # in seconds

    # create a TVM runner
    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat,
        timeout=timeout,
        min_repeat_ms=min_repeat_ms,
    )

    # Create a simple structure for holding tuning options. We use an XGBoost
    # algorithim for guiding the search. For a production job, you will want to set
    # the number of trials to be larger than the value of 10 used here. For CPU we
    # recommend 1500, for GPU 3000-4000. The number of trials required can depend
    # on the particular model and processor, so it's worth spending some time
    # evaluating performance across a range of values to find the best balance
    # between tuning time and model optimization. Because running tuning is time
    # intensive we set number of trials to 10, but do not recommend a value this
    # small. The ``early_stopping`` parameter is the minimum number of trails to
    # run before a condition that stops the search early can be applied. The
    # measure option indicates where trial code will be built, and where it will be
    # run. In this case, we're using the ``LocalRunner`` we just created and a
    # ``LocalBuilder``. The ``tuning_records`` option specifies a file to write
    # the tuning data to.

    tuning_option = {
        "tuner": "xgb",
        "trials": 10,
        "early_stopping": 100,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"), runner=runner
        ),
        "tuning_records": "resnet-50-v2-autotuning.json",
    }

    tasks = autotvm.task.extract_from_program(mod["main"], target=TARGET, params=params)

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type="rank")
        tuner_obj.tune(
            n_trial=min(tuning_option["trials"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )

    with autotvm.apply_history_best(tuning_option["tuning_records"]):
        with tvm.transform.PassContext(opt_level=3, config={}):
            lib = relay.build(mod, target=TARGET, params=params)

    dev = tvm.device(str(TARGET), 0)
    optimized_module = graph_executor.GraphModule(lib["default"](dev))

    optimized_module.set_input("input0", X_ex)
    optimized_module.run()  # dry run test

    return optimized_module
