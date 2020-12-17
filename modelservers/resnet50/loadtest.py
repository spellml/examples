from aiohttp import ClientSession, TraceConfig, ClientError
import multiprocessing
import asyncio
from dataclasses import dataclass
from pathlib import Path
import base64
import random
import statistics as stats
import time

import click


def generate_payloads(img_paths):
    payloads = []
    for image in img_paths:
        # image is stateful, so without the seek, only the first process will read anything
        image.seek(0)
        encoded = base64.b64encode(image.read()).decode("utf-8")
        payloads.append({"img": encoded})
    return payloads


@dataclass
class ProcessContext:
    num_processes: int
    process_index: int
    barrier: multiprocessing.Barrier
    latencies: multiprocessing.Array
    msg_queue: multiprocessing.Queue

    def wait(self):
        self.barrier.wait()

    def record(self, latency):
        self.latencies[self.process_index] = latency

    def percent_over(self):
        return sum(self.num_over_arr) / sum(self.total_req_arr)

    def est_median_latency(self):
        return stats.median([x for x in self.latencies if x > 0])

    def log(self, msg):
        self.msg_queue.put(msg)


def sleep_times(request_rate, process_ctx, hold_seconds):
    request_freq = 1 / request_rate
    sleep_time = request_freq * process_ctx.num_processes
    start_time = process_ctx.process_index * request_freq
    if start_time > hold_seconds:
        return
    total_time = start_time
    yield start_time
    while total_time + sleep_time <= hold_seconds:
        total_time += sleep_time
        yield total_time


async def send(session, sleep_time, url, payloads):
    await asyncio.sleep(sleep_time)
    try:
        async with session.post(url, json=random.choice(payloads), raise_for_status=True) as _:
            pass
    except ClientError as e:
        print(f"Got bad response! {e}")


async def send_batch(request_rate, process_ctx, hold_seconds, url, payloads):
    async with create_session() as session:
        await asyncio.gather(
            *[
                send(session, sleep_time, url, payloads)
                for sleep_time in sleep_times(request_rate, process_ctx, hold_seconds)
            ]
        )
        return session.latency_record


def create_session():
    async def on_request_start(session, trace_config_ctx, params):
        trace_config_ctx.start = time.time()

    async def on_request_end(session, trace_config_ctx, params):
        end = time.time()
        elapsed = end - trace_config_ctx.start
        session.latency_record.append((trace_config_ctx.start, end, elapsed))

    trace_config = TraceConfig()
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_end.append(on_request_end)
    session = ClientSession(trace_configs=[trace_config])
    session.latency_record = []
    return session


def get_median_and_record_latencies(request_rate, latency_record, out_dir, process_ctx):
    median_latency = -1
    if latency_record:
        latencies = [x[-1] for x in latency_record]
        median_latency = stats.median(latencies)
    process_ctx.record(median_latency)
    record_file = out_dir / f"{process_ctx.process_index}-{round(request_rate, 3)}"
    with record_file.open(mode="w") as f:
        for req in latency_record:
            f.write(",".join(map(str, req)) + "\n")
    process_ctx.wait()
    return process_ctx.est_median_latency() * 1000


async def load_test(process_ctx, rate_list, latency_limit, hold_seconds, out_dir, url, img_paths):
    payloads = generate_payloads(img_paths)
    process_ctx.log("Warming and Calibrating...")
    process_ctx.wait()
    warning_latency_record = await send_batch(1, process_ctx, hold_seconds, url, payloads)
    process_ctx.wait()
    if not latency_limit:
        base_latency = get_median_and_record_latencies(
            1, warning_latency_record, out_dir, process_ctx
        )
        latency_limit = 10 * base_latency
    process_ctx.log(f"Testing until latency exceeds {round(latency_limit)}ms...")
    process_ctx.wait()
    for request_rate in rate_list:
        process_ctx.wait()
        latency_record = await send_batch(request_rate, process_ctx, hold_seconds, url, payloads)
        est_median_latency = get_median_and_record_latencies(
            request_rate, latency_record, out_dir, process_ctx
        )
        process_ctx.log(
            f"Est. Median Latency for {request_rate} req/s: {round(est_median_latency)}ms"
        )
        if est_median_latency > latency_limit:
            process_ctx.log(
                f"Median Latency exceeded {round(latency_limit)}ms. Ending trial. Got to {request_rate} req/s"
            )
            break


def logger(msg_queue):
    # This is really crude, but it works
    messages = set()
    try:
        while True:
            msg = msg_queue.get()
            if msg not in messages:
                print(msg)
                messages.add(msg)
    except KeyboardInterrupt:
        return


def do_load_test(*args):
    try:
        asyncio.run(load_test(*args))
    except KeyboardInterrupt:
        return


def consolidate_records(out_dir):
    with (out_dir / "consolidated.csv").open(mode="w") as out:
        out.write("rate,proc,start,end,latency\n")
        for record in out_dir.iterdir():
            filename = record.parts[-1]
            if filename in {"consolidated.csv", ".ipynb_checkpoints"}:
                continue
            try:
                proc, rate = filename.split("-")
            except Exception as e:
                print(f"Got {e} while reading {filename}")
                continue
            with record.open() as f:
                for line in f:
                    out.write(f"{rate},{proc},{line}\n")


def generate_rates(range_spec):
    start, end, interval = map(float, range_spec.split(":"))
    rates = []
    curr = start
    while curr < end:
        rates.append(curr)
        curr += interval
    return rates


def generate_all_rates(range_specs):
    all_rates = []
    for range_spec in range_specs:
        all_rates.extend(generate_rates(range_spec))
    return sorted(list(set(all_rates)))


@click.command()
@click.option("--url", help="URL to predict endpoint to test", required=True)
@click.option(
    "--procs",
    type=int,
    default=multiprocessing.cpu_count() * 2,
    help="Number of processes to run on",
    show_default=True,
)
@click.option("--name", help="Name of this trial")
@click.option(
    "--hold-seconds",
    type=int,
    default=10,
    help="Number of seconds to hold at each request rate",
    show_default=True,
)
@click.option("--out-dir", type=click.Path(), default="./loadtest", help="Output path")
@click.option(
    "--latency-limit",
    type=int,
    default=None,
    help="The maximum latency observed before the test ends. If omitted, it is 10*(median latency of 1 req/s)",
    show_default=True,
)
@click.option(
    "--rates",
    required=True,
    help="Rates to use, specified as a comma-spearated list of half-open ranges as start:end:interval",
)
@click.option(
    "--img-path",
    "img_paths",
    type=click.File("rb"),
    multiple=True,
    required=True,
    help="Path to an image file to use in the test",
)
def main(url, name, procs, hold_seconds, out_dir, latency_limit, rates, img_paths):
    """Script to load test a model server

    This script load tests a model server with increasing request rates. At each request rate, it
    holds the request rate for an length specified by the --hold-seconds parameter. It gathers the
    latency of each request and writes them to a file named "consolidated.csv" to the directory
    {--out-dir}/{--name}. If the median latency of a single rate exceeds the --latency-limit
    (or 10 times the latency at 1 request per second), the script ends.
    """
    out_dir = Path(out_dir)
    if name:
        out_dir = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rate_list = generate_all_rates(rates.split(","))
    barrier = multiprocessing.Barrier(procs)
    latency_arr = multiprocessing.Array("d", [-1] * procs)
    msg_queue = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(
            target=do_load_test,
            args=(
                ProcessContext(procs, i, barrier, latency_arr, msg_queue),
                rate_list,
                latency_limit,
                hold_seconds,
                out_dir,
                url,
                img_paths,
            ),
        )
        for i in range(procs)
    ]
    try:
        logger_proc = multiprocessing.Process(target=logger, args=(msg_queue,))
        logger_proc.start()
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        logger_proc.terminate()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nConsolidating data...")
        consolidate_records(out_dir)
        print("DONE!")


if __name__ == "__main__":
    main()
