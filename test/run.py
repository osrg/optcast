import os
import subprocess
import argparse
import asyncio
import yaml
import platform
import sys
import os
import re
import argparse
import datetime
from functools import reduce

epoch = 0


SERVER_CMD = "reduction_server/target/release/optcast-reduction-server"
CLIENT_CMD = "test/nccl-tests/build/all_reduce_perf"
OPTCAST_PLUGIN_DIR = "nccl_plugin/src/.libs"


def show_stats(prefix, v):
    avg = sum(v) / len(v)
    # calculate standard deviation
    sd = 0
    for i in v:
        sd += (i - avg) ** 2
    sd = (sd / len(v)) ** 0.5
    # calculate median
    v.sort()
    median = v[len(v) // 2]

    print(
        f"{prefix} len: {len(v)}, avg: {avg:.2f}, sd: {sd:.2f}, median: {median:.2f}, min: {min(v):.2f}, max: {max(v):.2f}"
    )


def plot(data, classes, colormapping, xlim, output):
    # load matplotlib only when needed
    # otherwise, matplotlib needs to be installed on all nodes
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    cats = {}
    yticks = []
    yticklabels = []
    for i, k in enumerate(sorted(classes)):
        tick = i + 1
        label = k
        yticks.append(tick)
        yticklabels.append(label)
        cats[label] = tick

    verts = []
    colors = []
    for d in data:
        v = [
            (d[0], cats[d[2]] - 0.4),
            (d[0], cats[d[2]] + 0.4),
            (d[1], cats[d[2]] + 0.4),
            (d[1], cats[d[2]] - 0.4),
            (d[0], cats[d[2]] - 0.4),
        ]
        verts.append(v)
        colors.append(colormapping[d[3]])
    bars = PolyCollection(verts, facecolors=colors)
    _, ax = plt.subplots(figsize=(20, 10))
    ax.add_collection(bars)
    ax.autoscale()
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("time (ms)")
    if xlim:
        if "," in xlim:
            xlim = xlim.split(",")
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        else:
            ax.set_xlim(float(xlim))
    plt.savefig(output)


def analyze_client_log(filename, output, xlim=None, no_plot=False):
    global epoch
    epoch = 0

    with open(filename) as f:
        log = f.read()

    data = []
    classes = {}
    colormapping = {}
    stats = {}

    v = []

    def get_time(line):
        global epoch
        t = line.split(" ")[2]
        try:
            ts = float(t)
        except ValueError:
            r = re.search(r"(?P<t>\d+.\d+) \w+:\d+ NCCL TRACE", line)
            ts = float(r.group("t"))
        if epoch == 0:
            epoch = ts
        return float(ts - epoch)

    avgBusBw = 0

    for line in log.split("\n"):
        if line.startswith("#"):
            print(line)
            if "Avg bus bandwidth" in line:
                avgBusBw = float(line.strip().split(" ")[-1])
        r = re.search(r"req\((?P<req>.+?)\)", line)
        req = int(r.group("req"), 16) if r else None
        r = re.search(r"idx\((?P<idx>\d)\)", line)
        idx = int(r.group("idx")) if r else None

        start = False
        end = False
        cname = ""
        _, reqname = classes.get(req, (None, f"req({len(classes)})"))

        if "allreduce start" in line:
            start = True
            s = stats.get(req, [])
            s.append({"start": get_time(line)})
            stats[req] = s
        elif "allreduce requested" in line:
            start = True
            end = True
            cname = "requested"
            stats[req][-1]["requested"] = get_time(line)
        elif "send done" in line:
            start = True
            end = True
            cname = "send done"
            stats[req][-1]["send done"] = get_time(line)
        elif "recv done" in line:
            end = True
            cname = "recv done"
            stats[req][-1]["recv done"] = get_time(line)

        if end:
            e = get_time(line)
            s, reqname = classes[req]
            data.append((s, e, reqname, cname))
            ckey = cname
            if ckey not in colormapping:
                colormapping[ckey] = f"C{len(colormapping)}"

        if start:
            s = get_time(line)
            classes[req] = (s, reqname)

    # cut off first-half of the data to remove warmup phase
    for k, v in stats.items():
        stats[k] = v[len(v) // 2 :]

    vv = reduce(lambda a, b: a + b, stats.values(), [])
    if not len(vv):
        return

    print("client stats:")
    show_stats(f"  e2e ", [s["recv done"] - s["start"] for s in vv])
    show_stats(f"  req ", [s["requested"] - s["start"] for s in vv])
    show_stats(f"  send", [s["send done"] - s["requested"] for s in vv])
    show_stats(f"  recv", [s["recv done"] - s["send done"] for s in vv])
    print("")

    if no_plot:
        return avgBusBw

    classes = (r[1] for r in classes.values())
    plot(data, classes, colormapping, xlim, output)

    return avgBusBw


def analyze_server_log(filename, output, xlim=None, no_plot=False):
    global epoch
    epoch = 0

    with open(filename) as f:
        log = f.read()

    data = []

    classes = {}
    stats = {}

    epoch = 0

    def get_time(line):
        global epoch

        t = datetime.datetime.fromisoformat(line.split(" ")[1][1:])
        # datetime to unixnano
        ts = t.timestamp() * 1000

        if epoch == 0:
            epoch = ts
        return float(ts - epoch)

    for line in log.split("\n"):
        j = re.search(r"job\((?P<job>\d+)\)", line)
        if j:
            job = j.group("job")

        r = re.search(r"rank\((?P<rank>\d+)\)", line)
        if r:
            rank = r.group("rank")

        start = False
        end = False

        if "recv requested" in line:
            start = True
            name = f"recv({rank}/{job})"
        elif "recv latency" in line:
            end = True
            name = f"recv({rank}/{job})"
            statname = "recv"
        elif "send requested" in line:
            start = True
            name = f"send({rank}/{job})"
        elif "send latency" in line:
            end = True
            name = f"send({rank}/{job})"
            statname = "send"
        elif "convert start" in line:
            start = True
            name = f"convert({rank}/{job})"
        elif "convert latency" in line:
            end = True
            name = f"convert({rank}/{job})"
            statname = "convert"
        elif "reduce start" in line:
            start = True
            name = f"reduce({rank})/({job})"
        elif "reduce latency" in line:
            end = True
            name = f"reduce({rank})/({job})"
            statname = "reduce"

        if start:
            if name not in classes:
                classes[name] = {}
            classes[name] = get_time(line)
        elif end:
            e = get_time(line)
            data.append((classes[name], e, name, name))
            stat = stats.get(statname, [])
            stat.append(e - classes[name])
            stats[statname] = stat

    # cut off first-half of the data to remove warmup phase
    for k, v in stats.items():
        stats[k] = v[len(v) // 2 :]

    print("server stats:")
    for k, v in stats.items():
        show_stats("  " + k, v)
    print("")

    if no_plot:
        return

    classes = classes.keys()
    colormapping = {k: f"C{i}" for i, k in enumerate(sorted(classes))}
    plot(data, classes, colormapping, xlim, output)


def get_shared_dir():
    path = os.path.realpath(__file__)
    while os.path.islink(path):
        path = os.path.realpath(path)
    return os.path.dirname(os.path.dirname(path))


def gen_args(args):
    ret = []
    for k, v in args._get_kwargs():
        if not v:
            continue
        if v == True:
            ret.append(f"--{k.replace('_', '-')}")
        else:
            if type(v) == str and " " in v:
                v = f'"{v}"'
            ret.append(f"--{k.replace('_', '-')} {v}")
    return " ".join(ret)


def parse_chunksize(chunksize):
    if chunksize.endswith("M"):
        return int(chunksize[:-1]) * 1024 * 1024
    elif chunksize.endswith("K"):
        return int(chunksize[:-1]) * 1024
    else:
        return int(chunksize)


async def server(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    server = config["servers"][rank]
    port = server["port"]
    count = (
        parse_chunksize(args.chunksize)
        // (4 if args.data_type == "f32" else 2)
        // args.nsplit
    )

    env = server["env"]
    if "NCCL_DEBUG" not in env:
        env["NCCL_DEBUG"] = "TRACE" if rank == 0 else "INFO"
    if "RUST_LOG" not in env:
        env["RUST_LOG"] = "TRACE" if rank == 0 else "INFO"
    if "LD_LIBRARY_PATH" not in env:
        env["LD_LIBRARY_PATH"] = f"{args.shared_dir}/{OPTCAST_PLUGIN_DIR}"

    server_cmd = f"{args.shared_dir}/{SERVER_CMD}"
    cmd = (
        f"{server_cmd} --port {port} --nrank {args.nrank}"
        + f" --reduce-jobs {args.num_jobs} --reduce-threads {args.num_threads} --recv-threads {args.num_recvs} --send-threads {args.num_sends}"
        + f" --count {count}"
        + f" --data-type {args.data_type}"
    )
    # print(f"[{platform.node()}] server:", cmd, file=sys.stderr)

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    asyncio.create_task(
        read_stream(
            proc.stdout,
            platform.node(),
            None,
            True,
            sys.stdout,
        )
    )
    asyncio.create_task(
        read_stream(
            proc.stderr,
            f"{platform.node()}/err",
            None,
            True,
            sys.stdout,
        )
    )

    await proc.wait()


async def client(args):
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    dt = "float" if args.data_type == "f32" else "half"
    client_cmd = f"{args.shared_dir}/{CLIENT_CMD}"
    cmd = f"{client_cmd} -d {dt} -e {args.size} -b {args.size} {args.nccl_test_options}"
    # print(f"[{platform.node()}] client:", cmd, file=sys.stderr)

    os.environ["NCCL_DEBUG"] = "TRACE" if rank == 0 else "INFO"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_SHM_DISABLE"] = "1"

    if args.type == "optcast":
        os.environ["NCCL_COLLNET_ENABLE"] = "1"
        os.environ["LD_LIBRARY_PATH"] = (
            f"{args.shared_dir}/{OPTCAST_PLUGIN_DIR}:{os.environ['LD_LIBRARY_PATH']}"
        )
        os.environ["OPTCAST_REDUCTION_SERVERS"] = args.reduction_servers
        os.environ["NCCL_BUFFSIZE"] = str(64 * 1024 * 1024)
        chunksize = parse_chunksize(args.chunksize) // 2
        os.environ["NCCL_COLLNET_CHUNKSIZE"] = str(chunksize)
        os.environ["OPTCAST_SPLIT"] = str(args.nsplit)
    elif args.type == "sharp":
        os.environ["NCCL_COLLNET_ENABLE"] = "1"

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    )

    asyncio.create_task(
        read_stream(
            proc.stdout,
            None,
            None,
            False,
            sys.stdout,
        )
    )
    asyncio.create_task(
        read_stream(
            proc.stderr,
            f"{platform.node()}/err",
            None,
            False,
            sys.stdout,
        )
    )

    await proc.wait()


async def read_stream(stream, prefix, filename, p, out=sys.stdout):
    if filename:
        with open(filename, "w") as f:
            while True:
                line = await stream.readline()
                if not line:
                    break
                if p:
                    print(f"{prefix}: {line.decode().strip()}", file=out)
                f.write(line.decode())
    else:
        while True:
            line = await stream.readline()
            if not line:
                break
            if prefix:
                print(f"[{prefix}] {line.decode().strip()}", file=out)
            else:
                print(f"{line.decode().strip()}", file=out)


async def run(
    args,
):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.nrank == 0:
        args.nrank = len(config["clients"])
    if args.nservers == 0:
        args.nservers = min(len(config["servers"]), args.nrank)

    servers = config["servers"][: args.nservers]
    clients = config["clients"][: args.nrank]

    reduction_servers = ",".join(f"{s['ipaddr']}:{s['port']}" for s in servers)
    cmd = " ".join(
        (
            args.mpirun,
            f"-np {args.nrank} -H {','.join(c['name'] for c in clients)}",
            "-x LD_LIBRARY_PATH",
            f"{args.python} {args.shared_dir}/test/run.py",
            gen_args(args),
            f"--client --reduction-servers {reduction_servers}",
        )
    )
    client = await asyncio.create_subprocess_shell(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    )

    # make args.log_dir if not exists
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    with open(f"{args.log_dir}/setting.yaml", "w") as f:
        f.write(yaml.dump(args.__dict__))

    asyncio.create_task(
        read_stream(
            client.stdout,
            "client stdout",
            f"{args.log_dir}/client.log",
            args.verbose,
        )
    )
    asyncio.create_task(
        read_stream(
            client.stderr,
            "client stderr",
            f"{args.log_dir}/client_err.log",
            True,
        )
    )

    if args.type == "optcast":
        cmd = " ".join(
            (
                args.mpirun,
                f"-bind-to none -np {args.nservers} -H {','.join(s['name'] for s in servers)}",
                f"{args.python} {args.shared_dir}/test/run.py",
                gen_args(args),
                "--server",
            )
        )
        server = await asyncio.create_subprocess_shell(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        )
        asyncio.create_task(
            read_stream(
                server.stdout,
                "server stdout",
                f"{args.log_dir}/server.log",
                args.verbose,
            )
        )
        asyncio.create_task(
            read_stream(
                server.stderr,
                "server stderr",
                f"{args.log_dir}/server_err.log",
                True,
            )
        )

    try:
        await asyncio.wait_for(client.wait(), 40)
    except asyncio.TimeoutError:
        client.terminate()

    if args.type != "optcast":
        return analyze_client_log(
            args.log_dir + "/client.log",
            args.log_dir + "/client.png",
            args.xlim,
            args.no_plot,
        )

    server.terminate()
    await asyncio.wait_for(server.wait(), 40)

    analyze_server_log(
        args.log_dir + "/server.log",
        args.log_dir + "/server.png",
        args.xlim,
        args.no_plot,
    )

    return analyze_client_log(
        args.log_dir + "/client.log",
        args.log_dir + "/client.png",
        args.xlim,
        args.no_plot,
    )


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--size", default="512M")
    parser.add_argument("--chunksize", default="512K")
    parser.add_argument("--num-jobs", default=2, type=int)
    parser.add_argument("--num-threads", default=2, type=int)
    parser.add_argument("--num-sends", default=4, type=int)
    parser.add_argument("--num-recvs", default=4, type=int)
    parser.add_argument("--nrank", default=0, type=int)
    parser.add_argument("--nservers", default=0, type=int)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--nsplit", default=1, type=int)
    parser.add_argument("--reduction-servers")
    parser.add_argument(
        "--type", choices=["optcast", "sharp", "nccl"], default="optcast"
    )
    parser.add_argument("--nccl-test-options", default="-c 1 -n 1 -w 1")
    parser.add_argument("--data-type", default="f32", choices=["f32", "f16"])
    parser.add_argument("--shared-dir", default=get_shared_dir())
    parser.add_argument("--log-dir", default="log")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="config file path. must be located under the shared-dir",
    )
    parser.add_argument("--analyze", "-a", action="store_true")
    parser.add_argument("--no-plot", "-p", action="store_true")
    parser.add_argument("--xlim", "-x")

    return parser.parse_args()


def main():
    args = arguments()
    args.config = os.path.realpath(args.config)
    args.shared_dir = os.path.realpath(args.shared_dir)
    # check args.config is located under args.shared_dir
    if not args.config.startswith(args.shared_dir):
        print(f"config file must be located under {args.shared_dir}")

    if args.analyze:
        if os.stat(args.log_dir + "/client.log"):
            analyze_client_log(
                args.log_dir + "/client.log", args.log_dir + "/client.png", args.xlim
            )

        if os.stat(args.log_dir + "/server.log"):
            analyze_server_log(
                args.log_dir + "/server.log", args.log_dir + "/server.png", args.xlim
            )
        return

    if args.server:
        asyncio.run(server(args))
    elif args.client:
        asyncio.run(client(args))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
