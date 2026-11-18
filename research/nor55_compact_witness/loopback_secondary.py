"""NOR55 optional SECONDARY two-process TCP loopback producer/verifier.

This file is never imported by original first primary CPU benchmark. Two
separate processes on one host exchange length-framed signed compact witnesses
over real localhost TCP, with trusted source/root/challenges established
before the timed requests. It is NOT an off-host network benchmark.
"""
from __future__ import annotations
import argparse,json,multiprocessing,random,socket,statistics,sys,time
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.nor51_exact_r_crossover import algebra as native
from research.nor52_sparse_inverse import algebra as support
from research.nor54_matched_frontier import producer
from research.nor55_compact_witness import cohort,envelope

MAX_FRAME=8_500_000
REPEATS=3

def exact_recv(sock:socket.socket,n:int)->bytes:
    output=bytearray()
    while len(output)<n:
        packet=sock.recv(n-len(output))
        if not packet:
            raise OSError("loopback peer closed before exact requested bytes")
        output.extend(packet)
    return bytes(output)

def recv_frame(sock:socket.socket)->bytes:
    header=exact_recv(sock,4)
    size=int.from_bytes(header,"big")
    if not 1<=size<=MAX_FRAME:
        raise ValueError("invalid loopback frame length")
    return exact_recv(sock,size)

def send_frame(sock:socket.socket,body:bytes)->int:
    if type(body) is not bytes or not 1<=len(body)<=MAX_FRAME:
        raise ValueError("invalid binary loopback payload length")
    sock.sendall(len(body).to_bytes(4,"big")+body)
    return len(body)+4

def server_process(pipe,contexts):
    """Verifier child reads only trusted full source/context passed before TCP."""
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        srv.bind(("127.0.0.1",0))
        srv.listen(1)
        pipe.send(srv.getsockname()[1])
        pipe.close()
        conn,_=srv.accept()
        with conn:
            conn.settimeout(120.0)
            guard=envelope.ReplayGuard()
            for context in contexts:
                signed=recv_frame(conn)
                try:
                    values,root=envelope.verify_signed_envelope(signed,context,guard)
                    response=envelope.canonical({
                        "root_sha256":root,
                        "exact_full_coefficients_sha256":sha256(
                            producer.canonical_witness_bytes(values)).hexdigest(),
                        "session_sequence":context.sequence
                    })
                except Exception as exc:
                    response=envelope.canonical({"error":str(exc)[:300]})
                    send_frame(conn,response)
                    raise
                send_frame(conn,response)

def source_loopback(f,source_sha,stride,reference=None):
    f=support.checked_source(f,stride)
    r=native.native_single_inverse(f) if reference is None else tuple(reference)
    expected_root=native.root(r)
    expected_digest=sha256(producer.canonical_witness_bytes(r)).hexdigest()
    contexts=[
        envelope.make_context(
          source=f,source_sha256=source_sha,stride=stride,
          expected_root_sha256=expected_root,payload_format="compact-v1",
          session_id="nor55-loopback-"+source_sha[:20],sequence=i+1
        ) for i in range(REPEATS)
    ]
    mp=multiprocessing.get_context("fork")
    parent,child=mp.Pipe(duplex=False)
    server=mp.Process(target=server_process,args=(child,contexts),
                      daemon=True)
    # Startup and port negotiation are measured separately, outside the
    # three request-to-response producer-inclusive CPU wall times.
    startup=time.perf_counter_ns()
    server.start()
    child.close()
    if not parent.poll(45):
        server.terminate()
        raise RuntimeError("TCP research verifier server startup timed out")
    port=parent.recv()
    parent.close()
    startup_ms=(time.perf_counter_ns()-startup)/1e6
    wall=[]
    native_wall=[]
    req_sizes=[]
    res_sizes=[]
    with socket.create_connection(("127.0.0.1",port),timeout=120) as client:
        client.settimeout(120)
        for repeat,context in enumerate(contexts):
            native_start=time.perf_counter_ns()
            baseline=support.native_compressed_root(f,stride)
            native_ms=(time.perf_counter_ns()-native_start)/1e6
            assert baseline==(r,expected_root)
            start=time.perf_counter_ns()
            # Actual producer work in parent process, for each request.
            produced=producer.produce_compressed_full_witness(f,stride)
            signed=envelope.make_signed_envelope(produced,context)
            req_size=send_frame(client,signed)
            response=recv_frame(client)
            response_size=len(response)+4
            accepted=json.loads(response)
            if accepted!={
                "root_sha256":expected_root,
                "exact_full_coefficients_sha256":expected_digest,
                "session_sequence":repeat+1}:
                raise ValueError("loopback verifier root/vector digest/session mismatch: "+str(accepted)[:200])
            wall_ms=(time.perf_counter_ns()-start)/1e6
            wall.append(wall_ms)
            native_wall.append(native_ms)
            req_sizes.append(req_size)
            res_sizes.append(response_size)
    server.join(timeout=30)
    if server.is_alive():
        server.terminate()
        server.join(timeout=10)
        raise RuntimeError("verifier process remained active after all replies")
    if server.exitcode!=0:
        raise RuntimeError("independent verifier child process failed")
    return {
        "source_sha256":source_sha,"full_length":len(f),"stride":stride,
        "separate_producer_and_verifier_processes":True,
        "actual_TCP_transport":"127.0.0.1 loopback, same GitHub runner",
        "server_startup_ms_outside_repeats":startup_ms,
        "producer_to_verified_response_ms":wall,
        "native_compressed_same_job_baseline_ms":native_wall,
        "request_bytes":[int(x) for x in req_sizes],
        "response_bytes":[int(x) for x in res_sizes],
        "median_loopback_ms":statistics.median(wall),
        "median_native_same_job_ms":statistics.median(native_wall),
        "ratio_of_per_source_medians":
            statistics.median(wall)/statistics.median(native_wall),
        "full_exact_merkle_root":expected_root,
        "full_exact_coefficients_SHA256":expected_digest,
        "all_three_independent_verifier_responses_matched":True
    }

def run_secondary(source_file:Path,first_cpu_file:Path):
    fixture,sha=cohort.read_fixture(source_file)
    primary=json.loads(Path(first_cpu_file).read_bytes())
    assert primary["schema"]=="nor55.first_60_signed_compact_witness_cpu.v1"
    assert primary["first_public_only_source_sha256"]==sha
    assert len(primary["source_records"])==60
    selected=[r for r in fixture["cases"] if r["length"]==3073 and r["stride"]>1]
    assert len(selected)==16
    output=[]
    for i,row in enumerate(selected):
        f=tuple(int(x) for x in row["source_coefficients_decimal"])
        result=source_loopback(f,row["source_sha256"],row["stride"])
        result.update({k:row[k] for k in ("replicate","profile","family")})
        output.append(result)
        print("NOR55 SECONDARY LOOPBACK",i+1,"/16",
              row["profile"],round(result["ratio_of_per_source_medians"],3),
              flush=True)
    ratios=[r["ratio_of_per_source_medians"] for r in output]
    samples=[x for r in output for x in r["producer_to_verified_response_ms"]]
    samples.sort()
    return {
        "schema":"nor55.first_source_matched_local_loopback_secondary.v1",
        "original_first_CPU_sha256":sha256(Path(first_cpu_file).read_bytes()).hexdigest(),
        "source_only_sha256":sha,
        "source_count":len(output),
        "repeats":REPEATS,
        "producer_and_verifier_in_distinct_processes":True,
        "transport":"TCP 127.0.0.1 one runner only, not off-host",
        "server_startup_and_nonce_issuance_excluded_from_timed_requests":True,
        "producer_inverse_codec_signature_transport_and_verified_root_included":True,
        "median_sourcewise_loopback_over_same_job_native":statistics.median(ratios),
        "p95_of_48_request_response_samples_ms":samples[min(len(samples)-1,int(.95*len(samples)))],
        "source_records":output,
        "limitations":"Research-fixture Ed25519 private seed public in source. Local TCP, not deployed provider honesty, real cross-host latency or signed m3073 grammar."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--public-file",type=Path,required=True)
    p.add_argument("--first-cpu",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=run_secondary(a.public_file,a.first_cpu)
    raw=cohort.canonical(result)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_bytes(raw)
    print("NOR55 SECONDARY LOOPBACK ORIGINAL RAW BYTES SHA256",len(raw),sha256(raw).hexdigest())
    print("NOR55 LOOPBACK MEDIAN SOURCEWISE",result["median_sourcewise_loopback_over_same_job_native"])

if __name__=="__main__":
    main()
