"""Process-tree CPU, RAM and OS I/O metrics; startup and queries are separate."""
import time, threading, psutil


class Meter:
    def __init__(self):
        self.proc=psutil.Process();self.peak=0;self.stop=threading.Event()

    def snapshot(self):
        counters={'cpu_seconds':0.,'read_count':0,'write_count':0,'read_bytes':0,'write_bytes':0}
        rss=0
        for p in [self.proc]+self.proc.children(recursive=True):
            try:
                cpu=p.cpu_times();disk=p.io_counters();rss+=p.memory_info().rss
                counters['cpu_seconds']+=cpu.user+cpu.system
                for key in ['read_count','write_count','read_bytes','write_bytes']:
                    counters[key]+=getattr(disk,key)
            except psutil.Error:pass
        self.peak=max(self.peak,rss)
        return counters

    def start(self):
        self.before=self.snapshot();self.started=time.perf_counter()
        def watch():
            while not self.stop.wait(.02):self.snapshot()
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()
        return self

    def finish(self):
        elapsed=time.perf_counter()-self.started
        self.stop.set();self.thread.join();after=self.snapshot()
        out={key:max(0,after[key]-self.before[key]) for key in after}
        out.update(seconds=elapsed,peak_rss_bytes=self.peak,
                   cpu_percent_machine=100*out['cpu_seconds']/elapsed/psutil.cpu_count())
        return out
