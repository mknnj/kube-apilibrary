import requests, os, datetime
import xml.etree.ElementTree as ET

class KubeProfiler(object):
    def __init__(self, job_manager_hostname = "10.106.102.22", db_hostname = "10.106.72.44"):
        self.jmHostname = job_manager_hostname
        self.dbHostname = db_hostname
        try:
            self.JOBID = os.getenv("JOB_UUID")
            self.PROFILE_EACH = int(os.getenv("HEARTBEAT_FREQ"))
            self.RANK = int(os.getenv("NODE_RANK"))
            self.NODE_LIST = os.getenv("NODELIST")
            self.NUM_GPUS = int(os.getenv("NUM_GPUS"))
            self.NUM_NODES = int(os.getenv("NUM_NODES"))
            restarted = os.getenv("JOB_RESTARTED")
            self.RESTARTED =  not (restarted == "FALSE")
            self.BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
            self.EPOCHS = int(os.getenv("EPOCHS_TO_BE_DONE"))
            self.EPOCHS_DONE = int(os.getenv("EPOCHS_DONE"))
            self.TRAINING_SET_SIZE = int(os.getenv("TRAINING_SET_SIZE"))
        except Exception as ex:
            print(ex)
            print("Some env variables were not found, please set them manually or run the job by submitting it to the kubernetes scheduler")
        try:
            self.LOCAL_RANK = int(os.getenv("LOCAL_RANK"))
        except Exception:
            #print("INFO: LOCAL_RANK not found")
            self.LOCAL_RANK = -1
        self.mean_iteration_duration = 0
        self.iterations = 0
        self.epoch = 0
        self.iterationPerEpoch = int(self.TRAINING_SET_SIZE / (self.BATCH_SIZE * self.NUM_GPUS * self.NUM_NODES))
        self.maxIterations = self.iterationPerEpoch*self.EPOCHS
        self.last_time = None

        self.root_dir = os.path.join("/snapshots", self.JOBID)
        if self.RESTARTED:
            try:
                checkpoints_dir = os.path.join(self.root_dir, "lightning_logs/version_0/checkpoints")
                _, _, filenames = next(os.walk(checkpoints_dir), (None, None, []))
                self.ckpt_path = os.path.join(checkpoints_dir, filenames[0])
            except:
                #print("Pytorch Lightning checkpoint folder not found, ckpt_path is root_dir")
                self.ckpt_path = self.root_dir + "/tf_ckpt"
        elif not os.path.isdir(self.root_dir):
            os.mkdir(self.root_dir)
        self.start_epoch_time = None
        self.end_epoch_time = None
        
    
    def measure(self):
        now = datetime.datetime.now()
        if self.last_time:
             iter_duration = (now - self.last_time).total_seconds()
             self.last_time = now
             self.iterations += 1
             self.mean_iteration_duration += (iter_duration - self.mean_iteration_duration) / self.iterations
        else:
             self.last_time = now
        if self.iterations % self.PROFILE_EACH == 0 and self.iterations>0:
             self.send_profiling()

    def _measure_mem(self):
        #call to nvidia-smi (quite heavy)
        res = ET.fromstring(os.popen('nvidia-smi -q -x').read())
        mypid = os.getpid()
        for pinfo in res.findall("gpu/processes/process_info"):
            if pinfo.find("pid").text == str(mypid):
                memOcc = int(pinfo.find("used_memory").text.rstrip(" MiB"))
        return memOcc

    def send_profiling(self):
        profilingData = {
        "jobid" : self.JOBID,
        "timestamp" : datetime.datetime.now().isoformat(),
        "iternum" : self.iterations,
        "nodelist" : self.NODE_LIST,
        "numgpu" : self.NUM_GPUS,
        "rank" : self.RANK,
        "meandur" : self.mean_iteration_duration,
        "gpumem" : self._measure_mem()
        }
        #print("SENDING: ", profilingData, "to", self.dbHostname)
        requests.post("http://"+self.dbHostname+"/profiling/"+self.JOBID, json=profilingData)

    def start_epoch(self):
        self.start_epoch_time = datetime.datetime.now()

    def end_epoch(self):
        self.end_epoch_time = datetime.datetime.now()
        totalEpochDur = (self.end_epoch_time  - self.start_epoch_time).total_seconds()
        
        data = {
        "jobid" : self.JOBID,
        "timestamp" : datetime.datetime.now().isoformat(),
        "iternum" : self.iterations,
        "nodelist" : self.NODE_LIST,
        "numgpu" : self.NUM_GPUS,
        "rank" : self.RANK,
        "meandur" : self.mean_iteration_duration,
        "gpumem" : self._measure_mem(),
        "epochDur" : totalEpochDur
        }

        requests.post("http://"+self.dbHostname+"/profiling/"+self.JOBID, json=data)
        r = requests.get("http://"+self.jmHostname+"/has_to_stop/"+self.JOBID)
        if r.text == "1" and self.epoch < self.EPOCHS-1:
            exit()
        self.epoch+=1

    def end(self):
        if self.LOCAL_RANK == 0 or self.LOCAL_RANK==-1:
            res = requests.post("http://"+self.jmHostname+"/signal_end/"+self.JOBID)

