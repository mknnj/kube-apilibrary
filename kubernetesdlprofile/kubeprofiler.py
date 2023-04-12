import requests, os, datetime
import xml.etree.ElementTree as ET
import threading
import sys

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
        self.epoch = self.EPOCHS_DONE
        if self.epoch is None:
            raise Exception("ENV EPOCHS_DONE not set")
        self.iterationPerEpoch = int(self.TRAINING_SET_SIZE / (self.BATCH_SIZE * self.NUM_GPUS * self.NUM_NODES))
        self.maxIterations = self.iterationPerEpoch*self.EPOCHS
        self.last_time = None
        self.has_to_stop = False

        self.root_dir = os.path.join("/snapshots", self.JOBID)
        if self.RESTARTED:
            try:
                _, _, versions = next(os.walk(os.path.join(self.root_dir, "lightning_logs")), (None, None, []))
                print(versions)
                last_version = max([int(v.split("_")[1]) for v in versions])
                print("[KUBE_API_LIBRARY] last version found is ", last_version)
                checkpoints_dir = os.path.join(self.root_dir, "lightning_logs/version_"+str(last_version)+"/checkpoints")
                _, _, filenames = next(os.walk(checkpoints_dir), (None, None, []))
                self.ckpt_path = os.path.join(checkpoints_dir, filenames[0])
            except Exception as e:
                print(e)
                #print("Pytorch Lightning checkpoint folder not found, ckpt_path is root_dir")
                self.ckpt_path = self.root_dir + "/tf_ckpt"
            print("[KUBE_API_LIBRARY]: Job Restarted, ckpt_path: ", self.ckpt_path)
        elif not os.path.isdir(self.root_dir):
            os.mkdir(self.root_dir)
        self.start_epoch_time = None
        self.end_epoch_time = None
        self.block_send_sem = threading.Semaphore()
        
    
    def _measure(self, now):
        if self.last_time:
             iter_duration = (now - self.last_time).total_seconds()
             self.last_time = now
             self.iterations += 1
             self.mean_iteration_duration += (iter_duration - self.mean_iteration_duration) / self.iterations
        else:
             self.last_time = now
        if self.iterations % self.PROFILE_EACH == 0 and self.iterations>0:
             self.send_profiling()
    
    def measure(self):
        now = datetime.datetime.now()
        measureThread = threading.Thread(target = self._measure, args = (now,))
        measureThread.start()
        if self.has_to_stop:
            sys.exit()

    def _measure_mem(self):
        #call to nvidia-smi (quite heavy)
        res = ET.fromstring(os.popen('nvidia-smi -q -x').read())
        mypid = os.getpid()
        memOcc = 0
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
        try:
            requests.post("http://"+self.dbHostname+"/profiling/"+self.JOBID, json=profilingData)
        except Exception as e:
            print("Can't reach scheduler db",e)

    def _start_epoch(self, now):
        self.block_send_sem.acquire()
        self.start_epoch_time = now
        try:
            r = requests.get("http://"+self.jmHostname+"/has_to_stop/"+self.JOBID)
            if r.text == "1" and self.epoch < self.EPOCHS-1:
                self.has_to_stop = True
        except Exception as e:
            print("Can't reach scheduler controller",e)
        self.block_send_sem.release()

    def start_epoch(self):
        now = datetime.datetime.now()
        #startEpochThread = threading.Thread(target = self._start_epoch, args = (now))
        #startEpochThread.start()
        self._start_epoch(now)
        

    def _end_epoch(self, now):
        self.block_send_sem.acquire()
        self.end_epoch_time = now
        totalEpochDur = (self.end_epoch_time  - self.start_epoch_time).total_seconds()
        
        data = {
        "jobid" : self.JOBID,
        "timestamp" : now.isoformat(),
        "iternum" : self.iterations,
        "nodelist" : self.NODE_LIST,
        "numgpu" : self.NUM_GPUS,
        "rank" : self.RANK,
        "meandur" : self.mean_iteration_duration,
        "gpumem" : self._measure_mem(),
        "epochDur" : totalEpochDur
        }

        try:
            requests.post("http://"+self.dbHostname+"/profiling/"+self.JOBID, json=data)
        except Exception as e:
            print("Can't reach scheduler db",e)

        self.epoch+=1

        data = {
            "done" : self.epoch
        }
        try:
            requests.post("http://"+self.jmHostname+"/end_epoch/"+self.JOBID, json=data)
        except Exception as e:
            print("Can't reach scheduler job manager, holding epoch number for later sending",e)
        self.block_send_sem.release()
        
        
        
    
    def end_epoch(self):
        now = datetime.datetime.now()
        #endEpochThread = threading.Thread(target = self._end_epoch, args = (now))
        #endEpochThread.start()
        self._end_epoch(now)

    def end(self):
        return
        #if self.LOCAL_RANK == 0 or self.LOCAL_RANK==-1:
            #res = requests.post("http://"+self.jmHostname+"/signal_end/"+self.JOBID)

