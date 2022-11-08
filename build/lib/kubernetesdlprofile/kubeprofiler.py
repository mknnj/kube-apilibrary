import requests, os, datetime

class KubeProfiler(object):
    def __init__(self, iterationsToMeasure, ipAddress = "10.106.102.22"):
        self.ip = ipAddress
        self.jobID = os.getenv("JOB_UUID")
        self.configID = os.getenv("JOB_CONFIGID")
        self.measuredMem = 0
        self.meanIterationDuration = 0
        self.meanGPUUsage = 0
        self.iterations = 0
        self.iterationsToMeasure = iterationsToMeasure
        self.lastTime = None
    
    def measure(self):
        if self.iterations > self.iterationsToMeasure:
            return
        now = datetime.datetime.now()
        if self.lastTime:
            iterDuration = (now - self.lastTime).total_seconds()
            print("iter dur: ", iterDuration)
            self.lastTime = now
            self.iterations += 1
            #self.meanIterationDuration += (iterDuration - self.meanIterationDuration) / self.iterations

            gpu_data = self._measure_mem()
            self.measuredMem = max(self.measuredMem, gpu_data[0])
            self.meanGPUUsage += (gpu_data[1] - self.meanGPUUsage) / self.iterations
        else:
            self.lastTime = now

    def _measure_mem(self):
        #call to nvidia-smi or similar
        res = os.popen('nvidia-smi -x').read()
        print(res)
        memOcc = 0
        memUsage = 0
        return (memOcc, memUsage)

    def end_profiling(self):
        profilingData = {
        "meanIterDuration" : self.meanIterationDuration,
        "endTime" : datetime.datetime.now(),
        "gpuUsage" : self.measuredMem, 
        "gpuMemUsage" : self.meanGPUUsage
        }
        requests.post("http://"+self.ip+"/signal_end_profiling/"+self.jobID+"/"+self.configID, profilingData)
    
    def end(self):
        res = requests.post("http://10.106.102.22/signal_end/"+self.jobID)
        
