if __name__ == "__main__":
    import sys

    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
# from src.decisionMaking.threads.threadDecisionMaking2 import threadDecisionMaking
from src.decisionMaking.threads.threadDecisionMaking import threadDecisionMaking
from multiprocessing import Pipe

class processDecisionMaking(WorkerProcess):
    def __init__(self, queueList, logging, Speed, Steer, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        pipeRecv, pipeSend = Pipe(duplex=False)
        self.pipeRecv = pipeRecv
        self.pipeSend = pipeSend
        self.Speed, self.Steer = Speed, Steer
        self.debugging = debugging
        super(processDecisionMaking, self).__init__(self.queuesList)
        
        # ===================================== STOP ==========================================
    def stop(self):
        """Function for stopping threads and the process."""
        for thread in self.threads:
            thread.stop()
            thread.join()
        super(processDecisionMaking, self).stop()

    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processDecisionMaking, self).run()

    # ===================================== INIT TH ======================================
    def _init_threads(self):
        """Create the Segmentation Publisher thread and add to the list of threads."""
        Th = threadDecisionMaking(
            self.pipeRecv, self.pipeSend, self.queuesList, self.logging, self.Speed, self.Steer, self.debugging
        )
        self.threads.append(Th)
        