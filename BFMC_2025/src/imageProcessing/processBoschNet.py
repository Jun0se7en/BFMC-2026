if __name__ == "__main__":
    import sys

    sys.path.insert(0, "../../..")

from src.templates.workerprocess import WorkerProcess
from src.imageProcessing.threads.threadBoschNet4 import threadBoschNet
from multiprocessing import Pipe


class processBoschNet(WorkerProcess):
    """This process handle camera.\n
    Args:
            queueList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
            logging (logging object): Made for debugging.
            debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    # ====================================== INIT ==========================================
    def __init__(self, queueList, logging, Speed, Steer, Reset, debugging=False):
        self.queuesList = queueList
        self.Reset = Reset
        self.logging = logging
        pipeRecv, pipeSend = Pipe(duplex=False)
        self.pipeRecv = pipeRecv
        self.pipeSend = pipeSend
        self.Speed, self.Steer = Speed, Steer
        self.debugging = debugging
        super(processBoschNet, self).__init__(self.queuesList)
        # print('Initialize camera process!!!')

    # ===================================== STOP ==========================================
    def stop(self):
        """Function for stopping threads and the process."""
        for thread in self.threads:
            thread.stop()
            thread.join()
        super(processBoschNet, self).stop()

    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processBoschNet, self).run()

    # ===================================== INIT TH ======================================
    def _init_threads(self):
        """Create the Segmentation Publisher thread and add to the list of threads."""
        Th = threadBoschNet(
            self.pipeRecv, self.pipeSend, self.queuesList, self.logging, self.Speed, self.Steer, self.Reset, self.debugging
        )
        self.threads.append(Th)
        