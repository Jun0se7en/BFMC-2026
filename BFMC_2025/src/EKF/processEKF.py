from src.templates.workerprocess import WorkerProcess
# from src.EKF.threads.threadEKF import threadEKF
from src.EKF.threads.threadWitMotion import threadWitMotion
from src.EKF.threads.threadMPC import threadMPC
from multiprocessing import Pipe
from threading import Event
from multiprocessing.sharedctypes import Value
class processEKF(WorkerProcess):
    """This process decide car speed and angle\n
    Args:
        queueList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
        example (bool, optional): A flag for running the example. Defaults to False.
    """

    # ===================================== INIT =========================================
    def __init__(self, queueList, Speed, Steer, logging, debugging=False):
        self.queuesList = queueList
        self.Speed = Speed
        self.Steer = Steer
        self.logging = logging
        pipeRecv, pipeSend = Pipe(duplex=False)
        self.pipeRecv = pipeRecv
        self.pipeSend = pipeSend
        self.debugging = debugging
        super(processEKF, self).__init__(self.queuesList)

    # ===================================== STOP ==========================================
    def stop(self):
        """Function for stopping threads and the process."""
        for thread in self.threads:
            thread.stop()
            thread.join()
        super(processEKF, self).stop()

    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processEKF, self).run()

    # ===================================== INIT TH =================================
    def _init_threads(self):
        """Initializes the read and the write thread."""

        WMTh = threadWitMotion(self.pipeRecv, self.pipeSend, self.queuesList, self.Speed, self.Steer, self.logging, self.debugging)
        self.threads.append(WMTh)

        # MPCTh = threadMPC(self.pipeRecv, self.pipeSend, self.queuesList, self.logging, self.debugging)
        # self.threads.append(MPCTh)
