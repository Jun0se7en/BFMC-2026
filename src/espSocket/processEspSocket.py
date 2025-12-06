from src.templates.workerprocess import WorkerProcess
from src.espSocket.threads.threadEspSocket import threadEspSocket
from multiprocessing import Pipe
from threading import Event
from multiprocessing.sharedctypes import Value
class processEspSocket(WorkerProcess):
    """This process decide car speed and angle\n
    Args:
        queueList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
        example (bool, optional): A flag for running the example. Defaults to False.
    """

    # ===================================== INIT =========================================
    def __init__(self, queueList, logging, ip_address, port, gps_flag, debugging=False):
        self.queuesList = queueList
        self.logging = logging
        pipeRecv, pipeSend = Pipe(duplex=False)
        self.pipeRecv = pipeRecv
        self.pipeSend = pipeSend
        self.ip_address = ip_address
        self.port = port
        self.gps_flag = gps_flag
        self.debugging = debugging
        super(processEspSocket, self).__init__(self.queuesList)

    # ===================================== STOP ==========================================
    def stop(self):
        """Function for stopping threads and the process."""
        for thread in self.threads:
            thread.stop()
            thread.join()
        super(processEspSocket, self).stop()

    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processEspSocket, self).run()

    # ===================================== INIT TH =================================
    def _init_threads(self):
        """Initializes the read and the write thread."""

        EspSocketTh = threadEspSocket(self.pipeRecv, self.pipeSend, self.queuesList, self.logging, self.ip_address, self.port, self.gps_flag, self.debugging)
        self.threads.append(EspSocketTh)
