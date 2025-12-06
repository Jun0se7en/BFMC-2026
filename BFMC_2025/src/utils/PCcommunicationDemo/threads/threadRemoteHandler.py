from twisted.internet import reactor

from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    Cars,
    EnableButton,
    Location,
    Recording,
    Semaphores,
    SignalRunning,
    serialCamera,
)
from src.utils.PCcommunicationDemo.threads.connection import FactoryDealer
from src.utils.PCcommunicationDemo.threads.periodics import PeriodicTask


class threadRemoteHandler(ThreadWithStop):
    """Thread which will handle processPCcommunicationDemo functionalities. We will initailize a reactor with the factory class.

    Args:
        queueList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logging (logging object): Made for debugging.
        pipeRecv (multiprocessing.pipes.Pipe): The receiving pipe. This pipe will get the information from process gateway.
        pipeSend (multiprocessing.pipes.Pipe): The sending pipe. This pipe will be sent to process gateway as a way to send information.
    """

    # ===================================== INIT =====================================
    def __init__(self, queuesList, logging, pipeRecv, pipeSend):
        super(threadRemoteHandler, self).__init__()
        self.factory = FactoryDealer(queuesList)
        self.reactor = reactor
        self.reactor.listenTCP(5001, self.factory)
        self.queues = queuesList
        self.logging = logging
        self.pipeRecv = pipeRecv
        self.pipeSend = pipeSend
        self.subscribe(pipeSend)
        self.task = PeriodicTask(
            self.factory, 0.2, self.pipeRecv, self.pipeSend, self.queues
        )

    def subscribe(self, pipeSend):
        """Subscribing function

        Args:
            pipeSend (multiprocessing.pipes.Pipe): The sending pipe
        """
        self.queues["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": serialCamera.Owner.value,
                "msgID": serialCamera.msgID.value,
                "To": {"receiver": "threadRemoteHandler", "pipe": pipeSend},
            }
        )
        self.queues["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Cars.Owner.value,
                "msgID": Cars.msgID.value,
                "To": {"receiver": "threadRemoteHandler", "pipe": pipeSend},
            }
        )
        self.queues["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Semaphores.Owner.value,
                "msgID": Semaphores.msgID.value,
                "To": {"receiver": "threadRemoteHandler", "pipe": pipeSend},
            }
        )
        self.queues["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": EnableButton.Owner.value,
                "msgID": EnableButton.msgID.value,
                "To": {"receiver": "threadRemoteHandler", "pipe": pipeSend},
            }
        )
        self.queues["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": SignalRunning.Owner.value,
                "msgID": SignalRunning.msgID.value,
                "To": {"receiver": "threadRemoteHandler", "pipe": pipeSend},
            }
        )
        self.queues["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Recording.Owner.value,
                "msgID": Recording.msgID.value,
                "To": {"receiver": "threadRemoteHandler", "pipe": pipeSend},
            }
        )
        self.queues["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Location.Owner.value,
                "msgID": Location.msgID.value,
                "To": {"receiver": "threadRemoteHandler", "pipe": pipeSend},
            }
        )

    # ===================================== RUN ======================================
    def run(self):
        self.task.start()
        print("before run")
        self.reactor.run(installSignalHandlers=False)
        print("after run")

    # ==================================== STOP ======================================
    def stop(self):
        self.reactor.stop()
        super(threadRemoteHandler, self).stop()
