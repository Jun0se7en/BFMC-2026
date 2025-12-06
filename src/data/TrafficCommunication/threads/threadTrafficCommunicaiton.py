import logging
from twisted.internet import reactor
from src.templates.threadwithstop import ThreadWithStop
from src.data.TrafficCommunication.threads.udpListener import udpListener
from src.data.TrafficCommunication.threads.tcpClient import tcpClient
from src.data.TrafficCommunication.useful.periodicTask import periodicTask

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class threadTrafficCommunication(ThreadWithStop):
    """Thread which will handle processTrafficCommunication functionalities

    Args:
        shrd_mem (sharedMem): A space in memory for mwhere we will get and update data.
        queuesList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        deviceID (int): The id of the device.
        decrypt_key (String): A path to the decription key.
    """

    # ====================================== INIT ==========================================
    def __init__(self, shrd_mem, queueslist, deviceID, frequency, decrypt_key):
        logging.info("Initializing threadTrafficCommunication")
        super(threadTrafficCommunication, self).__init__()
        self.listenPort = 9000
        self.queue = queueslist

        self.tcp_factory = tcpClient(self.serverLost, deviceID, frequency, self.queue) # Handles the connection with the server

        self.udp_factory = udpListener(decrypt_key, self.serverFound) #Listens for server broadcast and validates it

        self.period_task = periodicTask(1, shrd_mem, self.tcp_factory, queueslist) # Handles the Queue of errors accumulated so far.

        self.reactor = reactor
        self.reactor.listenUDP(self.listenPort, self.udp_factory)
        logging.info("threadTrafficCommunication initialized")

    # =================================== CONNECTION =======================================
    def serverLost(self):
        """If the server disconnects, we stop the factory listening and start the reactor listening"""
        logging.info("Server lost")
        self.reactor.listenUDP(self.listenPort, self.udp_factory)
        self.tcp_factory.stopListening()
        self.period_task.stop()

    def serverFound(self, address, port):
        """If the server was found, we stop the factory listening, connect the reactor, and start the periodic task"""
        logging.info(f"Server found at {address}:{port}")
        self.reactor.connectTCP(address, port, self.tcp_factory)
        self.udp_factory.stopListening()
        self.period_task.start()

    # ======================================= RUN ==========================================
    def run(self):
        logging.info("Starting reactor")
        self.reactor.run(installSignalHandlers=False)
        logging.info("Reactor stopped")

    # ====================================== STOP ==========================================
    def stop(self):
        logging.info("Stopping reactor")
        self.reactor.stop()
        super(threadTrafficCommunication, self).stop()
        logging.info("Reactor stopped and thread stopped")