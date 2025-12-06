import logging
from twisted.internet import protocol
import src.data.TrafficCommunication.useful.keyDealer as keyDealer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class udpListener(protocol.DatagramProtocol):
    """This class will handle the connection.

    Args:
        decrypt_key (String): A path to the decripting key.
        serverfound (function): This function will be called if the server will be found.
    """

    def __init__(self, decrypt_key, serverfound):
        logging.info("Initializing udpListener")
        decrypt_key = decrypt_key
        self.pub_key = keyDealer.load_public_key(decrypt_key)
        self.serverfoundCllback = serverfound
        logging.info("udpListener initialized")

    def startProtocol(self):
        logging.info("Looking for Traffic Communication Server")

    def datagramReceived(self, datagram, address):
        """In this function we split the receive data and we call the callback function"""
        logging.info("Datagram received")
        try:
            dat = datagram.split(b"(-.-)")
            if len(dat) != 2:
                raise Exception("Plaintext or signature not present")
            a = keyDealer.verify_data(self.pub_key, dat[1], dat[0])
            if not a:
                raise Exception("Signature not valid")
            msg = dat[1].decode().split(":")
            port = int(msg[1])
            self.serverfoundCllback(address[0], port)
            logging.info(f"Server found at {address[0]}:{port}")
        except Exception as e:
            logging.error("TrafficCommunication -> udpListener -> datagramReceived:")
            logging.error(e)

    def stopListening(self):
        logging.info("Stopping UDP listener")
        self.transport.stopListening()