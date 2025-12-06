import json
import time
from twisted.internet import protocol, reactor
from twisted.internet.error import TimeoutError
from src.utils.messages.allMessages import Location
from src.utils.messages.messageHandlerSender import messageHandlerSender

class tcpClient(protocol.ClientFactory):
    def __init__(self, connectionBrokenCllbck, locsysID, locsysFrequency, queue):
        self.connectiondata = None
        self.connection = None
        self.retry_delay = 1  # Initial retry delay in seconds
        self.max_retry_delay = 16  # Maximum delay to prevent infinite retries
        self.connectionBrokenCllbck = connectionBrokenCllbck
        self.locsysID = locsysID
        self.locsysFrequency = locsysFrequency
        self.queue = queue
        self.sendLocation = messageHandlerSender(self.queue, Location)

    def clientConnectionLost(self, connector, reason):
        print(f"[ERROR] Connection lost with server {self.connectiondata}: {reason}")
        self.connectiondata = None
        self.connection = None
        self.retry_connection(connector)

    def clientConnectionFailed(self, connector, reason):
        if isinstance(reason.value, TimeoutError):
            print("[ERROR] Connection failed due to timeout. Check if the server is reachable.")
        else:
            print(f"[ERROR] Connection failed: {reason}")

        print(f"[INFO] Retrying in {self.retry_delay} seconds... Possible server down or incorrect IP:port match.")
        time.sleep(self.retry_delay)
        connector.connect()


    def retry_connection(self, connector):
        """Retries connection with exponential backoff"""
        time.sleep(self.retry_delay)
        self.retry_delay = min(self.retry_delay * 2, self.max_retry_delay)  # Exponential backoff
        connector.connect()

    def buildProtocol(self, addr):
        conn = SingleConnection()
        conn.factory = self
        self.retry_delay = 1  # Reset retry delay on successful connection
        return conn

    def send_data_to_server(self, message):
        if self.connection:
            self.connection.send_data(message)
        else:
            print("[WARNING] No active connection to send data!")

class SingleConnection(protocol.Protocol):
    def connectionMade(self):
        """Handles initial connection setup"""
        peer = self.transport.getPeer()
        self.factory.connectiondata = f"{peer.host}:{peer.port}"
        self.factory.connection = self
        print(f"[INFO] Connected to server: {self.factory.connectiondata}")

        # Subscribe to location data
        self.subscribeToLocationData(self.factory.locsysID, self.factory.locsysFrequency)

    def dataReceived(self, data):
        """Processes received data from the server"""
        try:
            dat = data.decode()
            tmp_data = dat.replace("}{", "}}{{")  # Fix incorrectly formatted JSON
            if tmp_data != dat:
                tmp_dat = tmp_data.split("}{")
                dat = tmp_dat[-1]

            da = json.loads(dat)

            if da["type"] == "location":
                da["id"] = self.factory.locsysID
                self.factory.sendLocation.send(da)
            else:
                print(f"[INFO] Received message from server: {self.factory.connectiondata}")
        except json.JSONDecodeError:
            print("[ERROR] Failed to decode JSON data from server.")
        except Exception as e:
            print(f"[ERROR] Unexpected error while processing data: {e}")

    def send_data(self, message):
        """Encodes and sends JSON data to the server"""
        try:
            msg = json.dumps(message)
            self.transport.write(msg.encode())
        except Exception as e:
            print(f"[ERROR] Failed to send data: {e}")

    def subscribeToLocationData(self, id, frequency):
        """Subscribes to location data with a specified frequency"""
        if 0.1 <= frequency <= 5:
            msg = {
                "reqORinfo": "info",
                "type": "locIDsub",
                "locID": id,
                "freq": frequency,
            }
            self.send_data(msg)
        else:
            print("[ERROR] Invalid frequency value. Must be between 0.1 and 5.")

    def unSubscribeToLocationData(self):
        """Unsubscribes from location data"""
        msg = {
            "reqORinfo": "info",
            "type": "locIDubsub",
        }
        self.send_data(msg)
