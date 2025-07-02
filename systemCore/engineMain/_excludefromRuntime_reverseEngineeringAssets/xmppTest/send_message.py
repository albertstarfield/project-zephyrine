import sys
import logging
import asyncio
from getpass import getpass
import slixmpp

class SendMsgClient(slixmpp.ClientXMPP):
    """
    A simple XMPP client that sends one message and then disconnects.
    This version correctly handles the asyncio event loop required by slixmpp.
    """
    def __init__(self, jid, password, recipient, message):
        super().__init__(jid, password)
        self.recipient = recipient
        self.msg_to_send = message

        # Register the event handlers
        self.add_event_handler("session_start", self.start)
        self.add_event_handler("failed_auth", self.failed_auth)

    async def start(self, event):
        """
        This coroutine is executed when the XMPP session is established.
        It sends the message and then disconnects, which stops the event loop.
        """
        print(f"\nINFO: XMPP Session started for {self.boundjid}.")
        self.send_presence()
        self.get_roster()

        print(f"INFO: Sending message to '{self.recipient}'...")
        self.send_message(
            mto=self.recipient,
            mbody=self.msg_to_send,
            mtype='chat'
        )
        print("INFO: Message sent successfully.")
        
        # This is the crucial step to end the process.
        self.disconnect()

    def failed_auth(self, event):
        """Handle authentication failure."""
        print("ERROR: Authentication failed. Please check your JID and password.")
        self.disconnect()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} \"Your message here\"")
        sys.exit(1)

    # --- Configuration (matches your Zephy setup) ---
    CLIENT_JID = 'albert@localhost'
    RECIPIENT_JID = 'zephy.localhost'
    CLIENT_PASSWORD = getpass(f"Enter password for XMPP account '{CLIENT_JID}': ")
    
    message = sys.argv[1]

    # Setup logging to see slixmpp's output
    logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')

    # Initialize the client
    xmpp = SendMsgClient(CLIENT_JID, CLIENT_PASSWORD, RECIPIENT_JID, message)
    
    # --- THIS IS THE FIX ---
    # Register plugins *before* connecting.
    xmpp.register_plugin('xep_0030') # Service Discovery
    xmpp.register_plugin('xep_0199') # XMPP Ping

    # The connect() method is synchronous and prepares the connection.
    # The process() method runs the event loop until disconnect() is called.
    if xmpp.connect():
        xmpp.process()
        print("INFO: Test script finished.")
    else:
        print("ERROR: Unable to connect to the XMPP server.")