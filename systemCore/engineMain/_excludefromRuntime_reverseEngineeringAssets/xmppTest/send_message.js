// send_message.js
import { client } from '@xmpp/client';
import pkg from '@xmpp/xml';
const { xml } = pkg;

// --- Configuration (matches your Zephy setup) ---
const CLIENT_JID = 'albert@localhost';
const RECIPIENT_JID = 'zephy.localhost';
const XMPP_SERVER_HOST = '127.0.0.1';
const XMPP_SERVER_PORT = 5222;

// --- MODIFICATION: Password is now embedded directly ---
const CLIENT_PASSWORD = 'a_very_strong_secret_for_local_use';
// --- END OF MODIFICATION ---

// --- Main Logic ---
async function main() {
    const messageToSend = process.argv[2];
    if (!messageToSend) {
        console.error('ERROR: Please provide a message to send as a command-line argument.');
        console.error('Usage: node send_message.js "Your message here"');
        process.exit(1);
    }

    const xmpp = client({
        service: `xmpp://${XMPP_SERVER_HOST}:${XMPP_SERVER_PORT}`,
        domain: 'localhost',
        resource: 'nodejs-test-client',
        username: 'albert',
        password: CLIENT_PASSWORD, // Use the hardcoded password
    });

    xmpp.on('error', (err) => {
        console.error('XMPP ERROR:', err.message);
    });

    xmpp.on('offline', () => {
        console.log('INFO: Client is offline.');
    });

    xmpp.on('online', async (address) => {
        console.log(`INFO: Client is online as ${address.toString()}`);
        
        await xmpp.send(xml('presence'));

        const messageStanza = xml(
            'message',
            { to: RECIPIENT_JID, type: 'chat' },
            xml('body', {}, messageToSend)
        );

        console.log(`INFO: Sending message to '${RECIPIENT_JID}'...`);
        await xmpp.send(messageStanza);
        console.log('INFO: Message sent successfully.');

        await xmpp.stop();
    });

    console.log('INFO: Starting XMPP client...');
    await xmpp.start();
}

main().catch(console.error);