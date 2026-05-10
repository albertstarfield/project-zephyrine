-- File: src/zephyrine-http_client.ads (Pure Ada Version)
package Zephyrine.HTTP_Client is
   HTTP_Error : exception;
   function Post (URL : String; Payload : String) return String;
end Zephyrine.HTTP_Client;