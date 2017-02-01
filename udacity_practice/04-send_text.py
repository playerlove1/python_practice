#使用twilio的服務傳送簡訊到手機
from twilio.rest import TwilioRestClient 
 
# put your own credentials here 
ACCOUNT_SID = ""  # Your Account SID from www.twilio.com/console
AUTH_TOKEN = ""     # Your Auth Token from www.twilio.com/console
 
client = TwilioRestClient(ACCOUNT_SID, AUTH_TOKEN) 
 
message = client.messages.create(to="要傳到哪的手機號碼",from_="twilio的PHONE NUMBER",body="簡訊內容")

print(message.sid)
