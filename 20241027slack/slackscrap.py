from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

client = WebClient(token="xoxb-your-slack-bot-token")

try:
    response = client.conversations_history(channel="your-channel-id")
    messages = response['messages']
    for message in messages:
        print(f"{message['user']}: {message['text']}")
except SlackApiError as e:
    print(f"Error fetching conversations: {e.response['error']}")
