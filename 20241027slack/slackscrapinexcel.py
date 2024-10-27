import requests
import time
import json
import pandas as pd

# Slack API token
api_token = "xoxb-4594289558310-7934228789494-TryXG0bZV1BjpGeZglNKQWmg"
headers = {"Authorization": f"Bearer {api_token}"}

# Function to get all channels in the workspace
def get_channel_ids():
    url = "https://slack.com/api/conversations.list"
    params = {"limit": 100}  # maximum number of channels per request
    channels = []

    while True:
        response = requests.get(url, headers=headers, params=params).json()

        if not response["ok"]:
            print(f"Error: {response['error']}")
            break

        for channel in response["channels"]:
            channels.append({"id": channel["id"], "name": channel["name"]})

        # Pagination handling
        if "next_cursor" in response.get("response_metadata", {}):
            params["cursor"] = response["response_metadata"]["next_cursor"]
        else:
            break

    return channels

# Function to get messages from a specific channel and focus on files, posts, and replies
def get_channel_messages(channel_id):
    url = "https://slack.com/api/conversations.history"
    params = {"channel": channel_id, "limit": 100}  # maximum messages per request
    messages_data = []

    while True:
        response = requests.get(url, headers=headers, params=params).json()

        if not response["ok"]:
            print(f"Error: {response['error']}")
            break

        for message in response["messages"]:
            # Focus on messages with files, posts, or replies
            message_type = "Message"
            if "files" in message:
                message_type = "File"
            elif "text" in message and "replies" in message:
                message_type = "Reply"
            elif "text" in message and "thread_ts" in message:
                message_type = "Thread"

            # Append relevant message data
            messages_data.append({
                "channel_id": channel_id,
                "user": message.get("user"),
                "text": message.get("text"),
                "type": message_type,
                "timestamp": message.get("ts"),
                "files": message.get("files", []),  # file metadata
                "replies": message.get("replies", [])
            })

        # Pagination handling
        if "next_cursor" in response.get("response_metadata", {}):
            params["cursor"] = response["response_metadata"]["next_cursor"]
        else:
            break

        # To avoid hitting rate limits
        time.sleep(1)

    return messages_data

# Run the script to get channel IDs and messages
channels = get_channel_ids()
print("Channels found:", [channel['name'] for channel in channels])

# Collect data from all channels
all_data = []
for channel in channels:
    print(f"Fetching messages from channel: {channel['name']}")
    messages = get_channel_messages(channel["id"])

    for message in messages:
        all_data.append({
            "Channel Name": channel["name"],
            "User": message["user"],
            "Text": message["text"],
            "Type": message["type"],
            "Timestamp": message["timestamp"],
            "Files": json.dumps(message["files"]),  # Convert list of files to JSON string
            "Replies": json.dumps(message["replies"])  # Convert list of replies to JSON string
        })

# Save collected data to an Excel file
df = pd.DataFrame(all_data)
df.to_excel("slack_channels_messages.xlsx", index=False)
print("Channels and messages saved to 'slack_channels_messages.xlsx'")
