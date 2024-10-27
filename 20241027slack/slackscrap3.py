import requests
import time
import json

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

# Function to get messages from a specific channel
def get_channel_messages(channel_id):
    url = "https://slack.com/api/conversations.history"
    params = {"channel": channel_id, "limit": 100}  # maximum messages per request
    messages = []

    while True:
        response = requests.get(url, headers=headers, params=params).json()

        if not response["ok"]:
            print(f"Error: {response['error']}")
            break

        messages.extend(response["messages"])

        # Pagination handling
        if "next_cursor" in response.get("response_metadata", {}):
            params["cursor"] = response["response_metadata"]["next_cursor"]
        else:
            break

        # To avoid hitting rate limits
        time.sleep(1)

    return messages

# Run the script to get channel IDs and messages
channels = get_channel_ids()
print("Channels found:")
for channel in channels:
    print(f"Channel Name: {channel['name']}, Channel ID: {channel['id']}")

    # Fetch and print messages for each channel
    messages = get_channel_messages(channel["id"])
    print(f"Messages in {channel['name']}:")

    for message in messages:
        print(message)

    print("\n")

# Save channels and messages to JSON file
with open("slack_channels_messages.json", "w") as file:
    json.dump({"channels": channels, "messages": messages}, file, indent=4)

print("Channels and messages saved to 'slack_channels_messages.json'")
