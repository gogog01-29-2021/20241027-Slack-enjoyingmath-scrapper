import os
import time
import requests
from datetime import datetime
import pandas as pd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
# https://chatgpt.com/c/675ce6ca-47cc-8008-b046-eb0c91b4e45a
# Initialize Slack client 
client = WebClient(token="")

def get_recently_active_channels(client, days=90):
    try:
        channels = client.conversations_list(types="public_channel,private_channel").get("channels", [])
        active_channels = []
        for channel in channels:
            try:
                # Get the latest message history for the channel
                history = client.conversations_history(channel=channel["id"], limit=1)
                # Check if there is any message in the history
                if "messages" in history and history["messages"]:
                    last_message = history["messages"][0]
                    timestamp = float(last_message["ts"])
                    # Check if the message is within the specified 'days' range
                    if (time.time() - timestamp) / 86400 <= days:
                        active_channels.append({"name": channel["name"], "id": channel["id"]})
                else:
                    print(f"No recent messages in channel {channel['name']}")
            except SlackApiError as e:
                if e.response['error'] == 'not_in_channel':
                    print(f"Skipping channel {channel['name']}: not in channel.")
                else:
                    print(f"Error accessing channel {channel['name']}: {e.response['error']}")
        return active_channels
    except SlackApiError as e:
        print(f"Error retrieving channels: {e.response['error']}")
        return []


def get_channel_files(client, channel_id, start_time=None, end_time=None):
    files = []
    try:
        response = client.files_list(channel=channel_id, ts_from=start_time, ts_to=end_time)
        files.extend(response["files"])
        return files
    except SlackApiError as e:
        print(f"Error fetching files for channel {channel_id}: {e.response['error']}")
        return []

def sanitize_filename(filename):
    """Sanitize filenames to remove invalid characters."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)

def download_file(url, token, path):
    headers = {'Authorization': f'Bearer {token}'}
    with requests.get(url, headers=headers, stream=True) as response:
        if response.status_code == 200:
            with open(path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        else:
            print(f"Failed to download file from {url} - Status code: {response.status_code}")

def get_channel_messages(client, channel_id, start_time=None, end_time=None):
    messages = []
    try:
        response = client.conversations_history(channel=channel_id, oldest=start_time, latest=end_time)
        messages.extend(response["messages"])
        return messages
    except SlackApiError as e:
        print(f"Error fetching messages for channel {channel_id}: {e.response['error']}")
        return []

active_channels = get_recently_active_channels(client, days=90)
if not active_channels:
    print("No accessible active channels found.")
    exit()

print("Accessible active channels found:")
for i, ch in enumerate(active_channels, start=1):
    print(f"{i}. {ch['name']}")

while True:
    user_input = input("Enter the numbers of channels to scrape (e.g., '1,2,5' or '1-3'), or press Enter to scrape all channels: ").strip()
    selected_channels = []

    if not user_input:
        selected_channels = active_channels
        break
    else:
        try:
            ranges = user_input.split(",")
            for part in ranges:
                if "-" in part:
                    start, end = map(int, part.split("-"))
                    selected_channels.extend(active_channels[start-1:end])
                else:
                    selected_channels.append(active_channels[int(part)-1])
            break
        except (ValueError, IndexError):
            print("Invalid input. Please enter valid channel numbers or ranges.")

start_date = input("Enter start date (YYYY-MM-DD) or leave blank for no limit: ")
end_date = input("Enter end date (YYYY-MM-DD) or leave blank for no limit: ")

start_time = int(time.mktime(datetime.strptime(start_date, "%Y-%m-%d").timetuple())) if start_date else None
end_time = int(time.mktime(datetime.strptime(end_date, "%Y-%m-%d").timetuple())) if end_date else None

output_folder = f"slack_output_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_folder, exist_ok=True)

for channel in selected_channels:
    channel_name = channel["name"]
    channel_id = channel["id"]
    channel_folder = os.path.join(output_folder, sanitize_filename(channel_name))
    os.makedirs(channel_folder, exist_ok=True)

    print(f"Scraping files from channel: {channel_name}")
    files = get_channel_files(client, channel_id, start_time=start_time, end_time=end_time)
    file_info_list = []

    # File scraping and renaming logic
    for idx, file in enumerate(files, start=1):
        sanitized_name = sanitize_filename(file["name"])
        if not sanitized_name.endswith(f".{file['filetype']}"):
            sanitized_name += f".{file['filetype']}"  # Ensure file extension is preserved
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_file_{idx}.{file['filetype']}"
        file_path = os.path.join(channel_folder, new_filename)

        try:
            download_file(file["url_private"], "", file_path)
            print(f"Downloaded: {new_filename}")
        except Exception as e:
            print(f"Error downloading {file['name']}: {e}")

        file_info = {
            "original_name": file["name"],
            "sanitized_name": new_filename,
            "type": file["filetype"],
            "url": file["url_private"]
        }
        file_info_list.append(file_info)

    if file_info_list:
        files_df = pd.DataFrame(file_info_list)
        files_df.to_excel(os.path.join(channel_folder, f"{sanitize_filename(channel_name)}_files.xlsx"), index=False)

    # Message scraping logic
    print(f"Scraping messages from channel: {channel_name}")
    messages = get_channel_messages(client, channel_id, start_time=start_time, end_time=end_time)
    message_texts = []
    
    for msg in messages:
        user = msg.get("user", "Unknown")  # Get the user ID, default to "Unknown"
        text = msg.get("text", "")         # Get the text content
        timestamp = float(msg["ts"])
        date_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        message_texts.append({
            "user": user,
            "text": text,
            "date": date_time
        })

    if message_texts:
        messages_df = pd.DataFrame(message_texts)
        messages_df.to_excel(os.path.join(channel_folder, f"{sanitize_filename(channel_name)}_messages.xlsx"), index=False)

print(f"Scraping completed. Data and files saved in folder: {output_folder}")
