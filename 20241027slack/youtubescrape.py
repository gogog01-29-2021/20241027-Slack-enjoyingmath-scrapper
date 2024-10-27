import requests
import csv

def get_youtube_video_titles(api_key, playlist_id):
    base_url = "https://www.googleapis.com/youtube/v3/playlistItems"
    params = {
        "part": "snippet",
        "maxResults": 50,  # The max value allowed by YouTube Data API
        "playlistId": playlist_id,
        "key": api_key
    }

    video_titles = []
    next_page_token = None

    while True:
        # Add the page token to the parameters if it exists
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(base_url, params=params)
        data = response.json()

        # Check for errors in the response
        if "error" in data:
            print(f"Error: {data['error']['message']}")
            break

        # Extract video titles
        for item in data.get("items", []):
            title = item["snippet"]["title"]
            video_titles.append(title)

        # Check if there is another page
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return video_titles

# Your provided API key and playlist ID
api_key = "AIzaSyDSV4-6h9sXZtOVA2kjncflmheZWjD4SoQ"
playlist_id = "LL"

# Get the video titles
titles = get_youtube_video_titles(api_key, playlist_id)

# Save to CSV file
with open("youtube_video_titles.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Index", "Title"])  # Header row
    for i, title in enumerate(titles, start=1):
        writer.writerow([i, title])

print("Video titles saved to 'youtube_video_titles.csv'")
