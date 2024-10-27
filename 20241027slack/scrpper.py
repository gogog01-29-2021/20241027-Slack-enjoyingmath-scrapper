import sqlite3

# Path to Chrome's history file (this could change based on the OS)
chrome_history = "C:\Users\kimseonghyun\AppData\Local\Google\Chrome\User Data\Default\History"

# Connect to the database
conn = sqlite3.connect(chrome_history)
cursor = conn.cursor()

# Query the URLs visited
cursor.execute("SELECT url, title, visit_count FROM urls")

# Get the history data and perform iterative/recursive operations
history_data = cursor.fetchall()
for url, title, visit_count in history_data:
    print(f"Visited URL: {url} with title {title} visited {visit_count} times")

conn.close()
