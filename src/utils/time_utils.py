from datetime import datetime, timedelta

def get_unix_timestamp(days_ago):
    return int((datetime.utcnow() - timedelta(days=days_ago)).timestamp())
