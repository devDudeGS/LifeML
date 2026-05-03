import json
import math
import requests
from datetime import datetime, timedelta, date
from collections import defaultdict
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from dotenv import load_dotenv
import os

load_dotenv()

# ── Oura setup ──────────────────────────────────────────────────────────────
OURA_TOKEN = os.environ["OURA_PAT"]
OURA_HEADERS = {"Authorization": f"Bearer {OURA_TOKEN}"}
OURA_BASE = "https://api.ouraring.com/v2/usercollection"

# ── Google Health setup ──────────────────────────────────────────────────────
with open("google_health_tokens.json") as f:
    tokens = json.load(f)

creds = Credentials(
    token=tokens["access_token"],
    refresh_token=tokens["refresh_token"],
    client_id=tokens["client_id"],
    client_secret=tokens["client_secret"],
    token_uri="https://oauth2.googleapis.com/token",
)
creds.refresh(Request())
tokens["access_token"] = creds.token
with open("google_health_tokens.json", "w") as f:
    json.dump(tokens, f, indent=2)

FITBIT_HEADERS = {
    "Authorization": f"Bearer {creds.token}",
    "Accept": "application/json",
}
FITBIT_BASE = "https://health.googleapis.com/v4/users/me"
UTC_OFFSET_HRS = -4

# ── Expected sleep schedule ───────────────────────────────────────────────────
# Keys are the day-of-week you WAKE UP (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun)
EXPECTED_BEDTIME = {
    0: "21:25",  # wake Mon  → bed Sun night
    1: "20:55",  # wake Tue  → bed Mon night
    2: "20:55",  # wake Wed  → bed Tue night
    3: "20:55",  # wake Thu  → bed Wed night
    4: "21:25",  # wake Fri  → bed Thu night
    5: "22:10",  # wake Sat  → bed Fri night
    6: "22:10",  # wake Sun  → bed Sat night
}
EXPECTED_WAKEUP = {
    0: "05:30",  # Mon
    1: "05:00",  # Tue
    2: "05:00",  # Wed
    3: "05:00",  # Thu
    4: "05:30",  # Fri
    5: "06:00",  # Sat
    6: "06:00",  # Sun
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def oura_get(endpoint, params):
    r = requests.get(f"{OURA_BASE}/{endpoint}", headers=OURA_HEADERS, params=params)
    r.raise_for_status()
    return r.json()

def fitbit_get_all_pages(endpoint, params):
    """Fetch all pages from a Fitbit endpoint, following nextPageToken."""
    all_points = []
    url = f"{FITBIT_BASE}/{endpoint}"
    while True:
        r = requests.get(url, headers=FITBIT_HEADERS, params=params)
        r.raise_for_status()
        data = r.json()
        all_points.extend(data.get("dataPoints", []))
        next_token = data.get("nextPageToken")
        if not next_token:
            break
        params = {**params, "pageToken": next_token}
    return all_points

def civil_date_tuple(dp):
    """Extract (year, month, day) from a steps data point's civilStartTime."""
    d = dp["steps"]["interval"]["civilStartTime"]["date"]
    return (d["year"], d["month"], d["day"])

def is_pixel_watch(dp):
    return dp["dataSource"]["device"].get("displayName") == "Pixel Watch"

def time_diff_minutes(actual_hhmm, expected_hhmm):
    """Return signed diff in minutes: positive = late, negative = early."""
    a_h, a_m = map(int, actual_hhmm.split(":"))
    e_h, e_m = map(int, expected_hhmm.split(":"))
    diff = (a_h * 60 + a_m) - (e_h * 60 + e_m)
    # Wrap around midnight for bedtime (e.g. 23:00 vs 01:00)
    if diff > 12 * 60:
        diff -= 24 * 60
    elif diff < -12 * 60:
        diff += 24 * 60
    return diff

def utc_to_local(utc_str):
    """Convert a UTC ISO string to local datetime using hardcoded offset."""
    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    return dt + timedelta(hours=UTC_OFFSET_HRS)

# ── Date range ───────────────────────────────────────────────────────────────
today = date.today()
start = today - timedelta(days=8)  # buffer for Oura session alignment
end = today
params_range = {"start_date": str(start), "end_date": str(end)}

# ── Fetch Oura data ──────────────────────────────────────────────────────────
sleep_raw     = oura_get("sleep", params_range)
sleep_daily   = oura_get("daily_sleep", params_range)
readiness_raw = oura_get("daily_readiness", params_range)

sleep_score_by_day    = {s["day"]: s["score"] for s in sleep_daily["data"]}
readiness_by_day      = {r["day"]: r["score"] for r in readiness_raw["data"]}
long_sleeps           = {s["day"]: s for s in sleep_raw["data"] if s["type"] == "long_sleep"}

NON_MAIN_TYPES = {"nap", "late_nap", "rest", "sleep"}
naps_by_day = defaultdict(int)
for s in sleep_raw["data"]:
    if s["type"] in NON_MAIN_TYPES:
        naps_by_day[s["day"]] += s["total_sleep_duration"]

# ── Fetch Fitbit data ────────────────────────────────────────────────────────
fitbit_start = str(today - timedelta(days=7))

# Steps — Pixel Watch only, all pages, filter client-side by date
all_steps = fitbit_get_all_pages(
    "dataTypes/steps/dataPoints",
    {"filter": f'steps.interval.civil_start_time >= "{fitbit_start}T00:00:00"'}
)
steps_by_day = defaultdict(int)
for dp in all_steps:
    if is_pixel_watch(dp):
        t = civil_date_tuple(dp)
        day_str = f"{t[0]:04d}-{t[1]:02d}-{t[2]:02d}"
        steps_by_day[day_str] += int(dp["steps"]["count"])

# Exercise sessions — Pixel Watch only
all_exercise = fitbit_get_all_pages(
    "dataTypes/exercise/dataPoints",
    {"filter": f'exercise.interval.civil_start_time >= "{fitbit_start}T00:00:00"'}
)

yoga_by_day    = defaultdict(list)   # list of durations in mins
exercise_by_day = defaultdict(list)  # list of non-yoga type strings

for dp in all_exercise:
    if not is_pixel_watch(dp):
        continue
    ex = dp["exercise"]
    start_local = utc_to_local(ex["interval"]["startTime"])
    day_str = start_local.strftime("%Y-%m-%d")
    start_hhmm = start_local.strftime("%H:%M")
    duration_mins = round(
        (datetime.fromisoformat(ex["interval"]["endTime"].replace("Z", "+00:00")) -
         datetime.fromisoformat(ex["interval"]["startTime"].replace("Z", "+00:00"))
        ).seconds / 60, 1
    )
    ex_type = ex.get("exerciseType", "UNKNOWN")

    # Use displayName when exerciseType is OTHER, otherwise use exerciseType
    if ex_type == "OTHER":
        ex_type = ex.get("displayName", "OTHER")

    if ex_type == "YOGA":
        yoga_by_day[day_str].append((start_hhmm, duration_mins))
    else:
        exercise_by_day[day_str].append(ex_type)

# ── Print results ────────────────────────────────────────────────────────────
cols = (
    f"{'Date':<12} {'bedtime_start':<16} {'expected_bed':<14} {'bed_diff':<10} "
    f"{'wakeup_end':<12} {'expected_wake':<15} {'wake_diff':<11} "
    f"{'sleep_length':<14} {'awake_mins':<12} {'sleep_score':<13} "
    f"{'readiness':<11} {'nap_length':<12} {'steps':<8} "
    f"{'med_first':<11} {'med_mins':<10} {'exercise_types'}"
)
print(cols)
print("-" * 200)

for i in range(7, 0, -1):
    day = str(today - timedelta(days=i))
    s = long_sleeps.get(day)

    if not s:
        print(f"{day:<12} {'No sleep data'}")
        continue

    # Oura fields
    bedtime_start_dt = datetime.fromisoformat(s["bedtime_start"])
    sleep_onset = bedtime_start_dt + timedelta(seconds=s["latency"])
    sleep_bedtime_start = sleep_onset.strftime("%H:%M")

    wakeup_dt = datetime.fromisoformat(s["bedtime_end"])
    sleep_wakeup_end = wakeup_dt.strftime("%H:%M")

    wakeup_dow = wakeup_dt.weekday()  # 0=Mon … 6=Sun
    exp_bed = EXPECTED_BEDTIME[wakeup_dow]
    exp_wake = EXPECTED_WAKEUP[wakeup_dow]
    bed_diff_mins = time_diff_minutes(sleep_bedtime_start, exp_bed)
    wake_diff_mins = time_diff_minutes(sleep_wakeup_end, exp_wake)
    bed_diff_str = f"{bed_diff_mins:+d}m"
    wake_diff_str = f"{wake_diff_mins:+d}m"

    sleep_length = round(s["total_sleep_duration"] / 3600, 2)
    sleep_awake_mins = math.ceil((s["awake_time"] - s["latency"]) / 60)
    sleep_score = sleep_score_by_day.get(day, "N/A")
    readiness = readiness_by_day.get(day, "N/A")

    nap_secs = naps_by_day.get(day, 0)
    nap_length = round(nap_secs / 3600, 2) if nap_secs else "—"

    # Fitbit fields
    steps = steps_by_day.get(day, 0)

    yoga_sessions = sorted(yoga_by_day.get(day, []))  # sorted by start time
    med_first = yoga_sessions[0][0] if yoga_sessions else "—"
    med_mins = round(sum(d for _, d in yoga_sessions), 1) if yoga_sessions else 0

    ex_types = ", ".join(sorted(set(exercise_by_day.get(day, [])))) or "—"

    print(
        f"{day:<12} {sleep_bedtime_start:<16} {exp_bed:<14} {bed_diff_str:<10} "
        f"{sleep_wakeup_end:<12} {exp_wake:<15} {wake_diff_str:<11} "
        f"{sleep_length:<14} {sleep_awake_mins:<12} {sleep_score:<13} "
        f"{readiness:<11} {str(nap_length):<12} {steps:<8} "
        f"{med_first:<11} {med_mins:<10} {ex_types}"
    )