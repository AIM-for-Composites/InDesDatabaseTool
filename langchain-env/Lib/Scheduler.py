"""
scheduler.py
Daily crawler  → every day at 02:00 AM UTC
Weekly discovery → every Sunday at 01:00 AM UTC (runs BEFORE daily)

Run: python scheduler.py
Test single run: python crawler_graph.py
Test discovery: python discovery.py
"""
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from crawler_graph import run_crawler
from Discovery import run_discovery

scheduler = BlockingScheduler(timezone="UTC")

# Weekly source discovery — Sunday 01:00 UTC
scheduler.add_job(
    run_discovery,
    CronTrigger(day_of_week="sun", hour=1, minute=0),
    id="weekly_discovery",
    name="Weekly source discovery (DuckDuckGo)"
)
# Daily crawler — every day 02:00 UTC
scheduler.add_job(
    lambda: run_crawler(days_back=1),
    trigger="interval",
    minutes = 1,
    id="daily_crawler",
    name="Daily materials crawler"
)

if __name__ == "__main__":
    print("[Scheduler] Running.")
    print("  Weekly discovery : Sunday 01:00 UTC")
    print("  Daily crawler    : Every day 02:00 UTC")
    print("  Press Ctrl+C to stop.\n")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("[Scheduler] Stopped.")