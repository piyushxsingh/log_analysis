"""
generate_logs.py
----------------
Generates realistic synthetic system/application logs for training and testing.
Produces a CSV with timestamps, severity levels, and log messages.
"""

import random
import pandas as pd
from datetime import datetime, timedelta

# ─── Seed for reproducibility ────────────────────────────────────────────────
random.seed(42)

# ─── Template messages per severity ──────────────────────────────────────────
LOG_TEMPLATES = {
    "INFO": [
        "User {user} logged in successfully from IP {ip}",
        "Service {service} started on port {port}",
        "Database connection established successfully",
        "Scheduled job {job} completed in {ms}ms",
        "Cache refreshed successfully for module {module}",
        "Configuration file loaded: {config}",
        "Health check passed for service {service}",
        "Backup completed successfully for {db}",
        "API request to {endpoint} returned 200 OK",
        "Session created for user {user}",
        "File {file} uploaded successfully",
        "Email notification sent to {user}",
        "Payment processed successfully for order {order}",
        "User {user} logged out",
        "New record inserted into table {table}",
    ],
    "WARNING": [
        "High memory usage detected: {pct}% utilization",
        "Response time for {endpoint} exceeded {ms}ms threshold",
        "Disk usage at {pct}% on volume {vol}",
        "Deprecated API endpoint {endpoint} called by {user}",
        "Failed login attempt for user {user} from IP {ip}",
        "Rate limit approaching for API key {key}",
        "SSL certificate for {domain} expires in {days} days",
        "Queue size for {queue} exceeds {n} messages",
        "Retry attempt {n} for service {service}",
        "Low available connections in pool: {n} remaining",
        "Slow query detected on table {table}: {ms}ms",
        "Config value {config} is using deprecated format",
        "Cache miss rate elevated: {pct}%",
        "User {user} attempted to access restricted resource",
        "Timezone mismatch detected in request from {ip}",
    ],
    "ERROR": [
        "Database connection failed: timeout after {ms}ms",
        "NullPointerException in module {module} at line {line}",
        "Failed to process payment for order {order}: {reason}",
        "Service {service} returned HTTP 500",
        "Authentication token expired for user {user}",
        "File not found: {file}",
        "Permission denied: user {user} accessing {resource}",
        "API call to {endpoint} failed after {n} retries",
        "Unhandled exception in thread {thread}: {reason}",
        "Data validation failed for field {field}",
        "Email delivery failed for {user}: {reason}",
        "Out of memory error in process {pid}",
        "Configuration missing required key: {config}",
        "Failed to write to disk: {reason}",
        "Session {session} not found or expired",
    ],
    "CRITICAL": [
        "SYSTEM CRASH: Kernel panic in process {pid}",
        "Database cluster {db} is completely unreachable",
        "Security breach detected: unauthorized root access from {ip}",
        "Data corruption detected in table {table}",
        "Service {service} has been down for {n} minutes",
        "DDoS attack detected: {n} requests/sec from {ip}",
        "Disk {vol} is FULL — writes are failing",
        "CPU usage at 100% for {n} consecutive minutes",
        "Ransomware activity suspected on host {host}",
        "All database replicas are out of sync",
        "Production deployment FAILED: rollback initiated",
        "Memory leak detected — system will restart in {n}s",
        "SSL private key {key} has been compromised",
        "Critical vulnerability CVE-{cve} actively exploited",
        "Backup restoration FAILED — data may be lost",
    ],
}

# ─── Random value helpers ─────────────────────────────────────────────────────
def _r(lst): return random.choice(lst)

USERS     = ["alice", "bob", "charlie", "dave", "eve", "system", "admin"]
SERVICES  = ["auth-service", "payment-api", "user-service", "data-processor", "scheduler"]
IPS       = ["192.168.1." + str(i) for i in range(1, 30)] + ["10.0.0." + str(i) for i in range(1, 10)]
ENDPOINTS = ["/api/v1/users", "/api/v2/payments", "/health", "/login", "/data/export"]
MODULES   = ["UserController", "PaymentProcessor", "DataExporter", "AuthManager"]
TABLES    = ["users", "transactions", "sessions", "audit_logs", "products"]
DOMAINS   = ["example.com", "api.example.com", "internal.company.io"]

def _fill(template: str) -> str:
    """Fill a template string with random realistic values."""
    return template.format(
        user=_r(USERS), service=_r(SERVICES), ip=_r(IPS),
        port=_r([8080, 3000, 5432, 6379, 443]),
        ms=random.randint(50, 9000), pct=random.randint(60, 99),
        vol=_r(["sda1", "sdb2", "nvme0n1"]), endpoint=_r(ENDPOINTS),
        module=_r(MODULES), config=_r(["app.yaml", "db.conf", "secrets.env"]),
        db=_r(["postgres-primary", "mongo-cluster", "redis-main"]),
        table=_r(TABLES), order=f"ORD-{random.randint(1000,9999)}",
        file=_r(["report.pdf", "data.csv", "backup.tar.gz"]),
        job=_r(["nightly-backup", "log-rotation", "db-cleanup"]),
        key=f"KEY-{random.randint(100,999)}", domain=_r(DOMAINS),
        days=random.randint(1, 30), queue=_r(["email-queue", "job-queue"]),
        n=random.randint(1, 500), line=random.randint(10, 500),
        reason=_r(["timeout", "invalid card", "insufficient funds", "server error"]),
        resource=_r(["/admin", "/root", "/etc/passwd"]),
        thread=f"Thread-{random.randint(1,20)}", field=_r(["email", "phone", "ssn"]),
        pid=random.randint(1000, 9999), session=f"SES-{random.randint(100,999)}",
        host=f"host-{random.randint(1,10)}", cve=f"{random.randint(2020,2024)}-{random.randint(1000,9999)}",
    )


def generate_logs(n: int = 1000, anomaly_rate: float = 0.15) -> pd.DataFrame:
    """
    Generate n synthetic log entries.
    anomaly_rate controls what fraction are ERROR/CRITICAL.
    """
    records = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    # Weight distribution: mostly INFO/WARNING, some ERROR, few CRITICAL
    weights = {"INFO": 0.50, "WARNING": 0.25, "ERROR": 0.17, "CRITICAL": 0.08}

    for i in range(n):
        # Simulate sequential timestamps with occasional bursts
        gap = random.expovariate(1 / 30)  # avg 30s between logs
        start_time += timedelta(seconds=gap)

        severity = random.choices(
            list(weights.keys()), weights=list(weights.values()), k=1
        )[0]

        template = _r(LOG_TEMPLATES[severity])
        message  = _fill(template)

        records.append({
            "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "severity":  severity,
            "message":   message,
            "source":    _r(SERVICES),
            "host":      f"host-{random.randint(1, 5)}",
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_logs(n=1200)
    out = "system_logs.csv"
    df.to_csv(out, index=False)
    print(f"✅  Generated {len(df)} log entries → {out}")
    print(df["severity"].value_counts())
