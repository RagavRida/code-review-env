"""
Data Generator for CodeReviewEnv

Generates realistic synthetic pull requests with actual code diffs.
The FIXED_TEST_SUITE provides 50 pre-generated PRs at seed=42 for
deterministic evaluation — all episodes draw from this fixed suite.

Bug categories and ground truth severity mapping:
  sql_injection         → critical
  security_vulnerability → critical
  race_condition        → high
  null_pointer          → high
  logic_error           → medium
  missing_error_handling → medium
  performance_issue     → low
  style_only            → none

Author experience affects bug probability:
  junior: 70% chance of bug (weighted: null_pointer, missing_error_handling)
  mid:    40% chance of bug (any category)
  senior: 15% chance of bug (weighted: performance_issue, style_only)
"""

import random
from typing import List, Dict, Optional, Tuple
from env.models import PRFile, Observation


# ─── Bug category → ground truth severity ────────────────────────────────────

BUG_SEVERITY_MAP: Dict[str, str] = {
    "sql_injection": "critical",
    "security_vulnerability": "critical",
    "race_condition": "high",
    "null_pointer": "high",
    "logic_error": "medium",
    "missing_error_handling": "medium",
    "performance_issue": "low",
    "style_only": "none",
}

# Ordered severity levels for adjacency calculations
SEVERITY_ORDER: List[str] = ["critical", "high", "medium", "low", "none"]

# Keywords associated with each bug category — used by grader_hard for
# specificity scoring via keyword matching (no LLM calls)
BUG_KEYWORDS: Dict[str, List[str]] = {
    "null_pointer": ["null", "None", "NullPointerException", "check", "guard", "nil"],
    "sql_injection": ["injection", "parameterize", "prepared", "sanitize", "escape", "query"],
    "race_condition": ["race", "lock", "mutex", "atomic", "thread-safe", "concurrent", "sync"],
    "logic_error": ["condition", "branch", "edge case", "off-by-one", "boundary", "logic"],
    "missing_error_handling": ["exception", "catch", "error", "handle", "try", "raise"],
    "security_vulnerability": ["auth", "token", "encrypt", "hash", "expose", "secret", "leak"],
    "performance_issue": ["O(n)", "complexity", "cache", "index", "query", "optimize", "loop"],
    "style_only": ["naming", "format", "indent", "style", "convention"],
}

# Actionability keywords — used by grader_hard to score whether comments
# contain concrete suggestions (empirically selected from code review corpora)
ACTIONABILITY_KEYWORDS: List[str] = [
    "use", "replace", "add", "remove", "consider", "should", "instead",
    "refactor", "extract", "avoid", "change", "move", "wrap",
]


# ─── PR Templates ────────────────────────────────────────────────────────────
# Each template is a complete PR with realistic code diff, ground truth bug
# info, and pre-annotated human labels for reliability analysis.

PR_TEMPLATES: List[Dict] = [
    # ── 1. Java null pointer ────────────────────────────────────────────
    {
        "pr_id": "PR-001",
        "title": "Fix null pointer in UserService.java",
        "description": "Refactored UserService to handle user lookup. Added caching for frequently accessed users.",
        "author_experience": "junior",
        "language": "java",
        "filename": "src/main/java/com/app/UserService.java",
        "diff": '''@@ -45,12 +45,18 @@ public class UserService {
+    public User getUserProfile(String userId) {
+        User user = userRepository.findById(userId);
+        // BUG: No null check — user could be null if not found
+        String displayName = user.getFirstName() + " " + user.getLastName();
+        user.setDisplayName(displayName);
+        cacheManager.put(userId, user);
+        return user;
+    }
+
+    public void updateUserEmail(String userId, String email) {
+        User user = userRepository.findById(userId);
+        user.setEmail(email);  // BUG: same null pointer pattern
+        userRepository.save(user);
+    }''',
        "lines_changed": 14,
        "has_tests": False,
        "bug_category": "null_pointer",
        "ground_truth_severity": "high",
        "bug_lines": [48, 56],  # lines where bugs exist in the diff
        "human_labels": ["high", "high", "high"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 2. Python missing error handling ────────────────────────────────
    {
        "pr_id": "PR-002",
        "title": "Add rate limiting to /api/auth",
        "description": "Implemented token bucket rate limiter for authentication endpoints to prevent brute force attacks.",
        "author_experience": "mid",
        "language": "python",
        "filename": "api/auth/rate_limiter.py",
        "diff": '''@@ -1,0 +1,28 @@
+import time
+from collections import defaultdict
+
+class RateLimiter:
+    def __init__(self, max_requests=100, window_seconds=60):
+        self.max_requests = max_requests
+        self.window_seconds = window_seconds
+        self.requests = defaultdict(list)
+
+    def is_allowed(self, client_ip):
+        now = time.time()
+        # BUG: No error handling if client_ip is None or empty
+        window_start = now - self.window_seconds
+        self.requests[client_ip] = [
+            t for t in self.requests[client_ip] if t > window_start
+        ]
+        if len(self.requests[client_ip]) >= self.max_requests:
+            return False
+        self.requests[client_ip].append(now)
+        return True
+
+    def get_remaining(self, client_ip):
+        # BUG: doesn't handle KeyError if client never made a request
+        count = len(self.requests[client_ip])
+        return max(0, self.max_requests - count)''',
        "lines_changed": 25,
        "has_tests": False,
        "bug_category": "missing_error_handling",
        "ground_truth_severity": "medium",
        "bug_lines": [12, 23],
        "human_labels": ["medium", "medium", "high"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.72,
    },
    # ── 3. Python SQL injection ─────────────────────────────────────────
    {
        "pr_id": "PR-003",
        "title": "Optimize database queries in ProductRepository",
        "description": "Added search functionality with direct SQL for performance. Bypasses ORM overhead for complex queries.",
        "author_experience": "senior",
        "language": "python",
        "filename": "repositories/product_repo.py",
        "diff": '''@@ -32,6 +32,22 @@ class ProductRepository:
+    def search_products(self, query_text, category=None):
+        """Fast product search bypassing ORM for performance."""
+        # BUG: SQL injection — string interpolation in query
+        sql = f"SELECT * FROM products WHERE name LIKE '%{query_text}%'"
+        if category:
+            sql += f" AND category = '{category}'"
+        cursor = self.db.execute(sql)
+        return [dict(row) for row in cursor.fetchall()]
+
+    def bulk_update_prices(self, updates):
+        """Batch price update for efficiency."""
+        for product_id, new_price in updates:
+            # BUG: Another SQL injection vector
+            self.db.execute(
+                f"UPDATE products SET price = {new_price} WHERE id = '{product_id}'"
+            )
+        self.db.commit()''',
        "lines_changed": 17,
        "has_tests": True,
        "bug_category": "sql_injection",
        "ground_truth_severity": "critical",
        "bug_lines": [36, 46],
        "human_labels": ["critical", "critical", "critical"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 4. JavaScript SQL injection ─────────────────────────────────────
    {
        "pr_id": "PR-004",
        "title": "Add user input validation",
        "description": "Added server-side validation for user registration. Validates email format and password strength.",
        "author_experience": "junior",
        "language": "javascript",
        "filename": "routes/users.js",
        "diff": '''@@ -15,4 +15,26 @@ const express = require('express');
+router.post('/register', async (req, res) => {
+    const { username, email, password } = req.body;
+
+    // Validate email format
+    if (!email.includes('@')) {
+        return res.status(400).json({ error: 'Invalid email' });
+    }
+
+    // BUG: SQL injection — concatenating user input into query
+    const checkQuery = `SELECT * FROM users WHERE username = '${username}'`;
+    const existing = await db.query(checkQuery);
+
+    if (existing.rows.length > 0) {
+        return res.status(409).json({ error: 'Username taken' });
+    }
+
+    // BUG: Password stored in plaintext — no hashing
+    const insertQuery = `INSERT INTO users (username, email, password)
+        VALUES ('${username}', '${email}', '${password}')`;
+    await db.query(insertQuery);
+
+    res.status(201).json({ message: 'User created' });
+});''',
        "lines_changed": 22,
        "has_tests": False,
        "bug_category": "sql_injection",
        "ground_truth_severity": "critical",
        "bug_lines": [24, 32],
        "human_labels": ["critical", "critical", "critical"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 5. Go race condition ────────────────────────────────────────────
    {
        "pr_id": "PR-005",
        "title": "Fix race condition in cache invalidation",
        "description": "Updated cache invalidation to handle concurrent access patterns. Added TTL-based expiry.",
        "author_experience": "mid",
        "language": "go",
        "filename": "pkg/cache/manager.go",
        "diff": '''@@ -18,6 +18,30 @@ type CacheManager struct {
+func (cm *CacheManager) Get(key string) (interface{}, bool) {
+    // BUG: No mutex lock — concurrent reads/writes cause data race
+    entry, exists := cm.store[key]
+    if !exists {
+        return nil, false
+    }
+    if time.Now().After(entry.ExpiresAt) {
+        // BUG: Deleting without lock while other goroutines may read
+        delete(cm.store, key)
+        return nil, false
+    }
+    return entry.Value, true
+}
+
+func (cm *CacheManager) Set(key string, value interface{}, ttl time.Duration) {
+    // BUG: No mutex lock on write — race with Get/Delete
+    cm.store[key] = CacheEntry{
+        Value:     value,
+        ExpiresAt: time.Now().Add(ttl),
+    }
+}
+
+func (cm *CacheManager) Delete(key string) {
+    delete(cm.store, key)
+}''',
        "lines_changed": 24,
        "has_tests": True,
        "bug_category": "race_condition",
        "ground_truth_severity": "high",
        "bug_lines": [20, 27, 34],
        "human_labels": ["high", "high", "critical"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.65,
    },
    # ── 6. Python security vulnerability ────────────────────────────────
    {
        "pr_id": "PR-006",
        "title": "Refactor authentication middleware",
        "description": "Simplified auth middleware and added JWT token verification. Moved secret key to config.",
        "author_experience": "senior",
        "language": "python",
        "filename": "middleware/auth.py",
        "diff": '''@@ -5,8 +5,28 @@ from functools import wraps
+import jwt
+import os
+
+# BUG: Hardcoded secret key — should use env var or vault
+SECRET_KEY = "super_secret_key_12345"
+
+def verify_token(token):
+    """Verify JWT token and return payload."""
+    try:
+        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
+        return payload
+    except jwt.ExpiredSignatureError:
+        return None
+    # BUG: Missing InvalidTokenError handling — crashes on malformed tokens
+
+def require_auth(f):
+    @wraps(f)
+    def decorated(*args, **kwargs):
+        token = request.headers.get("Authorization", "").replace("Bearer ", "")
+        # BUG: Token value exposed in debug log
+        print(f"DEBUG: verifying token {token}")
+        payload = verify_token(token)
+        if not payload:
+            return jsonify({"error": "Unauthorized"}), 401
+        return f(*args, **kwargs)
+    return decorated''',
        "lines_changed": 23,
        "has_tests": False,
        "bug_category": "security_vulnerability",
        "ground_truth_severity": "critical",
        "bug_lines": [9, 19, 25],
        "human_labels": ["critical", "critical", "critical"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 7. JavaScript logic error ───────────────────────────────────────
    {
        "pr_id": "PR-007",
        "title": "Add pagination to list endpoints",
        "description": "Implemented cursor-based pagination for all list endpoints. Added page_size parameter.",
        "author_experience": "mid",
        "language": "javascript",
        "filename": "controllers/listController.js",
        "diff": '''@@ -10,4 +10,28 @@ const { Op } = require('sequelize');
+async function listItems(req, res) {
+    const page = parseInt(req.query.page) || 1;
+    const pageSize = parseInt(req.query.page_size) || 20;
+
+    // BUG: Off-by-one error — first page skips first item
+    const offset = page * pageSize;
+    // Should be: (page - 1) * pageSize
+
+    const { count, rows } = await Item.findAndCountAll({
+        limit: pageSize,
+        offset: offset,
+        order: [['createdAt', 'DESC']],
+    });
+
+    // BUG: Total pages calculation wrong for exact multiples
+    const totalPages = Math.floor(count / pageSize);
+    // Should be: Math.ceil(count / pageSize)
+
+    res.json({
+        items: rows,
+        pagination: {
+            page,
+            pageSize,
+            totalPages,
+            totalItems: count,
+        },
+    });
+}''',
        "lines_changed": 24,
        "has_tests": True,
        "bug_category": "logic_error",
        "ground_truth_severity": "medium",
        "bug_lines": [15, 24],
        "human_labels": ["medium", "medium", "medium"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 8. Python style only ────────────────────────────────────────────
    {
        "pr_id": "PR-008",
        "title": "Update README formatting",
        "description": "Cleaned up README formatting, fixed markdown tables, and updated badge URLs.",
        "author_experience": "senior",
        "language": "python",
        "filename": "utils/formatter.py",
        "diff": '''@@ -1,15 +1,15 @@
-def formatUserName(firstName, lastName):
-    """format the user name"""
-    FullName = firstName + " " + lastName
-    return FullName
+def format_user_name(first_name, last_name):
+    """Format the user's display name from first and last name components."""
+    full_name = first_name + " " + last_name
+    return full_name
 
-def getUserAge(birthYear):
-    import datetime
-    currentYear = datetime.datetime.now().year
-    AGE = currentYear - birthYear
-    return AGE
+def get_user_age(birth_year):
+    """Calculate user age from birth year."""
+    import datetime
+    current_year = datetime.datetime.now().year
+    age = current_year - birth_year
+    return age''',
        "lines_changed": 12,
        "has_tests": True,
        "bug_category": "style_only",
        "ground_truth_severity": "none",
        "bug_lines": [],
        "human_labels": ["none", "none", "none"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 9. Java missing error handling ──────────────────────────────────
    {
        "pr_id": "PR-009",
        "title": "Add file upload endpoint",
        "description": "New endpoint for uploading user profile images. Supports JPEG and PNG up to 5MB.",
        "author_experience": "junior",
        "language": "java",
        "filename": "src/main/java/com/app/FileUploadController.java",
        "diff": '''@@ -20,6 +20,30 @@ public class FileUploadController {
+    @PostMapping("/upload")
+    public ResponseEntity<String> uploadFile(@RequestParam("file") MultipartFile file) {
+        String filename = file.getOriginalFilename();
+        // BUG: No file type validation — accepts any file type
+        // BUG: No file size check despite 5MB limit in description
+
+        String uploadDir = "/uploads/" + filename;
+        // BUG: Path traversal vulnerability — filename could contain ../
+
+        try {
+            File dest = new File(uploadDir);
+            file.transferTo(dest);
+        } catch (IOException e) {
+            // BUG: Swallows exception — returns 200 even on failure
+            System.out.println("Upload failed");
+        }
+
+        return ResponseEntity.ok("File uploaded: " + filename);
+    }
+
+    @GetMapping("/files/{filename}")
+    public byte[] getFile(@PathVariable String filename) {
+        // BUG: No error handling if file doesn't exist
+        return Files.readAllBytes(Paths.get("/uploads/" + filename));
+    }''',
        "lines_changed": 22,
        "has_tests": False,
        "bug_category": "missing_error_handling",
        "ground_truth_severity": "medium",
        "bug_lines": [23, 24, 27, 34, 43],
        "human_labels": ["high", "medium", "medium"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.61,
    },
    # ── 10. Go performance issue ────────────────────────────────────────
    {
        "pr_id": "PR-010",
        "title": "Add metrics aggregation endpoint",
        "description": "New endpoint to aggregate user activity metrics. Computes daily, weekly, monthly summaries.",
        "author_experience": "senior",
        "language": "go",
        "filename": "pkg/metrics/aggregator.go",
        "diff": '''@@ -12,4 +12,30 @@ type MetricsAggregator struct {
+func (ma *MetricsAggregator) ComputeDailySummary(userID string) (*Summary, error) {
+    // BUG: O(n) scan of entire events table — no index usage, no date filter
+    events, err := ma.db.Query("SELECT * FROM events WHERE user_id = $1", userID)
+    if err != nil {
+        return nil, err
+    }
+
+    summary := &Summary{}
+    for events.Next() {
+        var e Event
+        events.Scan(&e.ID, &e.Type, &e.Timestamp, &e.UserID, &e.Data)
+
+        // BUG: Parsing timestamp in tight loop — should pre-compute
+        t, _ := time.Parse(time.RFC3339, e.Timestamp)
+        if t.Day() == time.Now().Day() {
+            summary.Count++
+            summary.TotalDuration += e.Duration
+        }
+    }
+
+    // BUG: N+1 query — fetches user details for each event separately
+    for i, e := range summary.Events {
+        user, _ := ma.db.QueryRow("SELECT name FROM users WHERE id = $1", e.UserID)
+        summary.Events[i].UserName = user.Name
+    }
+
+    return summary, nil
+}''',
        "lines_changed": 26,
        "has_tests": True,
        "bug_category": "performance_issue",
        "ground_truth_severity": "low",
        "bug_lines": [14, 25, 32],
        "human_labels": ["low", "low", "medium"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.68,
    },
    # ── 11. Python null pointer ─────────────────────────────────────────
    {
        "pr_id": "PR-011",
        "title": "Add webhook notification handler",
        "description": "Handles incoming webhooks from payment provider. Processes payment success and failure events.",
        "author_experience": "junior",
        "language": "python",
        "filename": "handlers/webhook.py",
        "diff": '''@@ -1,0 +1,28 @@
+import json
+from flask import request, jsonify
+
+def handle_webhook():
+    payload = request.get_json()
+
+    # BUG: No null check — payload could be None if body isn't JSON
+    event_type = payload["event_type"]
+    transaction_id = payload["data"]["transaction_id"]
+
+    # BUG: No check if "data" key exists
+    amount = payload["data"]["amount"]
+    customer = payload["data"]["customer"]
+
+    if event_type == "payment_success":
+        # BUG: customer["email"] could be None
+        send_receipt(customer["email"], amount, transaction_id)
+    elif event_type == "payment_failed":
+        notify_support(transaction_id)
+
+    return jsonify({"status": "processed"}), 200
+
+def send_receipt(email, amount, txn_id):
+    """Send payment receipt email."""
+    msg = f"Payment of ${amount} received. Transaction: {txn_id}"
+    # email sending logic
+    print(f"Sending receipt to {email}: {msg}")''',
        "lines_changed": 27,
        "has_tests": False,
        "bug_category": "null_pointer",
        "ground_truth_severity": "high",
        "bug_lines": [8, 11, 17],
        "human_labels": ["high", "high", "medium"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.72,
    },
    # ── 12. JavaScript security vulnerability ───────────────────────────
    {
        "pr_id": "PR-012",
        "title": "Add session management",
        "description": "Implemented user session handling with cookie-based tokens. Added remember me functionality.",
        "author_experience": "junior",
        "language": "javascript",
        "filename": "middleware/session.js",
        "diff": '''@@ -1,0 +1,30 @@
+const crypto = require('crypto');
+
+// BUG: Weak secret — predictable session tokens
+const SESSION_SECRET = 'mysecret123';
+
+function createSession(userId) {
+    // BUG: Using MD5 — cryptographically broken hash function
+    const token = crypto.createHash('md5')
+        .update(userId + SESSION_SECRET + Date.now())
+        .digest('hex');
+
+    return {
+        token,
+        userId,
+        // BUG: No expiry set — sessions live forever
+        createdAt: new Date().toISOString(),
+    };
+}
+
+function validateSession(token) {
+    // BUG: Timing attack vulnerability — string comparison
+    const session = sessions.find(s => s.token === token);
+    if (!session) return null;
+
+    // BUG: No check for session expiry
+    return session.userId;
+}
+
+const sessions = [];
+module.exports = { createSession, validateSession };''',
        "lines_changed": 29,
        "has_tests": False,
        "bug_category": "security_vulnerability",
        "ground_truth_severity": "critical",
        "bug_lines": [4, 8, 16, 22, 26],
        "human_labels": ["critical", "critical", "high"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.78,
    },
    # ── 13. Python logic error ──────────────────────────────────────────
    {
        "pr_id": "PR-013",
        "title": "Implement discount calculation engine",
        "description": "New pricing engine with tiered discounts. Supports percentage and fixed amount discounts.",
        "author_experience": "mid",
        "language": "python",
        "filename": "pricing/discount_engine.py",
        "diff": '''@@ -1,0 +1,32 @@
+class DiscountEngine:
+    TIER_THRESHOLDS = {
+        "bronze": 0,
+        "silver": 1000,
+        "gold": 5000,
+        "platinum": 10000,
+    }
+
+    def calculate_discount(self, order_total, customer_tier, coupon_code=None):
+        discount = 0.0
+
+        # Tier-based discount
+        tier_rates = {"bronze": 0.0, "silver": 0.05, "gold": 0.10, "platinum": 0.15}
+        # BUG: Missing KeyError handling for unknown tier
+        discount += order_total * tier_rates[customer_tier]
+
+        # Coupon discount
+        if coupon_code:
+            coupon_discount = self._lookup_coupon(coupon_code)
+            # BUG: Discounts stack without cap — can exceed order total
+            discount += coupon_discount
+
+        # BUG: Off-by-one in boundary check — gold customers at exactly 5000 get silver rate
+        if order_total >= self.TIER_THRESHOLDS.get(customer_tier, 0):
+            discount *= 1.0  # threshold met, keep discount
+        else:
+            discount *= 0.5  # below threshold, halve it
+
+        return discount
+
+    def _lookup_coupon(self, code):
+        coupons = {"SAVE10": 10.0, "SAVE20": 20.0, "HALF50": 50.0}
+        return coupons.get(code, 0.0)''',
        "lines_changed": 32,
        "has_tests": True,
        "bug_category": "logic_error",
        "ground_truth_severity": "medium",
        "bug_lines": [15, 21, 24],
        "human_labels": ["medium", "medium", "high"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.72,
    },
    # ── 14. Go null pointer ─────────────────────────────────────────────
    {
        "pr_id": "PR-014",
        "title": "Add gRPC health check service",
        "description": "Implemented standard gRPC health check protocol for k8s liveness and readiness probes.",
        "author_experience": "mid",
        "language": "go",
        "filename": "pkg/health/checker.go",
        "diff": '''@@ -10,4 +10,28 @@ type HealthChecker struct {
+func (hc *HealthChecker) Check(ctx context.Context, req *pb.HealthCheckRequest) (*pb.HealthCheckResponse, error) {
+    service := req.GetService()
+
+    // BUG: No nil check on dependency map lookup
+    dep := hc.dependencies[service]
+    // dep could be nil if service name not registered
+
+    status := dep.Status()  // BUG: nil pointer dereference if dep is nil
+
+    response := &pb.HealthCheckResponse{
+        Status: status,
+    }
+
+    // Check sub-dependencies
+    for _, subDep := range dep.SubDependencies {
+        // BUG: No nil check on subDep
+        subStatus := subDep.Status()
+        if subStatus != pb.HealthCheckResponse_SERVING {
+            response.Status = pb.HealthCheckResponse_NOT_SERVING
+        }
+    }
+
+    return response, nil
+}''',
        "lines_changed": 22,
        "has_tests": False,
        "bug_category": "null_pointer",
        "ground_truth_severity": "high",
        "bug_lines": [15, 18, 27],
        "human_labels": ["high", "high", "high"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 15. Python performance issue ────────────────────────────────────
    {
        "pr_id": "PR-015",
        "title": "Add report generation module",
        "description": "Generates PDF reports for quarterly analytics. Aggregates data from multiple tables.",
        "author_experience": "senior",
        "language": "python",
        "filename": "reports/generator.py",
        "diff": '''@@ -8,4 +8,30 @@ class ReportGenerator:
+    def generate_quarterly_report(self, quarter, year):
+        """Generate full quarterly report with graphics and tables."""
+        users = self.db.execute("SELECT * FROM users").fetchall()
+
+        report_data = []
+        for user in users:
+            # BUG: N+1 query — individual query per user in loop
+            orders = self.db.execute(
+                f"SELECT * FROM orders WHERE user_id = {user['id']}"
+            ).fetchall()
+
+            total = 0
+            for order in orders:
+                # BUG: Loading all order items just to sum — could use SQL SUM
+                items = self.db.execute(
+                    f"SELECT * FROM order_items WHERE order_id = {order['id']}"
+                ).fetchall()
+                total += sum(item['price'] * item['quantity'] for item in items)
+
+            report_data.append({
+                "user": user['name'],
+                "total_spend": total,
+                "order_count": len(orders),
+            })
+
+        # BUG: Sorting entire list in memory instead of ORDER BY in SQL
+        report_data.sort(key=lambda x: x['total_spend'], reverse=True)
+
+        return report_data''',
        "lines_changed": 28,
        "has_tests": True,
        "bug_category": "performance_issue",
        "ground_truth_severity": "low",
        "bug_lines": [16, 22, 34],
        "human_labels": ["low", "medium", "low"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.68,
    },
    # ── 16. Java race condition ─────────────────────────────────────────
    {
        "pr_id": "PR-016",
        "title": "Implement connection pool manager",
        "description": "Custom connection pool for database connections. Supports max connections and connection reuse.",
        "author_experience": "mid",
        "language": "java",
        "filename": "src/main/java/com/app/ConnectionPool.java",
        "diff": '''@@ -15,6 +15,32 @@ public class ConnectionPool {
+    private List<Connection> available = new ArrayList<>();
+    private List<Connection> inUse = new ArrayList<>();
+    private int maxConnections = 10;
+
+    public Connection getConnection() {
+        // BUG: No synchronization — multiple threads can get same connection
+        if (available.isEmpty()) {
+            if (inUse.size() < maxConnections) {
+                Connection conn = createConnection();
+                inUse.add(conn);
+                return conn;
+            }
+            // BUG: Busy wait without backoff — CPU spin
+            while (available.isEmpty()) {
+                // spin
+            }
+        }
+        // BUG: Race condition — another thread could take last connection
+        Connection conn = available.remove(0);
+        inUse.add(conn);
+        return conn;
+    }
+
+    public void releaseConnection(Connection conn) {
+        // BUG: No validation that conn is actually from this pool
+        inUse.remove(conn);
+        available.add(conn);
+    }''',
        "lines_changed": 26,
        "has_tests": False,
        "bug_category": "race_condition",
        "ground_truth_severity": "high",
        "bug_lines": [21, 28, 33, 39],
        "human_labels": ["high", "critical", "high"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.65,
    },
    # ── 17. JavaScript missing error handling ───────────────────────────
    {
        "pr_id": "PR-017",
        "title": "Add WebSocket chat handler",
        "description": "Real-time chat implementation using WebSocket. Supports direct messages and group channels.",
        "author_experience": "junior",
        "language": "javascript",
        "filename": "handlers/chat.js",
        "diff": '''@@ -1,0 +1,30 @@
+const WebSocket = require('ws');
+
+function handleConnection(ws) {
+    ws.on('message', (data) => {
+        // BUG: No try-catch — invalid JSON crashes the server
+        const message = JSON.parse(data);
+
+        // BUG: No validation of message.type
+        if (message.type === 'direct') {
+            // BUG: No check if recipient exists
+            const recipient = connectedUsers[message.to];
+            recipient.send(JSON.stringify({
+                from: message.from,
+                text: message.text,
+                timestamp: new Date().toISOString(),
+            }));
+        } else if (message.type === 'channel') {
+            // BUG: No check if channel exists
+            channels[message.channel].forEach(user => {
+                user.send(JSON.stringify(message));
+            });
+        }
+    });
+
+    ws.on('close', () => {
+        // BUG: No cleanup of user from connectedUsers map
+        console.log('Client disconnected');
+    });
+}
+
+const connectedUsers = {};''',
        "lines_changed": 30,
        "has_tests": False,
        "bug_category": "missing_error_handling",
        "ground_truth_severity": "medium",
        "bug_lines": [6, 8, 11, 19, 27],
        "human_labels": ["medium", "high", "medium"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.61,
    },
    # ── 18. Python security vulnerability ───────────────────────────────
    {
        "pr_id": "PR-018",
        "title": "Add admin API key management",
        "description": "Admin panel for managing API keys. Allows creation, revocation, and listing of keys.",
        "author_experience": "junior",
        "language": "python",
        "filename": "admin/api_keys.py",
        "diff": '''@@ -1,0 +1,32 @@
+import hashlib
+import os
+from datetime import datetime
+
+class APIKeyManager:
+    def __init__(self):
+        self.keys = {}
+
+    def create_key(self, user_id, permissions):
+        # BUG: Using MD5 for key generation — weak hash
+        raw_key = hashlib.md5(
+            f"{user_id}{datetime.now()}".encode()
+        ).hexdigest()
+
+        # BUG: Storing key in plaintext — should store hash only
+        self.keys[raw_key] = {
+            "user_id": user_id,
+            "permissions": permissions,
+            "created_at": datetime.now().isoformat(),
+            "key_plaintext": raw_key,  # BUG: Storing plaintext key
+        }
+
+        return raw_key
+
+    def validate_key(self, key):
+        # BUG: No rate limiting on key validation — brute force possible
+        return key in self.keys
+
+    def list_keys(self, user_id):
+        # BUG: Returns full key data including plaintext — information leak
+        return [v for v in self.keys.values() if v["user_id"] == user_id]''',
        "lines_changed": 31,
        "has_tests": False,
        "bug_category": "security_vulnerability",
        "ground_truth_severity": "critical",
        "bug_lines": [11, 16, 21, 27, 31],
        "human_labels": ["critical", "critical", "critical"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 19. Go logic error ──────────────────────────────────────────────
    {
        "pr_id": "PR-019",
        "title": "Implement retry mechanism with backoff",
        "description": "Added exponential backoff retry for external API calls. Configurable max retries and base delay.",
        "author_experience": "mid",
        "language": "go",
        "filename": "pkg/retry/backoff.go",
        "diff": '''@@ -8,4 +8,28 @@ type RetryConfig struct {
+func WithRetry(config RetryConfig, fn func() error) error {
+    var lastErr error
+
+    for attempt := 0; attempt < config.MaxRetries; attempt++ {
+        err := fn()
+        if err == nil {
+            return nil
+        }
+        lastErr = err
+
+        // BUG: Delay doesn't actually use exponential backoff
+        // Should be: baseDelay * 2^attempt
+        delay := config.BaseDelay * time.Duration(attempt)
+        // When attempt=0, delay is 0 — no backoff on first retry
+
+        // BUG: No jitter — all clients retry at exact same time (thundering herd)
+        time.Sleep(delay)
+
+        // BUG: No context cancellation check — retries continue even if canceled
+    }
+
+    // BUG: Returns nil instead of lastErr when all retries exhausted
+    // due to loop boundary — attempt reaches MaxRetries and exits
+    return nil  // Should return lastErr
+}''',
        "lines_changed": 22,
        "has_tests": True,
        "bug_category": "logic_error",
        "ground_truth_severity": "medium",
        "bug_lines": [19, 22, 24, 29],
        "human_labels": ["medium", "medium", "high"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.72,
    },
    # ── 20. Java style only ─────────────────────────────────────────────
    {
        "pr_id": "PR-020",
        "title": "Refactor StringUtils for readability",
        "description": "Cleaned up StringUtils class. Renamed methods to follow Java conventions, added Javadoc.",
        "author_experience": "senior",
        "language": "java",
        "filename": "src/main/java/com/app/StringUtils.java",
        "diff": '''@@ -1,20 +1,20 @@
-public class StringUtils {
-    public static String CAMELCASE(String input) {
-        String[] parts = input.split("_");
-        StringBuilder sb = new StringBuilder();
-        for (String p : parts) {
-            sb.append(p.substring(0, 1).toUpperCase());
-            sb.append(p.substring(1).toLowerCase());
-        }
-        return sb.toString();
-    }
-    public static boolean checkEmpty(String s){
-        if(s == null) return true;
-        if(s.trim().length() == 0) return true;
-        return false;
-    }
-}
+/**
+ * Utility class for common string operations.
+ */
+public class StringUtils {
+    /**
+     * Convert snake_case to CamelCase.
+     */
+    public static String toCamelCase(String input) {
+        String[] parts = input.split("_");
+        StringBuilder sb = new StringBuilder();
+        for (String part : parts) {
+            sb.append(part.substring(0, 1).toUpperCase());
+            sb.append(part.substring(1).toLowerCase());
+        }
+        return sb.toString();
+    }
+
+    /**
+     * Check if a string is null or blank.
+     */
+    public static boolean isBlank(String value) {
+        return value == null || value.trim().isEmpty();
+    }
+}''',
        "lines_changed": 24,
        "has_tests": True,
        "bug_category": "style_only",
        "ground_truth_severity": "none",
        "bug_lines": [],
        "human_labels": ["none", "none", "none"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 21. Rust unsafe memory ─────────────────────────────────────────
    {
        "pr_id": "PR-021",
        "title": "Add FFI bindings for crypto library",
        "description": "Created Rust FFI wrappers for the C crypto library. Handles key generation and signing.",
        "author_experience": "mid",
        "language": "rust",
        "filename": "src/ffi/crypto.rs",
        "diff": '''@@ -1,0 +1,18 @@
+use std::ffi::{CStr, CString};
+use std::os::raw::c_char;
+
+pub fn sign_message(key: *const c_char, msg: &str) -> String {
+    // BUG: Dereferencing raw pointer without null check
+    let key_str = unsafe { CStr::from_ptr(key) }.to_str().unwrap();
+    let c_msg = CString::new(msg).unwrap();
+    // BUG: Result of FFI call not checked for errors
+    let sig = unsafe { ffi_sign(key_str.as_ptr(), c_msg.as_ptr()) };
+    unsafe { CStr::from_ptr(sig) }.to_string_lossy().into_owned()
+}''',
        "lines_changed": 11,
        "has_tests": False,
        "bug_category": "null_pointer",
        "ground_truth_severity": "high",
        "bug_lines": [5, 8],
        "human_labels": ["high", "critical", "high"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.58,
    },
    # ── 22. Go SQL injection ─────────────────────────────────────────────
    {
        "pr_id": "PR-022",
        "title": "Add search endpoint for products",
        "description": "New product search API with filtering by category and price range.",
        "author_experience": "junior",
        "language": "go",
        "filename": "handlers/search.go",
        "diff": '''@@ -1,0 +1,16 @@
+func SearchProducts(w http.ResponseWriter, r *http.Request) {
+    query := r.URL.Query().Get("q")
+    category := r.URL.Query().Get("category")
+    // BUG: Direct string interpolation in SQL — SQL injection
+    sql := fmt.Sprintf("SELECT * FROM products WHERE name LIKE '%%%s%%' AND category = '%s'", query, category)
+    rows, err := db.Query(sql)
+    if err != nil {
+        http.Error(w, "Search failed", 500)
+        return
+    }
+    defer rows.Close()
+    json.NewEncoder(w).Encode(scanProducts(rows))
+}''',
        "lines_changed": 13,
        "has_tests": False,
        "bug_category": "sql_injection",
        "ground_truth_severity": "critical",
        "bug_lines": [4, 5],
        "human_labels": ["critical", "critical", "critical"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 23. TypeScript race condition ─────────────────────────────────────
    {
        "pr_id": "PR-023",
        "title": "Add real-time inventory tracker",
        "description": "Tracks product inventory in real-time with WebSocket updates.",
        "author_experience": "mid",
        "language": "typescript",
        "filename": "src/inventory/tracker.ts",
        "diff": '''@@ -1,0 +1,20 @@
+let inventory: Map<string, number> = new Map();
+
+export async function reserveItem(productId: string, qty: number): Promise<boolean> {
+    const current = inventory.get(productId) || 0;
+    // BUG: TOCTOU race — another request could modify between check and update
+    if (current >= qty) {
+        // Simulate async DB write
+        await db.updateInventory(productId, current - qty);
+        inventory.set(productId, current - qty);
+        return true;
+    }
+    return false;
+}
+
+export async function restockItem(productId: string, qty: number) {
+    const current = inventory.get(productId) || 0;
+    // BUG: Same race condition on restock
+    inventory.set(productId, current + qty);
+    await db.updateInventory(productId, current + qty);
+}''',
        "lines_changed": 20,
        "has_tests": False,
        "bug_category": "race_condition",
        "ground_truth_severity": "high",
        "bug_lines": [5, 17],
        "human_labels": ["high", "high", "critical"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.65,
    },
    # ── 24. Python logic error ─────────────────────────────────────────
    {
        "pr_id": "PR-024",
        "title": "Implement discount calculation engine",
        "description": "Multi-tier discount system supporting percentage, fixed, and BOGO promotions.",
        "author_experience": "junior",
        "language": "python",
        "filename": "pricing/discounts.py",
        "diff": '''@@ -1,0 +1,22 @@
+def calculate_discount(cart_items, promotions):
+    total_discount = 0
+    for item in cart_items:
+        for promo in promotions:
+            if promo["type"] == "percentage":
+                # BUG: Off-by-one — divides by 10 not 100
+                total_discount += item["price"] * promo["value"] / 10
+            elif promo["type"] == "fixed":
+                total_discount += promo["value"]
+            elif promo["type"] == "bogo":
+                # BUG: Applies BOGO to every item, not just qualifying ones
+                total_discount += item["price"]
+    # BUG: No cap — discount can exceed cart total
+    return total_discount''',
        "lines_changed": 14,
        "has_tests": False,
        "bug_category": "logic_error",
        "ground_truth_severity": "medium",
        "bug_lines": [6, 11, 13],
        "human_labels": ["medium", "high", "medium"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.55,
    },
    # ── 25. Java performance ─────────────────────────────────────────────
    {
        "pr_id": "PR-025",
        "title": "Generate monthly analytics report",
        "description": "Aggregates transaction data for monthly PDF reports with charts.",
        "author_experience": "mid",
        "language": "java",
        "filename": "src/main/java/com/app/ReportGenerator.java",
        "diff": '''@@ -20,0 +20,18 @@
+    public Report generateMonthlyReport(int month, int year) {
+        List<Transaction> all = transactionRepo.findAll();
+        // BUG: Loading ALL transactions then filtering in memory — N+1
+        List<Transaction> monthly = new ArrayList<>();
+        for (Transaction t : all) {
+            if (t.getMonth() == month && t.getYear() == year) {
+                monthly.add(t);
+            }
+        }
+        // BUG: Sorting entire list for each category instead of groupBy
+        for (String category : getCategories()) {
+            List<Transaction> catTxns = monthly.stream()
+                .filter(t -> t.getCategory().equals(category))
+                .sorted(Comparator.comparing(Transaction::getAmount))
+                .collect(Collectors.toList());
+            report.addSection(category, aggregate(catTxns));
+        }
+        return report;
+    }''',
        "lines_changed": 18,
        "has_tests": True,
        "bug_category": "performance_issue",
        "ground_truth_severity": "low",
        "bug_lines": [2, 10],
        "human_labels": ["low", "low", "medium"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.60,
    },
    # ── 26. Python security ─────────────────────────────────────────────
    {
        "pr_id": "PR-026",
        "title": "Add JWT authentication middleware",
        "description": "JWT-based auth middleware for Flask. Validates tokens and extracts user claims.",
        "author_experience": "junior",
        "language": "python",
        "filename": "middleware/auth.py",
        "diff": '''@@ -1,0 +1,20 @@
+import jwt
+from functools import wraps
+from flask import request, jsonify
+
+SECRET_KEY = "my-super-secret-key-12345"  # BUG: Hardcoded secret
+
+def require_auth(f):
+    @wraps(f)
+    def decorated(*args, **kwargs):
+        token = request.headers.get("Authorization", "").replace("Bearer ", "")
+        try:
+            # BUG: algorithm not restricted — allows "none" algorithm attack
+            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256", "none"])
+            request.user = payload
+        except jwt.ExpiredSignatureError:
+            return jsonify({"error": "Token expired"}), 401
+        except:  # BUG: Bare except hides real errors
+            return jsonify({"error": "Invalid token"}), 401
+        return f(*args, **kwargs)
+    return decorated''',
        "lines_changed": 20,
        "has_tests": False,
        "bug_category": "security_vulnerability",
        "ground_truth_severity": "critical",
        "bug_lines": [5, 12, 17],
        "human_labels": ["critical", "critical", "critical"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 27. Go missing error handling ─────────────────────────────────
    {
        "pr_id": "PR-027",
        "title": "Add file upload handler",
        "description": "Handles multipart file uploads with size validation and storage.",
        "author_experience": "junior",
        "language": "go",
        "filename": "handlers/upload.go",
        "diff": '''@@ -1,0 +1,18 @@
+func UploadFile(w http.ResponseWriter, r *http.Request) {
+    // BUG: No max file size limit — DoS via large upload
+    file, header, _ := r.FormFile("upload")
+    defer file.Close()
+
+    // BUG: Ignoring error from FormFile
+    dst, _ := os.Create(filepath.Join("/uploads", header.Filename))
+    // BUG: Path traversal — filename could contain ../
+    defer dst.Close()
+
+    io.Copy(dst, file)
+    // BUG: io.Copy error not checked
+    w.WriteHeader(http.StatusOK)
+    fmt.Fprintf(w, "Uploaded: %s", header.Filename)
+}''',
        "lines_changed": 15,
        "has_tests": False,
        "bug_category": "missing_error_handling",
        "ground_truth_severity": "medium",
        "bug_lines": [2, 3, 7, 11],
        "human_labels": ["high", "medium", "medium"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.55,
    },
    # ── 28. Ruby style only ──────────────────────────────────────────────
    {
        "pr_id": "PR-028",
        "title": "Refactor User model validations",
        "description": "Cleaned up user validations to use Rails built-in validators.",
        "author_experience": "senior",
        "language": "ruby",
        "filename": "app/models/user.rb",
        "diff": '''@@ -1,15 +1,15 @@
 class User < ApplicationRecord
-  validate :check_email
-  validate :check_name
+  validates :email, presence: true, format: { with: URI::MailTo::EMAIL_REGEXP }
+  validates :name, presence: true, length: { minimum: 2, maximum: 100 }
+  validates :role, inclusion: { in: %w[admin editor viewer] }

-  def check_email
-    errors.add(:email, "invalid") unless email =~ /\A[\w+\-.]+@[a-z\d\-]+\.[a-z]+\z/i
-  end
-
-  def check_name
-    errors.add(:name, "too short") if name.length < 2
-  end
+  before_save :normalize_email
+
+  private
+
+  def normalize_email
+    self.email = email.downcase.strip
+  end
 end''',
        "lines_changed": 15,
        "has_tests": True,
        "bug_category": "style_only",
        "ground_truth_severity": "none",
        "bug_lines": [],
        "human_labels": ["none", "none", "none"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 29. Python race condition ─────────────────────────────────────────
    {
        "pr_id": "PR-029",
        "title": "Add distributed task queue processor",
        "description": "Background task processor using Redis for job coordination.",
        "author_experience": "mid",
        "language": "python",
        "filename": "workers/task_processor.py",
        "diff": '''@@ -1,0 +1,22 @@
+import redis
+import json
+
+class TaskProcessor:
+    def __init__(self):
+        self.redis = redis.Redis()
+        self.processing = {}
+
+    def claim_task(self, queue_name):
+        task_data = self.redis.lpop(queue_name)
+        if task_data:
+            task = json.loads(task_data)
+            # BUG: No atomic claim — two workers can pop same task
+            self.processing[task["id"]] = task
+            return task
+        return None
+
+    def complete_task(self, task_id):
+        # BUG: No check if task is still ours (could have been reclaimed)
+        task = self.processing.pop(task_id, None)
+        self.redis.hset("completed", task_id, json.dumps(task))
+        # BUG: task could be None here''',
        "lines_changed": 22,
        "has_tests": False,
        "bug_category": "race_condition",
        "ground_truth_severity": "high",
        "bug_lines": [12, 19, 22],
        "human_labels": ["high", "high", "high"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 30. TypeScript null pointer ────────────────────────────────────
    {
        "pr_id": "PR-030",
        "title": "Add user preferences API",
        "description": "CRUD endpoints for user preferences with nested settings.",
        "author_experience": "junior",
        "language": "typescript",
        "filename": "src/api/preferences.ts",
        "diff": '''@@ -1,0 +1,18 @@
+interface UserPrefs { theme: string; notifications: { email: boolean; push: boolean } }
+
+export function getNotifSetting(prefs: UserPrefs | null, key: string): boolean {
+    // BUG: No null check on prefs
+    return prefs.notifications[key];
+}
+
+export function updatePrefs(userId: string, updates: Partial<UserPrefs>) {
+    const current = db.getPrefs(userId);
+    // BUG: current could be undefined for new users
+    const merged = { ...current, ...updates };
+    // BUG: Deep merge needed for nested notifications object
+    db.savePrefs(userId, merged);
+}''',
        "lines_changed": 14,
        "has_tests": False,
        "bug_category": "null_pointer",
        "ground_truth_severity": "high",
        "bug_lines": [4, 10, 12],
        "human_labels": ["high", "medium", "high"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.60,
    },
    # ── 31. Go performance ─────────────────────────────────────────────
    {
        "pr_id": "PR-031",
        "title": "Add log aggregation pipeline",
        "description": "Aggregates application logs from multiple sources for analysis.",
        "author_experience": "mid",
        "language": "go",
        "filename": "pipeline/aggregator.go",
        "diff": '''@@ -1,0 +1,18 @@
+func AggregateLogs(sources []LogSource, window time.Duration) []AggregatedLog {
+    var results []AggregatedLog
+    for _, source := range sources {
+        // BUG: Loading all logs into memory — no streaming
+        logs := source.FetchAll()
+        for _, log := range logs {
+            // BUG: O(n^2) — searching results linearly for each log
+            for i, r := range results {
+                if r.Key == log.Key {
+                    results[i].Count++
+                    break
+                }
+            }
+        }
+    }
+    // BUG: No deduplication across sources
+    return results
+}''',
        "lines_changed": 18,
        "has_tests": True,
        "bug_category": "performance_issue",
        "ground_truth_severity": "low",
        "bug_lines": [4, 7, 16],
        "human_labels": ["low", "low", "low"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 32. Java SQL injection ─────────────────────────────────────────
    {
        "pr_id": "PR-032",
        "title": "Add audit log search",
        "description": "Admin audit log search with date range and action type filters.",
        "author_experience": "junior",
        "language": "java",
        "filename": "src/main/java/com/app/AuditSearch.java",
        "diff": '''@@ -1,0 +1,14 @@
+public List<AuditLog> searchAuditLogs(String userId, String action, String dateRange) {
+    // BUG: SQL injection via string concatenation
+    String query = "SELECT * FROM audit_logs WHERE user_id = '" + userId + "'";
+    if (action != null) {
+        query += " AND action = '" + action + "'";
+    }
+    if (dateRange != null) {
+        query += " AND created_at > '" + dateRange + "'";
+    }
+    // BUG: Using raw Statement instead of PreparedStatement
+    Statement stmt = connection.createStatement();
+    ResultSet rs = stmt.executeQuery(query);
+    return mapResults(rs);
+}''',
        "lines_changed": 14,
        "has_tests": False,
        "bug_category": "sql_injection",
        "ground_truth_severity": "critical",
        "bug_lines": [2, 3, 5, 8, 10],
        "human_labels": ["critical", "critical", "critical"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 33. Python logic error ─────────────────────────────────────────
    {
        "pr_id": "PR-033",
        "title": "Implement A/B test bucketing",
        "description": "Assigns users to experiment variants based on hash bucketing.",
        "author_experience": "mid",
        "language": "python",
        "filename": "experiments/bucketing.py",
        "diff": '''@@ -1,0 +1,18 @@
+import hashlib
+
+def assign_variant(user_id, experiment_name, variants, traffic_pct=100):
+    hash_input = f"{user_id}:{experiment_name}"
+    hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
+    # BUG: Modulo 100 but traffic_pct could be 0 → ZeroDivisionError
+    bucket = hash_val % traffic_pct
+    # BUG: Off-by-one — if traffic_pct=50 and bucket=50, user is excluded
+    if bucket >= traffic_pct:
+        return None
+    variant_idx = bucket % len(variants)
+    return variants[variant_idx]''',
        "lines_changed": 12,
        "has_tests": True,
        "bug_category": "logic_error",
        "ground_truth_severity": "medium",
        "bug_lines": [6, 8],
        "human_labels": ["medium", "medium", "high"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.60,
    },
    # ── 34. Rust missing error handling ────────────────────────────────
    {
        "pr_id": "PR-034",
        "title": "Add config file parser",
        "description": "Parses YAML configuration files with environment variable substitution.",
        "author_experience": "junior",
        "language": "rust",
        "filename": "src/config/parser.rs",
        "diff": '''@@ -1,0 +1,16 @@
+use std::fs;
+use std::env;
+
+pub fn load_config(path: &str) -> Config {
+    // BUG: unwrap on file read — panics if file missing
+    let content = fs::read_to_string(path).unwrap();
+    // BUG: unwrap on YAML parse — panics on malformed config
+    let mut config: Config = serde_yaml::from_str(&content).unwrap();
+    // Substitute env vars
+    for (key, val) in &mut config.values {
+        if val.starts_with("${") && val.ends_with("}") {
+            let env_key = &val[2..val.len()-1];
+            // BUG: unwrap on env var — panics if not set
+            *val = env::var(env_key).unwrap();
+        }
+    }
+    config
+}''',
        "lines_changed": 16,
        "has_tests": False,
        "bug_category": "missing_error_handling",
        "ground_truth_severity": "medium",
        "bug_lines": [5, 7, 13],
        "human_labels": ["medium", "medium", "medium"],
        "human_agreement": 1.0,
        "cohen_kappa": 1.0,
    },
    # ── 35. JavaScript security ─────────────────────────────────────────
    {
        "pr_id": "PR-035",
        "title": "Add user profile image upload",
        "description": "Profile image upload with client-side preview and server storage.",
        "author_experience": "junior",
        "language": "javascript",
        "filename": "routes/profile.js",
        "diff": '''@@ -1,0 +1,16 @@
+const express = require('express');
+const fs = require('fs');
+const path = require('path');
+
+app.post('/profile/image', (req, res) => {
+    const file = req.files.avatar;
+    // BUG: No file type validation — could upload .exe, .php
+    // BUG: Path traversal — filename could contain ../
+    const savePath = path.join('/uploads/avatars', file.name);
+    file.mv(savePath, (err) => {
+        if (err) return res.status(500).send(err);
+        // BUG: Storing full server path in DB — info disclosure
+        db.updateUser(req.user.id, { avatar: savePath });
+        res.json({ url: savePath });
+    });
+});''',
        "lines_changed": 16,
        "has_tests": False,
        "bug_category": "security_vulnerability",
        "ground_truth_severity": "critical",
        "bug_lines": [7, 8, 12],
        "human_labels": ["critical", "critical", "high"],
        "human_agreement": 0.67,
        "cohen_kappa": 0.72,
    },
    # ── 36-50: Additional templates for diversity ────────────────────────
    {
        "pr_id": "PR-036", "title": "Add CSV export for reports",
        "description": "Export filtered report data to CSV with proper escaping.",
        "author_experience": "mid", "language": "python", "filename": "reports/export.py",
        "diff": '''@@ -1,0 +1,12 @@
+import csv, io
+def export_csv(data, columns):
+    output = io.StringIO()
+    writer = csv.writer(output)
+    writer.writerow(columns)
+    for row in data:
+        # BUG: No escaping — formula injection via =cmd() in cells
+        writer.writerow([row.get(c, "") for c in columns])
+    return output.getvalue()''',
        "lines_changed": 9, "has_tests": True,
        "bug_category": "security_vulnerability", "ground_truth_severity": "critical",
        "bug_lines": [7], "human_labels": ["critical", "high", "critical"],
        "human_agreement": 0.67, "cohen_kappa": 0.58,
    },
    {
        "pr_id": "PR-037", "title": "Fix pagination in user list",
        "description": "Added offset-based pagination to the user listing endpoint.",
        "author_experience": "junior", "language": "python", "filename": "api/users.py",
        "diff": '''@@ -10,6 +10,12 @@
+def list_users(page=1, per_page=20):
+    # BUG: No validation — negative page causes SQL error
+    offset = (page - 1) * per_page
+    # BUG: No upper bound on per_page — DoS via per_page=999999
+    users = db.query(f"SELECT * FROM users LIMIT {per_page} OFFSET {offset}")
+    return users''',
        "lines_changed": 6, "has_tests": False,
        "bug_category": "logic_error", "ground_truth_severity": "medium",
        "bug_lines": [2, 4], "human_labels": ["medium", "medium", "low"],
        "human_agreement": 0.67, "cohen_kappa": 0.55,
    },
    {
        "pr_id": "PR-038", "title": "Optimize image thumbnail generation",
        "description": "Batch thumbnail generation with parallel processing.",
        "author_experience": "senior", "language": "python", "filename": "media/thumbnails.py",
        "diff": '''@@ -5,8 +5,14 @@
+from concurrent.futures import ThreadPoolExecutor
+def generate_thumbnails(image_paths, sizes=[128, 256, 512]):
+    results = []
+    # BUG: No limit on thread pool — could exhaust system resources
+    with ThreadPoolExecutor() as pool:
+        for path in image_paths:
+            for size in sizes:
+                # BUG: No error handling — one failure kills entire batch
+                results.append(pool.submit(resize_image, path, size))
+    return [r.result() for r in results]''',
        "lines_changed": 10, "has_tests": True,
        "bug_category": "performance_issue", "ground_truth_severity": "low",
        "bug_lines": [4, 8], "human_labels": ["low", "medium", "low"],
        "human_agreement": 0.67, "cohen_kappa": 0.60,
    },
    {
        "pr_id": "PR-039", "title": "Refactor database connection pool",
        "description": "Replaced manual connection management with connection pooling.",
        "author_experience": "senior", "language": "java", "filename": "src/main/java/com/app/DbPool.java",
        "diff": '''@@ -1,18 +1,18 @@
-public class DbManager {
-    private Connection conn;
-    public Connection getConnection() {
-        if (conn == null) conn = DriverManager.getConnection(url);
-        return conn;
-    }
+public class DbPool {
+    private final HikariDataSource ds;
+    public DbPool(String url, int maxSize) {
+        HikariConfig config = new HikariConfig();
+        config.setJdbcUrl(url);
+        config.setMaximumPoolSize(maxSize);
+        ds = new HikariDataSource(config);
+    }
+    public Connection getConnection() throws SQLException {
+        return ds.getConnection();
+    }
 }''',
        "lines_changed": 18, "has_tests": True,
        "bug_category": "style_only", "ground_truth_severity": "none",
        "bug_lines": [], "human_labels": ["none", "none", "none"],
        "human_agreement": 1.0, "cohen_kappa": 1.0,
    },
    {
        "pr_id": "PR-040", "title": "Add email notification service",
        "description": "Sends transactional emails via SMTP with HTML templates.",
        "author_experience": "mid", "language": "python", "filename": "services/email.py",
        "diff": '''@@ -1,0 +1,14 @@
+import smtplib
+from email.mime.text import MIMEText
+class EmailService:
+    def __init__(self):
+        # BUG: Hardcoded SMTP credentials
+        self.server = smtplib.SMTP("smtp.gmail.com", 587)
+        self.server.login("app@company.com", "password123")
+    def send(self, to, subject, html_body):
+        msg = MIMEText(html_body, "html")
+        msg["Subject"] = subject
+        # BUG: No input validation on 'to' — could be used for spam
+        self.server.sendmail("app@company.com", to, msg.as_string())''',
        "lines_changed": 12, "has_tests": False,
        "bug_category": "security_vulnerability", "ground_truth_severity": "critical",
        "bug_lines": [5, 7, 11], "human_labels": ["critical", "critical", "critical"],
        "human_agreement": 1.0, "cohen_kappa": 1.0,
    },
    {
        "pr_id": "PR-041", "title": "Add caching layer for API responses",
        "description": "Redis-based response cache with TTL and invalidation.",
        "author_experience": "mid", "language": "python", "filename": "cache/api_cache.py",
        "diff": '''@@ -1,0 +1,14 @@
+import redis, json, hashlib
+cache = redis.Redis()
+def cached_response(func):
+    def wrapper(*args, **kwargs):
+        key = hashlib.md5(str(args).encode()).hexdigest()
+        cached = cache.get(key)
+        if cached:
+            return json.loads(cached)
+        result = func(*args, **kwargs)
+        # BUG: No TTL — cached data never expires
+        cache.set(key, json.dumps(result))
+        return result
+    return wrapper''',
        "lines_changed": 13, "has_tests": False,
        "bug_category": "logic_error", "ground_truth_severity": "medium",
        "bug_lines": [10], "human_labels": ["medium", "low", "medium"],
        "human_agreement": 0.67, "cohen_kappa": 0.55,
    },
    {
        "pr_id": "PR-042", "title": "Add webhook retry mechanism",
        "description": "Retries failed webhook deliveries with exponential backoff.",
        "author_experience": "junior", "language": "python", "filename": "webhooks/retry.py",
        "diff": '''@@ -1,0 +1,16 @@
+import requests, time
+def deliver_webhook(url, payload, max_retries=3):
+    for attempt in range(max_retries):
+        try:
+            resp = requests.post(url, json=payload, timeout=5)
+            if resp.status_code < 400:
+                return True
+        except requests.Timeout:
+            pass
+        # BUG: Fixed delay instead of exponential backoff
+        time.sleep(1)
+    # BUG: No dead letter queue for failed deliveries
+    return False''',
        "lines_changed": 13, "has_tests": False,
        "bug_category": "missing_error_handling", "ground_truth_severity": "medium",
        "bug_lines": [10, 12], "human_labels": ["medium", "medium", "low"],
        "human_agreement": 0.67, "cohen_kappa": 0.55,
    },
    {
        "pr_id": "PR-043", "title": "Add session management",
        "description": "Server-side session store with cookie-based session IDs.",
        "author_experience": "junior", "language": "javascript", "filename": "middleware/session.js",
        "diff": '''@@ -1,0 +1,14 @@
+const sessions = {};
+function createSession(userId) {
+    // BUG: Predictable session ID — sequential counter
+    const sessionId = String(Object.keys(sessions).length + 1);
+    sessions[sessionId] = { userId, created: Date.now() };
+    return sessionId;
+}
+function getSession(req) {
+    const sid = req.cookies.session_id;
+    // BUG: No session expiry check
+    return sessions[sid] || null;
+}''',
        "lines_changed": 12, "has_tests": False,
        "bug_category": "security_vulnerability", "ground_truth_severity": "critical",
        "bug_lines": [3, 10], "human_labels": ["critical", "critical", "high"],
        "human_agreement": 0.67, "cohen_kappa": 0.65,
    },
    {
        "pr_id": "PR-044", "title": "Add data migration script",
        "description": "Migrates user data from legacy schema to new normalized tables.",
        "author_experience": "mid", "language": "python", "filename": "migrations/migrate_users.py",
        "diff": '''@@ -1,0 +1,16 @@
+def migrate_users(old_db, new_db):
+    users = old_db.execute("SELECT * FROM legacy_users").fetchall()
+    for user in users:
+        # BUG: No transaction — partial migration on failure
+        new_db.execute("INSERT INTO users (id, name) VALUES (?, ?)",
+                      (user["id"], user["name"]))
+        if user.get("address"):
+            new_db.execute("INSERT INTO addresses (user_id, addr) VALUES (?, ?)",
+                          (user["id"], user["address"]))
+    # BUG: No commit call
+    print(f"Migrated {len(users)} users")''',
        "lines_changed": 11, "has_tests": False,
        "bug_category": "missing_error_handling", "ground_truth_severity": "medium",
        "bug_lines": [4, 10], "human_labels": ["medium", "high", "medium"],
        "human_agreement": 0.67, "cohen_kappa": 0.55,
    },
    {
        "pr_id": "PR-045", "title": "Refactor logging configuration",
        "description": "Centralized logging setup with structured JSON output.",
        "author_experience": "senior", "language": "python", "filename": "core/logging.py",
        "diff": '''@@ -1,12 +1,12 @@
-import logging
-logging.basicConfig(level=logging.DEBUG)
-logger = logging.getLogger(__name__)
+import logging, json, sys
+def setup_logging(level="INFO"):
+    handler = logging.StreamHandler(sys.stdout)
+    handler.setFormatter(JsonFormatter())
+    root = logging.getLogger()
+    root.setLevel(getattr(logging, level))
+    root.addHandler(handler)
+class JsonFormatter(logging.Formatter):
+    def format(self, record):
+        return json.dumps({"level": record.levelname,
+                           "msg": record.getMessage(),
+                           "time": self.formatTime(record)})''',
        "lines_changed": 12, "has_tests": True,
        "bug_category": "style_only", "ground_truth_severity": "none",
        "bug_lines": [], "human_labels": ["none", "none", "none"],
        "human_agreement": 1.0, "cohen_kappa": 1.0,
    },
    {
        "pr_id": "PR-046", "title": "Add GraphQL resolver for orders",
        "description": "GraphQL resolvers for order queries with nested product lookups.",
        "author_experience": "mid", "language": "typescript", "filename": "src/resolvers/orders.ts",
        "diff": '''@@ -1,0 +1,16 @@
+export const orderResolvers = {
+    Query: {
+        orders: async (_, { userId }) => {
+            // BUG: No authorization check — any user can query any user's orders
+            return db.orders.findMany({ where: { userId } });
+        },
+    },
+    Order: {
+        products: async (order) => {
+            // BUG: N+1 query — fetches products one by one per order
+            return Promise.all(order.productIds.map(id => db.products.findUnique({ where: { id } })));
+        },
+    },
+};''',
        "lines_changed": 14, "has_tests": False,
        "bug_category": "security_vulnerability", "ground_truth_severity": "critical",
        "bug_lines": [4, 10], "human_labels": ["critical", "high", "critical"],
        "human_agreement": 0.67, "cohen_kappa": 0.65,
    },
    {
        "pr_id": "PR-047", "title": "Add password reset flow",
        "description": "Password reset via email with token generation and validation.",
        "author_experience": "junior", "language": "python", "filename": "auth/password_reset.py",
        "diff": '''@@ -1,0 +1,14 @@
+import random, string, time
+tokens = {}
+def create_reset_token(email):
+    # BUG: Weak random — predictable token
+    token = ''.join(random.choices(string.ascii_letters, k=20))
+    tokens[token] = {"email": email, "created": time.time()}
+    return token
+def reset_password(token, new_password):
+    data = tokens.get(token)
+    if not data:
+        return False
+    # BUG: No token expiry check
+    # BUG: No password strength validation
+    db.update_password(data["email"], new_password)
+    return True''',
        "lines_changed": 15, "has_tests": False,
        "bug_category": "security_vulnerability", "ground_truth_severity": "critical",
        "bug_lines": [4, 12, 13], "human_labels": ["critical", "critical", "critical"],
        "human_agreement": 1.0, "cohen_kappa": 1.0,
    },
    {
        "pr_id": "PR-048", "title": "Optimize database indexes",
        "description": "Added composite indexes for common query patterns.",
        "author_experience": "senior", "language": "python", "filename": "migrations/add_indexes.py",
        "diff": '''@@ -1,0 +1,10 @@
+def upgrade():
+    # These are pure schema improvements — no bugs
+    op.create_index("idx_orders_user_date", "orders", ["user_id", "created_at"])
+    op.create_index("idx_products_category", "products", ["category_id", "is_active"])
+    op.create_index("idx_sessions_token", "sessions", ["token"], unique=True)
+
+def downgrade():
+    op.drop_index("idx_orders_user_date")
+    op.drop_index("idx_products_category")
+    op.drop_index("idx_sessions_token")''',
        "lines_changed": 10, "has_tests": True,
        "bug_category": "style_only", "ground_truth_severity": "none",
        "bug_lines": [], "human_labels": ["none", "none", "none"],
        "human_agreement": 1.0, "cohen_kappa": 1.0,
    },
    {
        "pr_id": "PR-049", "title": "Add event-driven notification system",
        "description": "Pub/sub event system for triggering notifications across services.",
        "author_experience": "mid", "language": "python", "filename": "events/dispatcher.py",
        "diff": '''@@ -1,0 +1,18 @@
+class EventDispatcher:
+    def __init__(self):
+        self.handlers = {}
+    def subscribe(self, event_type, handler):
+        self.handlers.setdefault(event_type, []).append(handler)
+    def dispatch(self, event_type, data):
+        for handler in self.handlers.get(event_type, []):
+            # BUG: No error isolation — one handler failure stops all
+            handler(data)
+    def dispatch_async(self, event_type, data):
+        import threading
+        for handler in self.handlers.get(event_type, []):
+            # BUG: Unbounded thread creation — no pool
+            t = threading.Thread(target=handler, args=(data,))
+            t.start()
+            # BUG: No join — threads left dangling''',
        "lines_changed": 16, "has_tests": False,
        "bug_category": "missing_error_handling", "ground_truth_severity": "medium",
        "bug_lines": [8, 13, 16], "human_labels": ["medium", "medium", "high"],
        "human_agreement": 0.67, "cohen_kappa": 0.60,
    },
    {
        "pr_id": "PR-050", "title": "Add data validation pipeline",
        "description": "Schema validation for incoming API payloads with custom rules.",
        "author_experience": "junior", "language": "python", "filename": "validation/pipeline.py",
        "diff": '''@@ -1,0 +1,18 @@
+def validate_payload(data, schema):
+    errors = []
+    for field, rules in schema.items():
+        value = data.get(field)
+        if rules.get("required") and value is None:
+            errors.append(f"{field} is required")
+            continue
+        if rules.get("type") and not isinstance(value, rules["type"]):
+            # BUG: isinstance check fails when value is None (already checked above)
+            errors.append(f"{field} must be {rules['type'].__name__}")
+        if rules.get("max_length") and len(value) > rules["max_length"]:
+            # BUG: len() on None crashes — need null check first
+            errors.append(f"{field} exceeds max length")
+    return errors''',
        "lines_changed": 14, "has_tests": False,
        "bug_category": "null_pointer", "ground_truth_severity": "high",
        "bug_lines": [9, 12], "human_labels": ["high", "medium", "high"],
        "human_agreement": 0.67, "cohen_kappa": 0.60,
    },
]


def _build_pr_file(template: Dict) -> PRFile:
    """Convert a template dict to a PRFile model."""
    return PRFile(
        filename=template["filename"],
        language=template["language"],
        diff=template["diff"],
        lines_changed=template["lines_changed"],
        has_tests=template["has_tests"],
    )


def _build_observation(
    template: Dict,
    step_number: int,
    episode_budget: int,
    review_queue: List[str],
    existing_comments: Optional[List[str]] = None,
) -> Observation:
    """Convert a template dict to a full Observation."""
    return Observation(
        pr_id=template["pr_id"],
        title=template["title"],
        description=template["description"],
        author_experience=template["author_experience"],
        files=[_build_pr_file(template)],
        existing_comments=existing_comments or [],
        review_queue=review_queue,
        step_number=step_number,
        episode_budget=episode_budget,
    )


def get_ground_truth(pr_id: str) -> Dict:
    """
    Get ground truth for a PR by its ID.

    Returns dict with: bug_category, ground_truth_severity, bug_lines,
    human_labels, human_agreement, cohen_kappa.

    Used by graders for deterministic scoring.
    """
    for t in PR_TEMPLATES:
        if t["pr_id"] == pr_id:
            return {
                "bug_category": t["bug_category"],
                "ground_truth_severity": t["ground_truth_severity"],
                "bug_lines": t["bug_lines"],
                "human_labels": t["human_labels"],
                "human_agreement": t["human_agreement"],
                "cohen_kappa": t["cohen_kappa"],
            }
    raise ValueError(f"Unknown PR ID: {pr_id}")


def get_template_by_id(pr_id: str) -> Dict:
    """Get full template dict by PR ID."""
    for t in PR_TEMPLATES:
        if t["pr_id"] == pr_id:
            return t
    raise ValueError(f"Unknown PR ID: {pr_id}")


class DataGenerator:
    """
    Generates episodes of PRs for each task difficulty.

    Uses FIXED_TEST_SUITE: all 20 pre-generated PR templates.
    Randomness affects only ordering within episodes, not PR content.
    This ensures evaluation is deterministic given the same seed.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.all_templates = list(PR_TEMPLATES)

    def generate_easy_episode(self, episode_length: int = 5) -> List[Dict]:
        """
        Generate an episode for easy task: sequence of individual PRs.

        Returns list of templates (one per step). The agent must label
        each PR's severity.
        """
        # Shuffle deterministically, pick episode_length PRs
        pool = list(self.all_templates)
        self.rng.shuffle(pool)
        return pool[:episode_length]

    def generate_medium_episode(self, num_queues: int = 3, queue_size: int = 5) -> List[List[Dict]]:
        """
        Generate an episode for medium task: sequence of PR queues.

        Returns list of queues, each queue is a list of templates.
        The agent must order each queue by priority.
        """
        pool = list(self.all_templates)
        self.rng.shuffle(pool)

        queues = []
        for i in range(num_queues):
            start = (i * queue_size) % len(pool)
            queue = []
            for j in range(queue_size):
                idx = (start + j) % len(pool)
                queue.append(pool[idx])
            # Shuffle within queue so agent can't rely on ordering
            self.rng.shuffle(queue)
            queues.append(queue)

        return queues

    def generate_hard_episode(self, num_prs: int = 3) -> List[Dict]:
        """
        Generate an episode for hard task: PRs requiring detailed review.

        Returns list of templates. For each PR, the agent may make
        multiple add_comment actions before approve/request_changes.
        Prioritize PRs with bugs for more interesting review scenarios.
        """
        # Select PRs with a mix of severities — ensure at least one critical
        critical = [t for t in self.all_templates if t["ground_truth_severity"] == "critical"]
        non_critical = [t for t in self.all_templates if t["ground_truth_severity"] != "critical"]

        self.rng.shuffle(critical)
        self.rng.shuffle(non_critical)

        selected = []
        if critical:
            selected.append(critical[0])
        remaining_needed = num_prs - len(selected)
        selected.extend(non_critical[:remaining_needed])

        self.rng.shuffle(selected)
        return selected[:num_prs]

    def compute_priority_order(self, queue: List[Dict]) -> List[str]:
        """
        Compute ground truth priority ordering for a queue of PRs.

        Priority rules (in order):
        1. Security PRs always first
        2. By severity: critical > high > medium > low > none
        3. Within same severity: junior authors first (they need review most urgently)
        """
        severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}
        experience_rank = {"junior": 0, "mid": 1, "senior": 2}

        def sort_key(template):
            is_security = 1 if template["bug_category"] in ("security_vulnerability", "sql_injection") else 0
            sev = severity_rank.get(template["ground_truth_severity"], 4)
            exp = experience_rank.get(template["author_experience"], 2)
            # Lower = higher priority. Security first, then severity, then experience
            return (1 - is_security, sev, exp)

        sorted_queue = sorted(queue, key=sort_key)
        return [t["pr_id"] for t in sorted_queue]
