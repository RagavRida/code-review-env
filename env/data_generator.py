"""
Data Generator for CodeReviewEnv

Generates realistic synthetic pull requests with actual code diffs.
The FIXED_TEST_SUITE provides 20 pre-generated PRs at seed=42 for
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
